from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tune_regularized_conditional_shrink import (
    apply_scales,
    fit_raw_scales,
    regularize_scales,
)
from walk_forward_tabular import (
    apply_shrink,
    calibrate_shrink_info,
    weighted_zero_mean_r2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="逐块前向验证专家融合与正则条件 shrink，禁止使用未来块 target。"
    )
    parser.add_argument(
        "--input-predictions",
        type=Path,
        default=Path(
            "results/asset_all_market_regime_experts_75k_probe/"
            "calibration_predictions.csv"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/regularized_conditional_walk_forward"),
    )
    parser.add_argument("--beta", type=float, default=-0.08)
    parser.add_argument("--regime-quantile", type=float, default=0.80)
    parser.add_argument("--regularization-strength", type=float, default=0.50)
    parser.add_argument("--scale-floor", type=float, default=0.75)
    parser.add_argument("--scale-cap", type=float, default=1.25)
    parser.add_argument("--shrink-cap", type=float, default=1.6)
    parser.add_argument("--block-counts", type=int, nargs="+", default=[4, 8])
    return parser.parse_args()


def evaluate_blocks(frame: pd.DataFrame, block_count: int, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    y_true = frame["target"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    asset_id = frame["asset_id"].to_numpy(dtype=np.int64)
    time_id = frame["time_id"].to_numpy(dtype=np.int64)
    base_prediction = frame["base_prediction"].to_numpy(dtype=np.float64)
    regime = frame["market_prediction"].to_numpy(dtype=np.float64)
    raw_prediction = base_prediction + float(args.beta) * regime

    time_blocks = [chunk for chunk in np.array_split(np.unique(time_id), block_count) if len(chunk)]
    rows = []
    evaluated_masks = []
    evaluated_base = []
    evaluated_plain = []
    evaluated_conditional = []

    # 第一个块只做 warm-up；从第二块开始，所有校准参数只能用更早的时间块拟合。
    for block_index in range(1, len(time_blocks)):
        history_end = int(time_blocks[block_index - 1][-1])
        test_start = int(time_blocks[block_index][0])
        test_end = int(time_blocks[block_index][-1])
        history_mask = time_id <= history_end
        test_mask = (time_id >= test_start) & (time_id <= test_end)

        shrink_info = calibrate_shrink_info(
            y_true[history_mask],
            raw_prediction[history_mask],
            weight[history_mask],
            asset_id[history_mask],
            "per_asset",
            float(args.shrink_cap),
        )
        history_plain = apply_shrink(
            raw_prediction[history_mask], asset_id[history_mask], shrink_info
        )
        test_plain = apply_shrink(raw_prediction[test_mask], asset_id[test_mask], shrink_info)

        edges, raw_scales = fit_raw_scales(
            y_true[history_mask],
            history_plain,
            weight[history_mask],
            regime[history_mask],
            [0.0, float(args.regime_quantile), 1.0],
        )
        scales = regularize_scales(
            raw_scales,
            float(args.regularization_strength),
            float(args.scale_floor),
            float(args.scale_cap),
        )
        test_conditional, _ = apply_scales(test_plain, regime[test_mask], edges, scales)

        base_score = weighted_zero_mean_r2(
            y_true[test_mask], base_prediction[test_mask], weight[test_mask]
        )
        plain_score = weighted_zero_mean_r2(
            y_true[test_mask], test_plain, weight[test_mask]
        )
        conditional_score = weighted_zero_mean_r2(
            y_true[test_mask], test_conditional, weight[test_mask]
        )
        rows.append(
            {
                "block_count": int(block_count),
                "eval_block": int(block_index + 1),
                "history_time_min": int(np.min(time_id[history_mask])),
                "history_time_max": history_end,
                "test_time_min": test_start,
                "test_time_max": test_end,
                "rows": int(test_mask.sum()),
                "base_score": float(base_score),
                "plain_score": float(plain_score),
                "conditional_score": float(conditional_score),
                "conditional_minus_base": float(conditional_score - base_score),
                "conditional_minus_plain": float(conditional_score - plain_score),
                "low_regime_scale": float(scales[0]),
                "high_regime_scale": float(scales[1]),
                "regime_cutoff": float(edges[1]),
            }
        )
        evaluated_masks.append(np.flatnonzero(test_mask))
        evaluated_base.append(base_prediction[test_mask])
        evaluated_plain.append(test_plain)
        evaluated_conditional.append(test_conditional)

    row_frame = pd.DataFrame(rows)
    indices = np.concatenate(evaluated_masks)
    base_all = np.concatenate(evaluated_base)
    plain_all = np.concatenate(evaluated_plain)
    conditional_all = np.concatenate(evaluated_conditional)
    summary = {
        "block_count": int(block_count),
        "evaluated_rows": int(len(indices)),
        "evaluated_time_min": int(np.min(time_id[indices])),
        "evaluated_time_max": int(np.max(time_id[indices])),
        "base_score": float(weighted_zero_mean_r2(y_true[indices], base_all, weight[indices])),
        "plain_score": float(weighted_zero_mean_r2(y_true[indices], plain_all, weight[indices])),
        "conditional_score": float(
            weighted_zero_mean_r2(y_true[indices], conditional_all, weight[indices])
        ),
        "mean_block_conditional_minus_base": float(row_frame["conditional_minus_base"].mean()),
        "min_block_conditional_minus_base": float(row_frame["conditional_minus_base"].min()),
        "positive_block_count": int((row_frame["conditional_minus_base"] > 0.0).sum()),
        "eval_block_count": int(len(row_frame)),
    }
    return row_frame, summary


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        args.input_predictions,
        usecols=[
            "time_id",
            "asset_id",
            "target",
            "weight",
            "base_prediction",
            "market_prediction",
        ],
    )
    frame = frame.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)

    block_frames = []
    summaries = []
    for block_count in args.block_counts:
        block_frame, summary = evaluate_blocks(frame, int(block_count), args)
        block_frames.append(block_frame)
        summaries.append(summary)

    all_blocks = pd.concat(block_frames, ignore_index=True)
    all_blocks.to_csv(args.results_dir / "walk_forward_block_scores.csv", index=False)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_predictions": str(args.input_predictions),
        "parameters": {
            "beta": float(args.beta),
            "regime_quantile": float(args.regime_quantile),
            "regularization_strength": float(args.regularization_strength),
            "scale_floor": float(args.scale_floor),
            "scale_cap": float(args.scale_cap),
        },
        "summaries": summaries,
        "output_file": str(args.results_dir / "walk_forward_block_scores.csv"),
    }
    with (args.results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
