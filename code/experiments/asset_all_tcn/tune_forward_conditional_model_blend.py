from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from market_mean_ts_model import score_time_blocks
from walk_forward_tabular import weighted_zero_mean_r2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仅用 calibration 前半段选择市场幅度条件融合权重，并冻结到后半段。"
    )
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=Path(
            "results/blend_best_panel_market32_75k_cal20k/"
            "best_blend_calibration_predictions.csv"
        ),
    )
    parser.add_argument(
        "--exact-market-predictions",
        type=Path,
        default=Path("results/exact_market_aggregate_75k_probe/calibration_predictions.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/forward_conditional_exactmarket_blend"),
    )
    parser.add_argument("--cutoff-quantiles", type=float, nargs="+", default=[0.70, 0.80, 0.90])
    parser.add_argument("--weight-min", type=float, default=0.0)
    parser.add_argument("--weight-max", type=float, default=0.40)
    parser.add_argument("--weight-step", type=float, default=0.02)
    return parser.parse_args()


def load_aligned(base_path: Path, exact_path: Path) -> pd.DataFrame:
    columns = ["time_id", "asset_id", "target", "weight", "prediction"]
    base = pd.read_csv(base_path, usecols=columns).rename(
        columns={"prediction": "base_prediction"}
    )
    exact = pd.read_csv(exact_path, usecols=columns).rename(
        columns={"prediction": "exact_prediction"}
    )
    base = base.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    exact = exact.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    if not np.array_equal(
        base[["time_id", "asset_id"]].to_numpy(),
        exact[["time_id", "asset_id"]].to_numpy(),
    ):
        raise ValueError("base 与 exact-market 的键不一致")
    if float(np.max(np.abs(base["target"].to_numpy() - exact["target"].to_numpy()))) > 1e-6:
        raise ValueError("base 与 exact-market 的 target 不一致")
    return pd.DataFrame(
        {
            "time_id": base["time_id"],
            "asset_id": base["asset_id"],
            "target": base["target"],
            "weight": base["weight"],
            "base_prediction": base["base_prediction"],
            "exact_prediction": exact["exact_prediction"],
        }
    )


def conditional_prediction(
    base_prediction: np.ndarray,
    exact_prediction: np.ndarray,
    cutoff: float,
    low_weight: float,
    high_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    high_regime = np.abs(exact_prediction) >= float(cutoff)
    blend_weight = np.where(high_regime, float(high_weight), float(low_weight))
    prediction = (1.0 - blend_weight) * base_prediction + blend_weight * exact_prediction
    return prediction, high_regime


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    frame = load_aligned(args.base_predictions, args.exact_market_predictions)
    y_true = frame["target"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    time_id = frame["time_id"].to_numpy(dtype=np.int64)
    base_prediction = frame["base_prediction"].to_numpy(dtype=np.float64)
    exact_prediction = frame["exact_prediction"].to_numpy(dtype=np.float64)

    unique_times = np.unique(time_id)
    split_time = int(unique_times[len(unique_times) // 2])
    first_mask = time_id < split_time
    second_mask = ~first_mask
    weight_values = np.arange(args.weight_min, args.weight_max + 1e-12, args.weight_step)

    rows = []
    for quantile in args.cutoff_quantiles:
        cutoff = float(np.quantile(np.abs(exact_prediction[first_mask]), quantile))
        for low_weight in weight_values:
            for high_weight in weight_values:
                prediction, high_regime = conditional_prediction(
                    base_prediction,
                    exact_prediction,
                    cutoff,
                    float(low_weight),
                    float(high_weight),
                )
                rows.append(
                    {
                        "cutoff_quantile": float(quantile),
                        "cutoff": cutoff,
                        "low_weight": float(low_weight),
                        "high_weight": float(high_weight),
                        "first_half_score": float(
                            weighted_zero_mean_r2(
                                y_true[first_mask], prediction[first_mask], weight[first_mask]
                            )
                        ),
                        "second_half_score": float(
                            weighted_zero_mean_r2(
                                y_true[second_mask], prediction[second_mask], weight[second_mask]
                            )
                        ),
                        "full_score": float(weighted_zero_mean_r2(y_true, prediction, weight)),
                        "high_regime_row_count": int(high_regime.sum()),
                    }
                )

    candidates = pd.DataFrame(rows)
    # 只按前半段分数选参数，后半段完全不参与模型选择。
    candidates = candidates.sort_values("first_half_score", ascending=False).reset_index(drop=True)
    selected = candidates.iloc[0]
    prediction, high_regime = conditional_prediction(
        base_prediction,
        exact_prediction,
        float(selected["cutoff"]),
        float(selected["low_weight"]),
        float(selected["high_weight"]),
    )

    output = frame.copy()
    output["high_market_regime"] = high_regime.astype(np.int8)
    output["prediction"] = prediction
    output["error"] = output["prediction"] - output["target"]
    candidates.to_csv(args.results_dir / "candidate_metrics.csv", index=False)
    candidates.head(100).to_csv(args.results_dir / "candidate_top100_by_first_half.csv", index=False)
    output.to_csv(args.results_dir / "calibration_predictions.csv", index=False)

    base_first = weighted_zero_mean_r2(
        y_true[first_mask], base_prediction[first_mask], weight[first_mask]
    )
    base_second = weighted_zero_mean_r2(
        y_true[second_mask], base_prediction[second_mask], weight[second_mask]
    )
    base_full = weighted_zero_mean_r2(y_true, base_prediction, weight)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_rule": "parameters selected only by first-half score",
        "base": {
            "full_score": float(base_full),
            "first_half_score": float(base_first),
            "second_half_score": float(base_second),
        },
        "selected": {
            "cutoff_quantile": float(selected["cutoff_quantile"]),
            "cutoff": float(selected["cutoff"]),
            "low_weight": float(selected["low_weight"]),
            "high_weight": float(selected["high_weight"]),
            "full_score": float(weighted_zero_mean_r2(y_true, prediction, weight)),
            "first_half_score": float(
                weighted_zero_mean_r2(
                    y_true[first_mask], prediction[first_mask], weight[first_mask]
                )
            ),
            "second_half_score": float(
                weighted_zero_mean_r2(
                    y_true[second_mask], prediction[second_mask], weight[second_mask]
                )
            ),
            "second_half_improvement_over_base": float(
                weighted_zero_mean_r2(
                    y_true[second_mask], prediction[second_mask], weight[second_mask]
                )
                - base_second
            ),
            **score_time_blocks(y_true, prediction, weight, time_id, 4),
            **score_time_blocks(y_true, prediction, weight, time_id, 8),
        },
        "output_files": {
            "candidate_metrics": str(args.results_dir / "candidate_metrics.csv"),
            "calibration_predictions": str(args.results_dir / "calibration_predictions.csv"),
        },
    }
    with (args.results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
