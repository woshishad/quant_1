from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from walk_forward_tabular import (
    apply_shrink,
    calibrate_shrink_info,
    score_candidate_on_calibration,
    summarize_shrink_info,
    weighted_zero_mean_r2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="融合多个 calibration prediction 文件，搜索线性权重和 shrink。"
    )
    parser.add_argument("--prediction-files", type=Path, nargs="+", required=True)
    parser.add_argument("--prediction-names", type=str, nargs="+", required=True)
    parser.add_argument(
        "--prediction-columns",
        type=str,
        nargs="+",
        default=None,
        help="每个文件使用哪一列作为预测；不传则全部使用 prediction。"
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--shrink-mode", choices=["global", "per_asset"], default="global")
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[1.0, 1.2, 1.4, 1.6])
    parser.add_argument("--candidate-score-mode", choices=["full", "mean_halves", "min_halves"], default="min_halves")
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def make_results_dir(path: Path | None) -> Path:
    if path is not None:
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"calibration_prediction_blend_{timestamp}"


def score_time_blocks(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    block_count: int,
) -> dict[str, float]:
    """按连续时间块评估稳定性，避免只看全局分数造成误判。"""
    unique_times = np.unique(time_id)
    chunks = [chunk for chunk in np.array_split(unique_times, block_count) if len(chunk) > 0]
    scores = []
    for chunk in chunks:
        mask = (time_id >= int(chunk[0])) & (time_id <= int(chunk[-1]))
        scores.append(float(weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])))
    values = np.asarray(scores, dtype=np.float64)
    return {
        f"block{block_count}_mean_score": float(np.mean(values)),
        f"block{block_count}_min_score": float(np.min(values)),
        f"block{block_count}_last_score": float(values[-1]),
        f"block{block_count}_negative_count": int(np.sum(values < 0.0)),
    }


def simplex_weights(count: int, step: float) -> list[tuple[float, ...]]:
    """生成非负且和为 1 的权重网格；模型数较少时这个穷举足够直接。"""
    units = int(round(1.0 / float(step)))
    if not np.isclose(units * float(step), 1.0):
        raise ValueError("--weight-step 必须能整除 1，例如 0.1/0.05/0.025")
    rows = []
    for cuts in itertools.product(range(units + 1), repeat=count):
        if sum(cuts) == units:
            rows.append(tuple(float(value) / float(units) for value in cuts))
    return rows


def load_prediction_file(path: Path, name: str, prediction_column: str) -> pd.DataFrame:
    """读取单个预测文件，并统一成 merge 需要的列名。"""
    use_columns = ["time_id", "asset_id", "target", "weight", prediction_column]
    frame = pd.read_csv(path, usecols=use_columns)
    frame = frame.rename(columns={prediction_column: f"prediction_{name}"})
    frame = frame.sort_values(["time_id", "asset_id"]).reset_index(drop=True)
    return frame


def merge_prediction_files(
    files: list[Path],
    names: list[str],
    columns: list[str],
) -> pd.DataFrame:
    """按 time_id/asset_id 对齐多个模型预测，并校验 target/weight 一致。"""
    merged = None
    for index, (path, name, column) in enumerate(zip(files, names, columns)):
        frame = load_prediction_file(path, name, column)
        if merged is None:
            merged = frame
            continue

        suffix = f"_{name}"
        merged = merged.merge(
            frame,
            on=["time_id", "asset_id"],
            how="inner",
            suffixes=("", suffix),
        )
        target_diff = np.max(np.abs(merged["target"].to_numpy() - merged[f"target{suffix}"].to_numpy()))
        weight_diff = np.max(np.abs(merged["weight"].to_numpy() - merged[f"weight{suffix}"].to_numpy()))
        if target_diff > 1e-6 or weight_diff > 1e-6:
            raise ValueError(f"{name} 与第一个预测文件的 target/weight 不一致")
        merged = merged.drop(columns=[f"target{suffix}", f"weight{suffix}"])

    if merged is None or merged.empty:
        raise ValueError("没有读到可融合的预测行")
    return merged


def plot_weight_bar(best: dict, names: list[str], output_path: Path) -> None:
    """保存最佳融合权重图，便于肉眼检查最终主要依赖哪些模型。"""
    weights = [float(best[f"weight_{name}"]) for name in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, weights, color="#3b82f6")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Blend weight")
    plt.title("Best calibration blend weights")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    if len(args.prediction_files) != len(args.prediction_names):
        raise ValueError("--prediction-files 和 --prediction-names 数量必须一致")
    prediction_columns = args.prediction_columns or ["prediction"] * len(args.prediction_files)
    if len(prediction_columns) != len(args.prediction_files):
        raise ValueError("--prediction-columns 数量必须和 --prediction-files 一致")

    args.results_dir = make_results_dir(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_prediction_files(args.prediction_files, args.prediction_names, prediction_columns)
    y_true = merged["target"].to_numpy(dtype=np.float64)
    sample_weight = merged["weight"].to_numpy(dtype=np.float64)
    asset_id = merged["asset_id"].to_numpy(dtype=np.int64)
    time_id = merged["time_id"].to_numpy(dtype=np.int64)
    prediction_matrix = np.column_stack(
        [merged[f"prediction_{name}"].to_numpy(dtype=np.float64) for name in args.prediction_names]
    )

    rows = []
    best = {"selection_score": -np.inf}
    for weights in simplex_weights(len(args.prediction_names), args.weight_step):
        raw_prediction = prediction_matrix @ np.asarray(weights, dtype=np.float64)
        for cap in args.shrink_cap_candidates:
            shrink_info = calibrate_shrink_info(
                y_true,
                raw_prediction,
                sample_weight,
                asset_id,
                args.shrink_mode,
                float(cap),
            )
            prediction = apply_shrink(raw_prediction, asset_id, shrink_info)
            score_info = score_candidate_on_calibration(
                y_true,
                prediction,
                sample_weight,
                time_id,
                args.candidate_score_mode,
            )
            shrink_summary = summarize_shrink_info(shrink_info)
            row = {
                "selection_score": float(score_info["selection_score"]),
                "full_score": float(score_info["full_score"]),
                "first_half_score": float(score_info["first_half_score"]),
                "second_half_score": float(score_info["second_half_score"]),
                "shrink_mode": args.shrink_mode,
                "shrink_cap": float(cap),
                "shrink": float(shrink_summary["cal_shrink"]),
                "shrink_min": float(shrink_summary["cal_shrink_min"]),
                "shrink_mean": float(shrink_summary["cal_shrink_mean"]),
                "shrink_max": float(shrink_summary["cal_shrink_max"]),
                "prediction_std": float(np.std(prediction)),
                "raw_prediction_std": float(np.std(raw_prediction)),
                "shrink_info": json.dumps(shrink_info, ensure_ascii=False, default=json_default),
            }
            row.update({f"weight_{name}": float(weight) for name, weight in zip(args.prediction_names, weights)})
            row.update(score_time_blocks(y_true, prediction, sample_weight, time_id, 4))
            row.update(score_time_blocks(y_true, prediction, sample_weight, time_id, 8))
            rows.append(row)
            if row["selection_score"] > float(best["selection_score"]):
                best = {**row, "_prediction": prediction, "_raw_prediction": raw_prediction}

    metrics = pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    metrics.to_csv(args.results_dir / "blend_candidate_metrics.csv", index=False)
    metrics.head(50).to_csv(args.results_dir / "blend_top50_candidates.csv", index=False)

    output_predictions = merged[["time_id", "asset_id", "target", "weight"]].copy()
    output_predictions["raw_prediction"] = np.asarray(best["_raw_prediction"], dtype=np.float32)
    output_predictions["prediction"] = np.asarray(best["_prediction"], dtype=np.float32)
    output_predictions["error"] = output_predictions["prediction"] - output_predictions["target"]
    output_predictions.to_csv(args.results_dir / "best_blend_calibration_predictions.csv", index=False)

    best_payload = {key: value for key, value in best.items() if not key.startswith("_")}
    summary = {
        "leakage_safe": True,
        "official_test_used": False,
        "row_count": int(len(merged)),
        "time_min": int(np.min(time_id)),
        "time_max": int(np.max(time_id)),
        "prediction_files": [str(path) for path in args.prediction_files],
        "prediction_names": args.prediction_names,
        "prediction_columns": prediction_columns,
        "weight_step": float(args.weight_step),
        "best_candidate": best_payload,
        "output_files": {
            "candidate_metrics": str(args.results_dir / "blend_candidate_metrics.csv"),
            "top50": str(args.results_dir / "blend_top50_candidates.csv"),
            "best_predictions": str(args.results_dir / "best_blend_calibration_predictions.csv"),
            "weight_plot": str(args.results_dir / "best_blend_weights.png"),
        },
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    plot_weight_bar(best_payload, args.prediction_names, args.results_dir / "best_blend_weights.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
