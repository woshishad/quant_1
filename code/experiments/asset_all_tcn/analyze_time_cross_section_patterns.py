from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from walk_forward_tabular import weighted_zero_mean_r2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析每个 time_id 内 15 个 target 的方向一致性、幅度集中度和预测误差。"
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=Path(
            "results/forward_conditional_exactmarket_blend/calibration_predictions.csv"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/time_cross_section_pattern_analysis"),
    )
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


def weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(values * weight) / max(float(np.sum(weight)), 1e-12))


def weighted_correlation(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    weight_sum = max(float(np.sum(weight)), 1e-12)
    x_mean = float(np.sum(weight * x) / weight_sum)
    y_mean = float(np.sum(weight * y) / weight_sum)
    x_centered = x - x_mean
    y_centered = y - y_mean
    covariance = float(np.sum(weight * x_centered * y_centered))
    x_var = float(np.sum(weight * x_centered**2))
    y_var = float(np.sum(weight * y_centered**2))
    denominator = np.sqrt(max(x_var * y_var, 0.0))
    return covariance / denominator if denominator > 1e-18 else np.nan


def summarize_time(group: pd.DataFrame) -> dict:
    target = group["target"].to_numpy(dtype=np.float64)
    prediction = group["prediction"].to_numpy(dtype=np.float64)
    weight = group["weight"].to_numpy(dtype=np.float64)
    abs_target = np.abs(target)
    positive_count = int(np.sum(target > 0.0))
    negative_count = int(np.sum(target < 0.0))
    target_square = weight * target**2
    squared_error = weight * (target - prediction) ** 2
    denominator = float(np.sum(target_square))
    sse = float(np.sum(squared_error))
    abs_sum = max(float(np.sum(abs_target)), 1e-12)
    sorted_abs = np.sort(abs_target)[::-1]
    top_index = int(np.argmax(abs_target))
    target_weighted_mean = weighted_mean(target, weight)
    prediction_weighted_mean = weighted_mean(prediction, weight)

    # 有效标的数接近 1 表示几乎由一个标的主导，接近 15 表示幅度比较均匀。
    effective_asset_count = abs_sum**2 / max(float(np.sum(abs_target**2)), 1e-12)
    return {
        "time_id": int(group["time_id"].iloc[0]),
        "row_count": int(len(group)),
        "target_mean": float(np.mean(target)),
        "target_weighted_mean": target_weighted_mean,
        "target_std": float(np.std(target)),
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "target_abs_mean": float(np.mean(abs_target)),
        "target_abs_max": float(np.max(abs_target)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_share": positive_count / max(len(target), 1),
        "majority_sign_share": max(positive_count, negative_count) / max(len(target), 1),
        "top1_abs_share": float(sorted_abs[0] / abs_sum),
        "top3_abs_share": float(np.sum(sorted_abs[:3]) / abs_sum),
        "top1_square_share": float(np.max(target_square) / max(denominator, 1e-12)),
        "effective_asset_count": float(effective_asset_count),
        "dominant_asset_id": int(group.iloc[top_index]["asset_id"]),
        "dominant_target": float(target[top_index]),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_weighted_mean": prediction_weighted_mean,
        "prediction_std": float(np.std(prediction)),
        "prediction_abs_mean": float(np.mean(np.abs(prediction))),
        "prediction_abs_max": float(np.max(np.abs(prediction))),
        "magnitude_ratio": float(
            np.mean(np.abs(prediction)) / max(float(np.mean(abs_target)), 1e-12)
        ),
        "weighted_direction_accuracy": float(
            np.sum(weight * (np.sign(target) == np.sign(prediction)))
            / max(float(np.sum(weight)), 1e-12)
        ),
        "market_sign_correct": int(
            np.sign(target_weighted_mean) == np.sign(prediction_weighted_mean)
        ),
        "cross_section_weighted_correlation": weighted_correlation(
            target, prediction, weight
        ),
        "time_r2": float(1.0 - sse / max(denominator, 1e-12)),
        "weighted_target_square": denominator,
        "weighted_squared_error": sse,
        "score_numerator": float(denominator - sse),
    }


def classify_pattern(row: pd.Series) -> str:
    if row["positive_count"] >= 12:
        direction = "strong_positive_consensus"
    elif row["positive_count"] <= 3:
        direction = "strong_negative_consensus"
    elif 6 <= row["positive_count"] <= 9:
        direction = "mixed_balanced"
    elif row["positive_count"] >= 10:
        direction = "moderate_positive_consensus"
    else:
        direction = "moderate_negative_consensus"

    if row["top1_abs_share"] >= 0.30:
        concentration = "one_asset_dominant"
    elif row["top3_abs_share"] >= 0.60:
        concentration = "few_assets_dominant"
    else:
        concentration = "diffuse"
    return f"{direction}|{concentration}"


def select_representative_times(summary: pd.DataFrame) -> list[tuple[str, int]]:
    selected: list[tuple[str, int]] = []
    used: set[int] = set()

    def add(label: str, candidates: pd.DataFrame, sort_column: str, ascending: bool) -> None:
        ordered = candidates.sort_values(sort_column, ascending=ascending)
        for time_id in ordered["time_id"].astype(int):
            if time_id not in used:
                selected.append((label, time_id))
                used.add(time_id)
                return

    # 最大总评分损失比纯 time R² 更合理，避免选择 target 本来就接近 0 的时间点。
    add("largest_score_damage", summary, "score_numerator", True)
    add(
        "strong_common_positive",
        summary[summary["positive_count"] >= 12],
        "target_weighted_mean",
        False,
    )
    add(
        "strong_common_negative",
        summary[summary["positive_count"] <= 3],
        "target_weighted_mean",
        True,
    )
    add(
        "one_asset_dominant",
        summary[summary["top1_abs_share"] >= 0.30],
        "weighted_target_square",
        False,
    )
    add(
        "mixed_high_dispersion",
        summary[(summary["positive_count"] >= 6) & (summary["positive_count"] <= 9)],
        "target_std",
        False,
    )
    magnitude_fail = summary[
        (summary["market_sign_correct"] == 1)
        & (summary["magnitude_ratio"] < 0.10)
        & (np.abs(summary["target_weighted_mean"]) >= np.abs(summary["target_weighted_mean"]).quantile(0.90))
    ]
    add("direction_right_magnitude_small", magnitude_fail, "magnitude_ratio", True)
    return selected


def pattern_summary(frame: pd.DataFrame, time_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pattern, group in time_summary.groupby("pattern", sort=False):
        time_ids = set(group["time_id"].astype(int))
        mask = frame["time_id"].isin(time_ids).to_numpy()
        rows.append(
            {
                "pattern": pattern,
                "time_count": int(len(group)),
                "time_share": float(len(group) / len(time_summary)),
                "mean_positive_count": float(group["positive_count"].mean()),
                "mean_top1_abs_share": float(group["top1_abs_share"].mean()),
                "mean_magnitude_ratio": float(group["magnitude_ratio"].mean()),
                "market_sign_accuracy": float(group["market_sign_correct"].mean()),
                "row_weighted_r2": float(
                    weighted_zero_mean_r2(
                        frame.loc[mask, "target"].to_numpy(dtype=np.float64),
                        frame.loc[mask, "prediction"].to_numpy(dtype=np.float64),
                        frame.loc[mask, "weight"].to_numpy(dtype=np.float64),
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("time_count", ascending=False)


def plot_representative_times(
    rows: pd.DataFrame,
    selected: list[tuple[str, int]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    for axis, (label, time_id) in zip(axes.ravel(), selected):
        group = rows[rows["time_id"] == time_id].sort_values("asset_id")
        x = np.arange(len(group))
        axis.bar(x - 0.2, group["target"], width=0.4, label="target")
        axis.bar(x + 0.2, group["prediction"], width=0.4, label="prediction")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(group["asset_id"].astype(int))
        axis.set_title(f"{label}\ntime_id={time_id}")
        axis.set_xlabel("asset_id")
        axis.set_ylabel("value")
    axes.ravel()[0].legend()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_positive_count_distribution(summary: pd.DataFrame, output_path: Path) -> None:
    counts = summary["positive_count"].value_counts().sort_index()
    plt.figure(figsize=(10, 4.5))
    plt.bar(counts.index, counts.values, color="#2563eb")
    plt.xticks(range(16))
    plt.xlabel("positive target count among 15 assets")
    plt.ylabel("time_id count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_structure_scatter(summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        summary["majority_sign_share"],
        summary["top1_abs_share"],
        c=np.clip(summary["time_r2"], -0.5, 0.5),
        s=8,
        alpha=0.35,
        cmap="coolwarm",
        vmin=-0.5,
        vmax=0.5,
    )
    plt.colorbar(scatter, label="time-level R2 (clipped)")
    plt.xlabel("majority sign share")
    plt.ylabel("largest |target| share")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        args.prediction_file,
        usecols=["time_id", "asset_id", "target", "weight", "prediction"],
    )
    frame = frame.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)

    summaries = [summarize_time(group) for _, group in frame.groupby("time_id", sort=True)]
    time_summary = pd.DataFrame(summaries)
    time_summary["pattern"] = time_summary.apply(classify_pattern, axis=1)
    total_denominator = max(float(time_summary["weighted_target_square"].sum()), 1e-12)
    time_summary["global_score_contribution"] = time_summary["score_numerator"] / total_denominator

    selected = select_representative_times(time_summary)
    selected_map = {time_id: label for label, time_id in selected}
    representative = frame[frame["time_id"].isin(selected_map)].copy()
    representative["case_label"] = representative["time_id"].map(selected_map)
    representative = representative.merge(
        time_summary[
            [
                "time_id",
                "positive_count",
                "target_weighted_mean",
                "target_std",
                "top1_abs_share",
                "magnitude_ratio",
                "time_r2",
                "pattern",
            ]
        ],
        on="time_id",
        how="left",
    )

    sign_distribution = (
        time_summary.groupby("positive_count", sort=True)
        .agg(
            time_count=("time_id", "size"),
            mean_target_weighted_mean=("target_weighted_mean", "mean"),
            mean_target_std=("target_std", "mean"),
            mean_top1_abs_share=("top1_abs_share", "mean"),
            mean_time_r2=("time_r2", "mean"),
        )
        .reset_index()
    )
    patterns = pattern_summary(frame, time_summary)

    time_summary.to_csv(args.results_dir / "time_cross_section_summary.csv", index=False)
    representative.to_csv(args.results_dir / "representative_time_predictions.csv", index=False)
    sign_distribution.to_csv(args.results_dir / "positive_count_distribution.csv", index=False)
    patterns.to_csv(args.results_dir / "pattern_summary.csv", index=False)
    plot_representative_times(
        frame, selected, args.results_dir / "representative_time_target_vs_prediction.png"
    )
    plot_positive_count_distribution(
        time_summary, args.results_dir / "positive_count_distribution.png"
    )
    plot_structure_scatter(
        time_summary, args.results_dir / "target_structure_vs_time_r2.png"
    )

    positive_majority = float((time_summary["positive_count"] >= 8).mean())
    negative_majority = float((time_summary["positive_count"] <= 7).mean())
    strong_consensus = float(
        ((time_summary["positive_count"] >= 12) | (time_summary["positive_count"] <= 3)).mean()
    )
    one_dominant = float((time_summary["top1_abs_share"] >= 0.30).mean())
    few_dominant = float((time_summary["top3_abs_share"] >= 0.60).mean())
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prediction_file": str(args.prediction_file),
        "row_count": int(len(frame)),
        "time_count": int(len(time_summary)),
        "overall_r2": float(
            weighted_zero_mean_r2(
                frame["target"].to_numpy(dtype=np.float64),
                frame["prediction"].to_numpy(dtype=np.float64),
                frame["weight"].to_numpy(dtype=np.float64),
            )
        ),
        "cross_section_structure": {
            "positive_majority_time_share": positive_majority,
            "negative_majority_time_share": negative_majority,
            "strong_sign_consensus_time_share": strong_consensus,
            "all_positive_time_share": float((time_summary["positive_count"] == 15).mean()),
            "all_negative_time_share": float((time_summary["positive_count"] == 0).mean()),
            "one_asset_dominant_time_share_top1_ge_30pct": one_dominant,
            "few_assets_dominant_time_share_top3_ge_60pct": few_dominant,
            "top1_abs_share_quantiles": {
                "p50": float(time_summary["top1_abs_share"].quantile(0.50)),
                "p90": float(time_summary["top1_abs_share"].quantile(0.90)),
                "p95": float(time_summary["top1_abs_share"].quantile(0.95)),
                "p99": float(time_summary["top1_abs_share"].quantile(0.99)),
            },
            "effective_asset_count_mean": float(time_summary["effective_asset_count"].mean()),
            "market_sign_accuracy": float(time_summary["market_sign_correct"].mean()),
            "median_prediction_to_target_magnitude_ratio": float(
                time_summary["magnitude_ratio"].median()
            ),
        },
        "representative_times": [
            {
                "label": label,
                **time_summary[time_summary["time_id"] == time_id]
                .iloc[0]
                .to_dict(),
            }
            for label, time_id in selected
        ],
        "output_files": {
            "time_summary": str(args.results_dir / "time_cross_section_summary.csv"),
            "representative_predictions": str(
                args.results_dir / "representative_time_predictions.csv"
            ),
            "pattern_summary": str(args.results_dir / "pattern_summary.csv"),
            "representative_plot": str(
                args.results_dir / "representative_time_target_vs_prediction.png"
            ),
            "positive_count_plot": str(
                args.results_dir / "positive_count_distribution.png"
            ),
            "structure_scatter": str(
                args.results_dir / "target_structure_vs_time_r2.png"
            ),
        },
    }
    with (args.results_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=json_default)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
