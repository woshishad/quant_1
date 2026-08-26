from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from walk_forward_tabular import weighted_zero_mean_r2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单独分析 raw train target 的分布和当前预测误差结构。")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=Path("results/blend_best_neutralized_with_panel_75k_cal20k/best_blend_calibration_predictions.csv"),
        help="可选：带 target/prediction 的 calibration 预测文件，用于分析误差随 target 分布的变化。",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/target_distribution_profile"))
    parser.add_argument("--example-time-id", type=int, default=868480)
    parser.add_argument("--rolling-window", type=int, default=2000)
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


def train_parquet_paths(raw_data_dir: Path) -> list[Path]:
    paths = sorted((raw_data_dir / "train").glob("train_partition_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"找不到 raw train parquet: {raw_data_dir / 'train'}")
    return paths


def read_target_frame(raw_data_dir: Path) -> pd.DataFrame:
    """只读取分布分析需要的 4 列，避免把 300 多个 feature 全读进内存。"""
    paths = train_parquet_paths(raw_data_dir)
    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    table = dataset.to_table(columns=["time_id", "asset_id", "weight", "target"])
    frame = table.to_pandas()
    return frame.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total_weight)


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_mean(values, weights)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return float(np.std(values))
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2) / total_weight))


def describe_values(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    result: dict[str, float | int] = {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p001": float(np.quantile(values, 0.001)),
        "p005": float(np.quantile(values, 0.005)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p995": float(np.quantile(values, 0.995)),
        "p999": float(np.quantile(values, 0.999)),
        "max": float(np.max(values)),
        "abs_mean": float(np.mean(np.abs(values))),
        "abs_p95": float(np.quantile(np.abs(values), 0.95)),
        "abs_p99": float(np.quantile(np.abs(values), 0.99)),
    }
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        result["weighted_mean"] = weighted_mean(values, weights)
        result["weighted_std"] = weighted_std(values, weights)
        result["weighted_abs_mean"] = weighted_mean(np.abs(values), weights)
        result["weighted_target_square_sum"] = float(np.sum(weights * values * values))
        result["weight_sum"] = float(np.sum(weights))
    return result


def build_asset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset_id, group in frame.groupby("asset_id", sort=True):
        values = group["target"].to_numpy(dtype=np.float64)
        weights = group["weight"].to_numpy(dtype=np.float64)
        row = {"asset_id": int(asset_id), **describe_values(values, weights)}
        rows.append(row)
    return pd.DataFrame(rows)


def build_time_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for time_id, group in frame.groupby("time_id", sort=True):
        values = group["target"].to_numpy(dtype=np.float64)
        weights = group["weight"].to_numpy(dtype=np.float64)
        market_mean = weighted_mean(values, weights)
        rows.append(
            {
                "time_id": int(time_id),
                "row_count": int(len(group)),
                "target_mean": float(np.mean(values)),
                "target_std": float(np.std(values)),
                "target_min": float(np.min(values)),
                "target_max": float(np.max(values)),
                "target_abs_mean": float(np.mean(np.abs(values))),
                "weighted_mean": float(market_mean),
                "weighted_std": weighted_std(values, weights),
                "weighted_abs_mean": weighted_mean(np.abs(values), weights),
                "weighted_target_square_sum": float(np.sum(weights * values * values)),
                "weight_sum": float(np.sum(weights)),
            }
        )
    return pd.DataFrame(rows)


def build_tail_contribution(frame: pd.DataFrame, bucket_count: int = 20) -> pd.DataFrame:
    """看大 target 尾部对 R2 分母 sum(weight * target^2) 的贡献。"""
    values = frame["target"].to_numpy(dtype=np.float64)
    weights = frame["weight"].to_numpy(dtype=np.float64)
    abs_values = np.abs(values)
    contribution = weights * values * values
    order = np.argsort(abs_values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    buckets = np.floor(ranks / max(len(order) / bucket_count, 1)).astype(int)
    buckets = np.clip(buckets, 0, bucket_count - 1)
    tmp = pd.DataFrame(
        {
            "bucket": buckets,
            "abs_target": abs_values,
            "weighted_target_square": contribution,
        }
    )
    rows = []
    total = float(np.sum(contribution))
    for bucket, group in tmp.groupby("bucket", sort=True):
        rows.append(
            {
                "abs_target_bucket": int(bucket),
                "row_count": int(len(group)),
                "abs_target_min": float(group["abs_target"].min()),
                "abs_target_max": float(group["abs_target"].max()),
                "abs_target_mean": float(group["abs_target"].mean()),
                "weighted_target_square_sum": float(group["weighted_target_square"].sum()),
                "denominator_share": float(group["weighted_target_square"].sum() / total),
            }
        )
    out = pd.DataFrame(rows)
    out["cumulative_share_from_small"] = out["denominator_share"].cumsum()
    out["cumulative_share_from_large"] = out["denominator_share"][::-1].cumsum()[::-1]
    return out


def plot_histogram(frame: pd.DataFrame, output_path: Path) -> None:
    target = frame["target"].to_numpy(dtype=np.float64)
    lo, hi = np.quantile(target, [0.001, 0.999])
    plt.figure(figsize=(9, 5))
    plt.hist(target, bins=200, range=(lo, hi), color="#2563eb", alpha=0.85)
    plt.axvline(0.0, color="#111827", linewidth=1)
    plt.title("Target Distribution (0.1%-99.9% clipped)")
    plt.xlabel("target")
    plt.ylabel("row count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_asset_quantiles(asset_summary: pd.DataFrame, output_path: Path) -> None:
    x = asset_summary["asset_id"].to_numpy()
    plt.figure(figsize=(10, 5))
    plt.fill_between(x, asset_summary["p05"], asset_summary["p95"], color="#93c5fd", alpha=0.45, label="p05-p95")
    plt.fill_between(x, asset_summary["p25"], asset_summary["p75"], color="#2563eb", alpha=0.35, label="p25-p75")
    plt.plot(x, asset_summary["p50"], color="#111827", linewidth=1.4, label="median")
    plt.axhline(0.0, color="#6b7280", linewidth=1)
    plt.xticks(x)
    plt.title("Target Quantiles by Asset")
    plt.xlabel("asset_id")
    plt.ylabel("target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_time_distribution(time_summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(time_summary["weighted_mean"], bins=160, color="#0891b2", alpha=0.85)
    axes[0].axvline(0.0, color="#111827", linewidth=1)
    axes[0].set_title("Per-time Weighted Target Mean")
    axes[0].set_xlabel("weighted mean target")
    axes[0].set_ylabel("time_id count")
    axes[1].hist(time_summary["weighted_std"], bins=160, color="#7c3aed", alpha=0.85)
    axes[1].set_title("Per-time Cross-sectional Target Std")
    axes[1].set_xlabel("weighted std across assets")
    axes[1].set_ylabel("time_id count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_market_target_timeseries(time_summary: pd.DataFrame, rolling_window: int, output_path: Path) -> None:
    ordered = time_summary.sort_values("time_id").copy()
    ordered["rolling_weighted_mean"] = ordered["weighted_mean"].rolling(rolling_window, min_periods=50).mean()
    ordered["rolling_abs_mean"] = ordered["weighted_abs_mean"].rolling(rolling_window, min_periods=50).mean()
    plt.figure(figsize=(12, 5))
    plt.plot(ordered["time_id"], ordered["rolling_weighted_mean"], linewidth=1.0, label=f"rolling mean target ({rolling_window})")
    plt.plot(ordered["time_id"], ordered["rolling_abs_mean"], linewidth=1.0, label=f"rolling abs target ({rolling_window})")
    plt.axhline(0.0, color="#111827", linewidth=1)
    plt.title("Rolling Market Target Statistics")
    plt.xlabel("time_id")
    plt.ylabel("target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_tail_contribution(tail: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(tail["abs_target_bucket"], tail["denominator_share"], color="#dc2626", alpha=0.85)
    plt.title("R2 Denominator Share by |target| Bucket")
    plt.xlabel("|target| bucket, small to large")
    plt.ylabel("share of sum(weight * target^2)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def profile_prediction_file(prediction_file: Path, results_dir: Path, example_time_id: int) -> dict | None:
    if prediction_file is None or not prediction_file.exists():
        return None
    pred = pd.read_csv(prediction_file)
    if not {"time_id", "asset_id", "target", "weight", "prediction"}.issubset(pred.columns):
        return None

    y = pred["target"].to_numpy(dtype=np.float64)
    p = pred["prediction"].to_numpy(dtype=np.float64)
    w = pred["weight"].to_numpy(dtype=np.float64)
    pred["abs_target"] = np.abs(pred["target"])
    pred["abs_error"] = np.abs(pred["prediction"] - pred["target"])
    pred["squared_error_weighted"] = pred["weight"] * (pred["prediction"] - pred["target"]) ** 2
    pred["target_square_weighted"] = pred["weight"] * pred["target"] ** 2

    # 按 |target| 分桶看误差和预测幅度，解释为什么大 target 时视觉误差很大。
    pred["abs_target_bucket"] = pd.qcut(pred["abs_target"], 10, labels=False, duplicates="drop")
    bucket = pred.groupby("abs_target_bucket", sort=True).agg(
        row_count=("target", "size"),
        abs_target_mean=("abs_target", "mean"),
        target_std=("target", "std"),
        prediction_abs_mean=("prediction", lambda s: float(np.mean(np.abs(s)))),
        abs_error_mean=("abs_error", "mean"),
        weighted_squared_error_sum=("squared_error_weighted", "sum"),
        weighted_target_square_sum=("target_square_weighted", "sum"),
    ).reset_index()
    bucket["bucket_r2"] = 1.0 - bucket["weighted_squared_error_sum"] / bucket["weighted_target_square_sum"]
    bucket.to_csv(results_dir / "prediction_error_by_abs_target_bucket.csv", index=False)

    example = pred.loc[pred["time_id"] == int(example_time_id)].sort_values("asset_id")
    if example.empty:
        # 如果指定 time_id 不在 prediction 文件里，就取第一段 calibration 的第一个 time_id。
        example_time_id = int(pred["time_id"].min())
        example = pred.loc[pred["time_id"] == int(example_time_id)].sort_values("asset_id")
    example.to_csv(results_dir / "example_time_prediction.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(bucket["abs_target_mean"], bucket["abs_error_mean"], marker="o", label="mean |error|")
    plt.plot(bucket["abs_target_mean"], bucket["prediction_abs_mean"], marker="o", label="mean |prediction|")
    plt.xlabel("mean |target| in bucket")
    plt.ylabel("value")
    plt.title("Prediction Magnitude vs Target Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "prediction_error_by_target_size.png", dpi=160)
    plt.close()

    return {
        "prediction_file": str(prediction_file),
        "row_count": int(len(pred)),
        "weighted_zero_mean_r2": float(weighted_zero_mean_r2(y, p, w)),
        "prediction_summary": describe_values(p, w),
        "error_summary": describe_values(p - y, w),
        "abs_error_summary": describe_values(np.abs(p - y), w),
        "example_time_id": int(example_time_id),
        "example_time_target_mean": float(example["target"].mean()),
        "example_time_prediction_mean": float(example["prediction"].mean()),
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    frame = read_target_frame(args.raw_data_dir)
    target = frame["target"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)

    asset_summary = build_asset_summary(frame)
    time_summary = build_time_summary(frame)
    tail_contribution = build_tail_contribution(frame)

    asset_summary.to_csv(args.results_dir / "target_by_asset_summary.csv", index=False)
    time_summary.to_csv(args.results_dir / "target_by_time_summary.csv", index=False)
    tail_contribution.to_csv(args.results_dir / "target_tail_contribution.csv", index=False)

    plot_histogram(frame, args.results_dir / "target_histogram.png")
    plot_asset_quantiles(asset_summary, args.results_dir / "target_quantiles_by_asset.png")
    plot_time_distribution(time_summary, args.results_dir / "target_by_time_distribution.png")
    plot_market_target_timeseries(time_summary, args.rolling_window, args.results_dir / "target_market_timeseries.png")
    plot_tail_contribution(tail_contribution, args.results_dir / "target_tail_contribution.png")

    prediction_profile = profile_prediction_file(args.prediction_file, args.results_dir, args.example_time_id)

    time_desc_columns = ["weighted_mean", "weighted_std", "weighted_abs_mean", "target_mean", "target_std"]
    time_distribution = {
        column: describe_values(time_summary[column].to_numpy(dtype=np.float64))
        for column in time_desc_columns
    }
    summary = {
        "raw_data_dir": str(args.raw_data_dir),
        "row_count": int(len(frame)),
        "time_min": int(frame["time_id"].min()),
        "time_max": int(frame["time_id"].max()),
        "asset_count": int(frame["asset_id"].nunique()),
        "time_count": int(frame["time_id"].nunique()),
        "target_summary": describe_values(target, weight),
        "time_distribution_summary": time_distribution,
        "tail_contribution_top_abs_5pct_share": float(
            tail_contribution.tail(1)["cumulative_share_from_large"].iloc[0]
        ),
        "tail_contribution_top_abs_10pct_share": float(
            tail_contribution.tail(2)["denominator_share"].sum()
        ),
        "prediction_profile": prediction_profile,
        "output_files": {
            "target_by_asset_summary": str(args.results_dir / "target_by_asset_summary.csv"),
            "target_by_time_summary": str(args.results_dir / "target_by_time_summary.csv"),
            "target_tail_contribution": str(args.results_dir / "target_tail_contribution.csv"),
            "target_histogram": str(args.results_dir / "target_histogram.png"),
            "target_quantiles_by_asset": str(args.results_dir / "target_quantiles_by_asset.png"),
            "target_by_time_distribution": str(args.results_dir / "target_by_time_distribution.png"),
            "target_market_timeseries": str(args.results_dir / "target_market_timeseries.png"),
            "target_tail_contribution_plot": str(args.results_dir / "target_tail_contribution.png"),
            "prediction_error_by_abs_target_bucket": str(args.results_dir / "prediction_error_by_abs_target_bucket.csv"),
            "prediction_error_by_target_size_plot": str(args.results_dir / "prediction_error_by_target_size.png"),
            "example_time_prediction": str(args.results_dir / "example_time_prediction.csv"),
        },
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
