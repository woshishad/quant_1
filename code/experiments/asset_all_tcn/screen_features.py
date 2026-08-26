from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen all feature_* columns by all-asset weighted univariate R2.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/asset_all_time50000"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset_all_feature_screen"))
    parser.add_argument("--train-end-time", type=int, default=39_999)
    parser.add_argument("--valid-start-time", type=int, default=40_000)
    parser.add_argument("--valid-end-time", type=int, default=49_999)
    parser.add_argument("--top-k", type=int, nargs="+", default=[32, 64, 128])
    return parser.parse_args()


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.sum(weight * y_true * y_true))
    if denominator <= 0:
        return 0.0
    numerator = float(np.sum(weight * (y_true - y_pred) ** 2))
    return 1.0 - numerator / denominator


def safe_standardize(train_values: np.ndarray, valid_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    # 单因子筛选也严格只用训练段拟合均值/标准差，验证段只复用训练统计量。
    mean = float(np.nanmean(train_values))
    scale = float(np.nanstd(train_values))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    train_x = np.nan_to_num((train_values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    valid_x = np.nan_to_num((valid_values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    return train_x, valid_x, mean, scale


def fit_weighted_single_feature(train_x: np.ndarray, y_train: np.ndarray, w_train: np.ndarray) -> float:
    # 零截距单因子加权最小二乘，和 zero-mean R2 的“预测 0”基准一致。
    denominator = float(np.sum(w_train * train_x * train_x))
    if denominator <= 1e-18:
        return 0.0
    return float(np.sum(w_train * train_x * y_train) / denominator)


def optimal_shrink(y_true: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.sum(weight * prediction * prediction))
    if denominator <= 1e-18:
        return 0.0
    shrink = float(np.sum(weight * y_true * prediction) / denominator)
    return min(1.2, max(0.0, shrink))


def score_by_asset(y_true: np.ndarray, prediction: np.ndarray, weight: np.ndarray, asset_id: np.ndarray) -> dict[str, float]:
    scores = {}
    for asset in sorted(np.unique(asset_id)):
        mask = asset_id == asset
        scores[str(int(asset))] = weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])
    return scores


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_dir / "train.parquet"
    schema_columns = pq.ParquetFile(data_path).schema_arrow.names
    feature_columns = [column for column in schema_columns if column.startswith("feature_")]
    base_columns = ["time_id", "asset_id", "weight", "target"]

    # 先读基础列，所有 feature 共享同一套 train/valid mask 和 y/weight。
    base = pd.read_parquet(data_path, columns=base_columns)
    time_id = base["time_id"].to_numpy(dtype=np.int64)
    train_mask = time_id <= args.train_end_time
    valid_mask = (time_id >= args.valid_start_time) & (time_id <= args.valid_end_time)
    y_train = base.loc[train_mask, "target"].to_numpy(dtype=np.float64)
    y_valid = base.loc[valid_mask, "target"].to_numpy(dtype=np.float64)
    w_train = base.loc[train_mask, "weight"].to_numpy(dtype=np.float64)
    w_valid = base.loc[valid_mask, "weight"].to_numpy(dtype=np.float64)
    asset_valid = base.loc[valid_mask, "asset_id"].to_numpy(dtype=np.int64)
    train_weight = w_train / max(float(np.mean(w_train)), 1e-12)

    rows = []
    for index, feature_name in enumerate(feature_columns):
        values = pd.read_parquet(data_path, columns=[feature_name])[feature_name]
        train_values = values.loc[train_mask].to_numpy(dtype=np.float64)
        valid_values = values.loc[valid_mask].to_numpy(dtype=np.float64)
        train_x, valid_x, mean, scale = safe_standardize(train_values, valid_values)
        coef = fit_weighted_single_feature(train_x, y_train, train_weight)
        raw_prediction = coef * valid_x
        raw_score = weighted_zero_mean_r2(y_valid, raw_prediction, w_valid)
        shrink = optimal_shrink(y_valid, raw_prediction, w_valid)
        prediction = shrink * raw_prediction
        shrink_score = weighted_zero_mean_r2(y_valid, prediction, w_valid)
        asset_scores = score_by_asset(y_valid, prediction, w_valid, asset_valid)
        rows.append(
            {
                "feature_index": index,
                "feature_name": feature_name,
                "coef": coef,
                "mean": mean,
                "scale": scale,
                "raw_score": raw_score,
                "shrink": shrink,
                "shrink_score": shrink_score,
                "positive_asset_count": int(sum(score > 0 for score in asset_scores.values())),
                "min_asset_score": float(min(asset_scores.values())),
                "max_asset_score": float(max(asset_scores.values())),
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(feature_columns):
            print(f"screened {index + 1}/{len(feature_columns)} features")

    ranking = pd.DataFrame(rows).sort_values("shrink_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(args.results_dir / "feature_ranking.csv", index=False)

    top_files = {}
    for top_k in args.top_k:
        selected = ranking.head(top_k)[["feature_index", "feature_name", "shrink_score", "positive_asset_count"]]
        output_path = args.results_dir / f"selected_features_top{top_k}.csv"
        selected.to_csv(output_path, index=False)
        top_files[str(top_k)] = str(output_path)

    manifest = {
        "data_dir": str(args.data_dir),
        "feature_count": int(len(feature_columns)),
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "metric": "single-feature weighted zero-mean R2 with validation shrink",
        "top_files": top_files,
        "best_features": ranking.head(20)[["rank", "feature_name", "shrink_score", "positive_asset_count"]].to_dict(
            orient="records"
        ),
    }
    (args.results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
