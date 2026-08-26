from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_train_predict import (
    BASE_COLUMNS_TRAIN,
    load_feature_ranking,
    parquet_paths,
    read_partitioned_frame,
    schema_columns,
    time_range,
)
from market_mean_ts_model import build_time_feature_frame, weighted_market_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="针对 time_id 市场加权 target 均值做时间 OOF 因子筛选。"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--source-features-file",
        type=Path,
        default=Path(
            "results/asset_all_stable_features_100k/selected_features_stable_top128.csv"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/market_target_feature_screen_75k"),
    )
    parser.add_argument("--train-lookback-time-points", type=int, default=75_000)
    parser.add_argument("--cal-time-points", type=int, default=20_000)
    parser.add_argument("--source-top-k", type=int, default=128)
    parser.add_argument("--oof-initial-time-points", type=int, default=25_000)
    parser.add_argument("--oof-block-time-points", type=int, default=10_000)
    parser.add_argument("--max-train-time-id", type=int, default=None)
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


def fill_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    matrix = frame[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    missing = ~np.isfinite(matrix)
    if missing.any():
        matrix[missing] = np.take(medians, np.where(missing)[1])
    return matrix


def score_univariate_fold(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    valid_weight: np.ndarray,
) -> np.ndarray:
    """向量化拟合每个聚合特征的一元加权线性模型，并计算样本外 R²。"""
    train_weight = train_weight.astype(np.float64)
    valid_weight = valid_weight.astype(np.float64)
    weight_sum = max(float(np.sum(train_weight)), 1e-12)
    x_mean = np.einsum("i,ij->j", train_weight, train_x, optimize=True) / weight_sum
    y_mean = float(np.sum(train_weight * train_y) / weight_sum)
    x_centered = train_x.astype(np.float64) - x_mean
    y_centered = train_y.astype(np.float64) - y_mean
    covariance = np.einsum(
        "i,ij,i->j", train_weight, x_centered, y_centered, optimize=True
    )
    variance = np.einsum(
        "i,ij,ij->j", train_weight, x_centered, x_centered, optimize=True
    )
    slope = covariance / np.maximum(variance, 1e-12)
    intercept = y_mean - slope * x_mean
    prediction = valid_x.astype(np.float64) * slope + intercept
    denominator = float(np.sum(valid_weight * valid_y.astype(np.float64) ** 2))
    squared_error = np.einsum(
        "i,ij,ij->j",
        valid_weight,
        valid_y.astype(np.float64)[:, None] - prediction,
        valid_y.astype(np.float64)[:, None] - prediction,
        optimize=True,
    )
    return 1.0 - squared_error / max(denominator, 1e-12)


def raw_feature_name(aggregate_column: str) -> str:
    for suffix in ["_xmean", "_xstd", "_xmin", "_xmax", "_xrange"]:
        if aggregate_column.endswith(suffix):
            return aggregate_column[: -len(suffix)]
    raise ValueError(f"无法识别聚合特征列：{aggregate_column}")


def plot_top_features(frame: pd.DataFrame, output_path: Path) -> None:
    top = frame.head(30).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature_name"], top["best_aggregate_mean_score"], color="#2563eb")
    plt.xlabel("mean OOF market-target R2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train_paths = parquet_paths(args.raw_data_dir, "train")
    train_min, train_max = time_range(train_paths)
    train_end = (
        min(train_max, int(args.max_train_time_id))
        if args.max_train_time_id is not None
        else train_max
    )
    lookback_start = max(
        train_min, train_end - int(args.train_lookback_time_points) + 1
    )
    fit_end = train_end - int(args.cal_time_points)

    available = schema_columns(train_paths)
    ranking = load_feature_ranking(args.source_features_file, available)
    raw_features = ranking.head(int(args.source_top_k))["feature_name"].astype(str).tolist()
    print(
        f"Market feature screen: fit={lookback_start}..{fit_end}, "
        f"raw_features={len(raw_features)}"
    )
    raw_train = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + raw_features,
        min_time=lookback_start,
        max_time=fit_end,
    )
    market_targets = weighted_market_target(raw_train)
    time_features, aggregate_columns = build_time_feature_frame(
        raw_train, raw_features, [], [], [], []
    )
    time_frame = time_features.merge(market_targets, on="time_id", how="left")
    matrix = fill_matrix(time_frame, aggregate_columns)
    target = time_frame["market_target"].to_numpy(dtype=np.float64)
    weight = time_frame["weight_sum"].to_numpy(dtype=np.float64)

    row_count = len(time_frame)
    initial = int(args.oof_initial_time_points)
    block_size = int(args.oof_block_time_points)
    fold_scores = []
    fold_meta = []
    block_start = initial
    fold_index = 1
    while block_start < row_count:
        block_end = min(row_count, block_start + block_size)
        scores = score_univariate_fold(
            matrix[:block_start],
            target[:block_start],
            weight[:block_start],
            matrix[block_start:block_end],
            target[block_start:block_end],
            weight[block_start:block_end],
        )
        fold_scores.append(scores)
        fold_meta.append(
            {
                "fold": fold_index,
                "train_time_min": int(time_frame.iloc[0]["time_id"]),
                "train_time_max": int(time_frame.iloc[block_start - 1]["time_id"]),
                "valid_time_min": int(time_frame.iloc[block_start]["time_id"]),
                "valid_time_max": int(time_frame.iloc[block_end - 1]["time_id"]),
                "valid_rows": int(block_end - block_start),
            }
        )
        block_start = block_end
        fold_index += 1

    score_matrix = np.vstack(fold_scores)
    aggregate_ranking = pd.DataFrame({"aggregate_feature": aggregate_columns})
    aggregate_ranking["feature_name"] = [
        raw_feature_name(column) for column in aggregate_columns
    ]
    for fold_index in range(score_matrix.shape[0]):
        aggregate_ranking[f"fold_{fold_index + 1}_score"] = score_matrix[fold_index]
        aggregate_ranking[f"fold_{fold_index + 1}_rank"] = (
            pd.Series(score_matrix[fold_index]).rank(ascending=False, method="average").to_numpy()
        )
    score_columns = [column for column in aggregate_ranking if column.endswith("_score")]
    rank_columns = [column for column in aggregate_ranking if column.endswith("_rank")]
    aggregate_ranking["mean_oof_score"] = aggregate_ranking[score_columns].mean(axis=1)
    aggregate_ranking["min_oof_score"] = aggregate_ranking[score_columns].min(axis=1)
    aggregate_ranking["positive_fold_count"] = (
        aggregate_ranking[score_columns] > 0.0
    ).sum(axis=1)
    aggregate_ranking["mean_rank"] = aggregate_ranking[rank_columns].mean(axis=1)
    aggregate_ranking["worst_rank"] = aggregate_ranking[rank_columns].max(axis=1)
    aggregate_ranking = aggregate_ranking.sort_values(
        ["positive_fold_count", "mean_rank", "worst_rank", "mean_oof_score"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    aggregate_ranking.insert(0, "aggregate_rank", np.arange(1, len(aggregate_ranking) + 1))

    # 每个原始 feature 取表现最稳定的横截面统计量，避免同一 feature 重复占满榜单。
    best_rows = []
    for feature_name, group in aggregate_ranking.groupby("feature_name", sort=False):
        best = group.sort_values(
            ["positive_fold_count", "mean_rank", "worst_rank", "mean_oof_score"],
            ascending=[False, True, True, False],
        ).iloc[0]
        best_rows.append(
            {
                "feature_name": feature_name,
                "best_aggregate_feature": best["aggregate_feature"],
                "positive_fold_count": int(best["positive_fold_count"]),
                "mean_rank": float(best["mean_rank"]),
                "worst_rank": float(best["worst_rank"]),
                "best_aggregate_mean_score": float(best["mean_oof_score"]),
                "best_aggregate_min_score": float(best["min_oof_score"]),
            }
        )
    feature_ranking = pd.DataFrame(best_rows).sort_values(
        ["positive_fold_count", "mean_rank", "worst_rank", "best_aggregate_mean_score"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    feature_ranking.insert(0, "market_rank", np.arange(1, len(feature_ranking) + 1))

    aggregate_ranking.to_csv(args.results_dir / "aggregate_feature_ranking.csv", index=False)
    feature_ranking.to_csv(args.results_dir / "market_feature_ranking.csv", index=False)
    for top_k in [16, 32, 48, 64, 128]:
        feature_ranking.head(top_k).to_csv(
            args.results_dir / f"selected_market_features_top{top_k}.csv", index=False
        )
    pd.DataFrame(fold_meta).to_csv(args.results_dir / "oof_folds.csv", index=False)
    plot_top_features(feature_ranking, args.results_dir / "top_market_features.png")

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_safe": True,
        "fit_time_range": [int(lookback_start), int(fit_end)],
        "raw_feature_count": int(len(raw_features)),
        "aggregate_feature_count": int(len(aggregate_columns)),
        "time_point_count": int(len(time_frame)),
        "folds": fold_meta,
        "top32_positive_all_folds": int(
            (feature_ranking.head(32)["positive_fold_count"] == len(fold_meta)).sum()
        ),
        "output_files": {
            "market_feature_ranking": str(args.results_dir / "market_feature_ranking.csv"),
            "selected_top32": str(args.results_dir / "selected_market_features_top32.csv"),
            "aggregate_feature_ranking": str(
                args.results_dir / "aggregate_feature_ranking.csv"
            ),
            "plot": str(args.results_dir / "top_market_features.png"),
        },
    }
    with (args.results_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=json_default)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
