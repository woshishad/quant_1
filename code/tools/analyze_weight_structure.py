from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose how asset identity and observable features proxy the training-only weight."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--feature-profile",
        type=Path,
        default=Path("results/feature_weight_target_analysis_20260824/feature_profile.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/feature_weight_target_analysis_20260824"),
    )
    parser.add_argument("--train-last-partition", type=int, default=6)
    parser.add_argument("--sample-mod", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=200_000)
    return parser.parse_args()


def within_group_corr(stats: dict[str, np.ndarray]) -> np.ndarray:
    n = np.maximum(stats["n"], 1.0)
    cov = np.sum(stats["sxw"] - stats["sx"] * stats["sw"] / n, axis=0)
    var_x = np.sum(stats["sxx"] - stats["sx"] * stats["sx"] / n, axis=0)
    var_w = np.sum(stats["sww"] - stats["sw"] * stats["sw"] / n, axis=0)
    denominator = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_w, 0.0))
    return np.divide(cov, denominator, out=np.zeros_like(cov), where=denominator > 1e-12)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted((args.data_root / "train").glob("train_partition_*.parquet"))
    if not paths:
        raise FileNotFoundError(args.data_root / "train")

    profile = pd.read_csv(args.feature_profile)
    profile["abs_corr_weight"] = profile["corr_weight"].abs()
    profile = profile.sort_values("abs_corr_weight", ascending=False)
    feature_names = profile["feature_name"].head(args.max_features).tolist()
    feature_count = len(feature_names)
    asset_count = 15
    stats = {
        key: np.zeros((asset_count, feature_count), dtype=np.float64)
        for key in ("n", "sx", "sw", "sxx", "sww", "sxw")
    }
    train_weight_sum = np.zeros(asset_count, dtype=np.float64)
    train_asset_count = np.zeros(asset_count, dtype=np.float64)
    valid_weight_sum = np.zeros(asset_count, dtype=np.float64)
    valid_weight_sumsq = np.zeros(asset_count, dtype=np.float64)
    valid_asset_count = np.zeros(asset_count, dtype=np.float64)
    sampled_train: list[pd.DataFrame] = []
    sampled_valid: list[pd.DataFrame] = []
    columns = ["row_id", "asset_id", "weight", *feature_names]

    for partition_index, path in enumerate(paths):
        print(f"reading {partition_index + 1}/{len(paths)}: {path.name}", flush=True)
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=args.batch_size):
            frame = batch.to_pandas(split_blocks=True, self_destruct=True)
            asset = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
            weight = frame["weight"].to_numpy(dtype=np.float64, copy=False)
            values = frame[feature_names].to_numpy(dtype=np.float64, copy=False)
            finite = np.isfinite(values) & np.isfinite(weight[:, None])
            for asset_id in np.unique(asset):
                mask = asset == asset_id
                valid = finite[mask]
                x = values[mask]
                w = weight[mask, None]
                stats["n"][asset_id] += valid.sum(axis=0)
                stats["sx"][asset_id] += np.where(valid, x, 0.0).sum(axis=0)
                stats["sw"][asset_id] += np.where(valid, w, 0.0).sum(axis=0)
                stats["sxx"][asset_id] += np.where(valid, x * x, 0.0).sum(axis=0)
                stats["sww"][asset_id] += np.where(valid, w * w, 0.0).sum(axis=0)
                stats["sxw"][asset_id] += np.where(valid, x * w, 0.0).sum(axis=0)

            if partition_index <= args.train_last_partition:
                train_weight_sum += np.bincount(asset, weights=weight, minlength=asset_count)
                train_asset_count += np.bincount(asset, minlength=asset_count)
            else:
                valid_weight_sum += np.bincount(asset, weights=weight, minlength=asset_count)
                valid_weight_sumsq += np.bincount(asset, weights=weight * weight, minlength=asset_count)
                valid_asset_count += np.bincount(asset, minlength=asset_count)
            row_id = frame["row_id"].to_numpy(dtype=np.uint64, copy=False)
            hashed_row_id = row_id * np.uint64(11400714819323198485)
            sample_cutoff = np.iinfo(np.uint64).max // args.sample_mod
            sampled = frame.loc[hashed_row_id <= sample_cutoff].copy()
            if partition_index <= args.train_last_partition:
                sampled_train.append(sampled)
            else:
                sampled_valid.append(sampled)

    train = pd.concat(sampled_train, ignore_index=True)
    valid = pd.concat(sampled_valid, ignore_index=True)
    asset_means = np.divide(
        train_weight_sum,
        train_asset_count,
        out=np.zeros_like(train_weight_sum),
        where=train_asset_count > 0,
    )
    valid_weight = valid["weight"].to_numpy(dtype=np.float64)
    asset_sse = np.sum(
        valid_weight_sumsq
        - 2.0 * asset_means * valid_weight_sum
        + valid_asset_count * asset_means * asset_means
    )
    valid_count = np.sum(valid_asset_count)
    valid_total_sum = np.sum(valid_weight_sum)
    valid_total_sumsq = np.sum(valid_weight_sumsq)
    valid_sst = valid_total_sumsq - valid_total_sum * valid_total_sum / valid_count
    proxy_rows: list[dict[str, float | int | str]] = [
        {
            "model": "asset_mean_only",
            "feature_count": 0,
            "validation_r2": float(1.0 - asset_sse / valid_sst),
        }
    ]

    for count in (5, 10, 20, 50):
        if count > feature_count:
            continue
        selected = feature_names[:count]
        model_columns = ["asset_id", *selected]
        model = lgb.LGBMRegressor(
            objective="regression_l2",
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            random_state=20260824,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            train[model_columns],
            train["weight"],
            categorical_feature=["asset_id"],
        )
        prediction = model.predict(valid[model_columns])
        proxy_rows.append(
            {
                "model": "lightgbm_asset_plus_features",
                "feature_count": count,
                "validation_r2": float(r2_score(valid_weight, prediction)),
            }
        )

    within_corr = within_group_corr(stats)
    weight_feature_structure = profile.head(feature_count).copy()
    weight_feature_structure["within_asset_corr_weight"] = within_corr
    weight_feature_structure["abs_within_asset_corr_weight"] = np.abs(within_corr)
    output_columns = [
        "feature_name",
        "corr_weight",
        "within_asset_corr_weight",
        "weight_block_mean_corr",
        "weight_block_std_corr",
        "weight_block_positive_share",
    ]
    weight_feature_structure[output_columns].to_csv(
        args.output_dir / "weight_feature_structure.csv", index=False
    )
    proxy_frame = pd.DataFrame(proxy_rows)
    proxy_frame.to_csv(args.output_dir / "weight_proxy_validation.csv", index=False)

    summary = {
        "purpose": "diagnostic only; weight is unavailable at inference time",
        "feature_selection": "top absolute weight correlations from the full-release exploratory feature profile",
        "validation_caveat": "fit and validation partitions are time ordered, but feature selection is exploratory rather than a nested time-forward selection",
        "train_partitions": [0, args.train_last_partition],
        "validation_partitions": [args.train_last_partition + 1, len(paths) - 1],
        "sample_rule": f"deterministic uint64 multiplicative hash, approximately 1/{args.sample_mod} rows for LightGBM proxy models",
        "sampled_train_rows": int(len(train)),
        "sampled_validation_rows": int(len(valid)),
        "proxy_validation": proxy_rows,
        "outputs": ["weight_feature_structure.csv", "weight_proxy_validation.csv"],
    }
    (args.output_dir / "weight_structure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
