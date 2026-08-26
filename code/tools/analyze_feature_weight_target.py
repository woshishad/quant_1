from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile anonymous features, responders, target and weight on the full train release."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/feature_weight_target_analysis_20260824"),
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def safe_corr(n: np.ndarray, sx: np.ndarray, sy: np.ndarray, sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray) -> np.ndarray:
    cov = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sxx - sx * sx, 0.0) * np.maximum(n * syy - sy * sy, 0.0))
    return np.divide(cov, den, out=np.zeros_like(cov, dtype=np.float64), where=den > 1e-12)


def update_pair_stats(
    x: np.ndarray,
    y: np.ndarray,
    stats: dict[str, np.ndarray],
    prefix: str,
    valid: np.ndarray,
) -> None:
    """Update per-column Pearson sufficient statistics without retaining rows."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    clean_x = np.where(valid, x, 0.0)
    clean_y = np.where(valid, y[:, None], 0.0)
    n = valid.sum(axis=0).astype(np.float64)
    stats[f"n_{prefix}"] += n
    stats[f"sx_{prefix}"] += clean_x.sum(axis=0)
    stats[f"sxx_{prefix}"] += (clean_x * clean_x).sum(axis=0)
    stats[f"sy_{prefix}"] += (clean_y * valid).sum(axis=0)
    stats[f"syy_{prefix}"] += ((clean_y * clean_y) * valid).sum(axis=0)
    stats[f"sxy_{prefix}"] += (clean_x * clean_y).sum(axis=0)


def new_pair_stats(count: int, blocks: int) -> dict[str, np.ndarray]:
    keys = ("n", "sx", "sxx", "sy", "syy", "sxy")
    return {f"{key}_{name}": np.zeros((blocks, count), dtype=np.float64) for name in ("global",) for key in keys} | {
        f"{key}_block": np.zeros((blocks, count), dtype=np.float64) for key in keys
    }


def update_block_pair_stats(
    x: np.ndarray,
    y: np.ndarray,
    stats: dict[str, np.ndarray],
    block: int,
    valid: np.ndarray,
) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    clean_x = np.where(valid, x, 0.0)
    clean_y = np.where(valid, y[:, None], 0.0)
    n = valid.sum(axis=0).astype(np.float64)
    stats["n_block"][block] += n
    stats["sx_block"][block] += clean_x.sum(axis=0)
    stats["sxx_block"][block] += (clean_x * clean_x).sum(axis=0)
    stats["sy_block"][block] += (clean_y * valid).sum(axis=0)
    stats["syy_block"][block] += ((clean_y * clean_y) * valid).sum(axis=0)
    stats["sxy_block"][block] += (clean_x * clean_y).sum(axis=0)


def profile_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y[:, None])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = finite.sum(axis=0).astype(np.float64)
    sx = np.where(finite, x, 0.0).sum(axis=0)
    sy = np.where(finite, y[:, None], 0.0).sum(axis=0)
    sxx = np.where(finite, x * x, 0.0).sum(axis=0)
    syy = np.where(finite, y[:, None] * y[:, None], 0.0).sum(axis=0)
    sxy = np.where(finite, x * y[:, None], 0.0).sum(axis=0)
    return n, safe_corr(n, sx, sy, sxx, syy, sxy)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_paths = sorted((args.data_root / "train").glob("train_partition_*.parquet"))
    if not train_paths:
        raise FileNotFoundError(args.data_root / "train")
    schema = pq.ParquetFile(train_paths[0]).schema_arrow.names
    feature_names = [name for name in schema if name.startswith("feature_")]
    responder_names = [name for name in schema if name.startswith("responder_")]
    blocks = len(train_paths)
    feature_stats = {
        key: np.zeros(len(feature_names), dtype=np.float64)
        for key in ("count", "missing", "sum", "sumsq", "target_n", "target_sx", "target_sxx", "target_sy", "target_syy", "target_sxy", "abs_n", "abs_sx", "abs_sxx", "abs_sy", "abs_syy", "abs_sxy", "weight_n", "weight_sx", "weight_sxx", "weight_sy", "weight_syy", "weight_sxy")
    }
    responder_stats = {
        key: np.zeros(len(responder_names), dtype=np.float64)
        for key in ("count", "missing", "target_n", "target_sx", "target_sxx", "target_sy", "target_syy", "target_sxy", "abs_n", "abs_sx", "abs_sxx", "abs_sy", "abs_syy", "abs_sxy", "weight_n", "weight_sx", "weight_sxx", "weight_sy", "weight_syy", "weight_sxy")
    }
    block_target = {key: np.zeros((blocks, len(feature_names)), dtype=np.float64) for key in ("n", "sx", "sxx", "sy", "syy", "sxy")}
    block_weight = {key: np.zeros((blocks, len(feature_names)), dtype=np.float64) for key in ("n", "sx", "sxx", "sy", "syy", "sxy")}
    block_abs = {key: np.zeros((blocks, len(feature_names)), dtype=np.float64) for key in ("n", "sx", "sxx", "sy", "syy", "sxy")}
    asset_rows: list[dict[str, float | int]] = []
    sample_rng = np.random.default_rng(20260824)
    reservoir: list[np.ndarray] = []
    reservoir_limit = 500_000
    total_rows = 0
    target_sum = target_sumsq = target_abs_sum = weight_sum = weight_sumsq = weight_target_sum = weight_abs_target_sum = weight_target_sq_sum = 0.0
    asset_acc: dict[int, dict[str, float]] = {}

    read_columns = ["time_id", "asset_id", "weight", "target", *feature_names, *responder_names]
    for block, path in enumerate(train_paths):
        print(f"reading {block + 1}/{blocks}: {path.name}", flush=True)
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=read_columns, batch_size=args.batch_size):
            frame = batch.to_pandas(split_blocks=True, self_destruct=True)
            y = frame["target"].to_numpy(dtype=np.float64, copy=False)
            w = frame["weight"].to_numpy(dtype=np.float64, copy=False)
            asset = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
            finite_y = np.isfinite(y)
            finite_w = np.isfinite(w) & (w >= 0.0)
            total_rows += len(frame)
            target_sum += float(np.nansum(y))
            target_sumsq += float(np.nansum(y * y))
            target_abs_sum += float(np.nansum(np.abs(y)))
            weight_sum += float(np.nansum(w))
            weight_sumsq += float(np.nansum(w * w))
            weight_target_sum += float(np.nansum(w * y))
            weight_abs_target_sum += float(np.nansum(w * np.abs(y)))
            weight_target_sq_sum += float(np.nansum(w * y * y))
            rows_before = total_rows - len(frame)
            if rows_before < reservoir_limit:
                take = min(len(frame), reservoir_limit - rows_before)
                reservoir.append(np.column_stack((y[:take], w[:take])))
            for a in np.unique(asset):
                mask = asset == a
                state = asset_acc.setdefault(int(a), {"rows": 0.0, "weight_sum": 0.0, "weight_target_sq": 0.0, "abs_target": 0.0, "target_sq": 0.0})
                state["rows"] += float(mask.sum())
                state["weight_sum"] += float(np.nansum(w[mask]))
                state["weight_target_sq"] += float(np.nansum(w[mask] * y[mask] * y[mask]))
                state["abs_target"] += float(np.nansum(np.abs(y[mask])))
                state["target_sq"] += float(np.nansum(y[mask] * y[mask]))

            x = frame[feature_names].to_numpy(dtype=np.float64, copy=False)
            valid_x = np.isfinite(x)
            feature_stats["count"] += len(frame)
            feature_stats["missing"] += (~valid_x).sum(axis=0)
            clean_x = np.where(valid_x, x, 0.0)
            feature_stats["sum"] += clean_x.sum(axis=0)
            feature_stats["sumsq"] += (clean_x * clean_x).sum(axis=0)
            for label, z in (("target", y), ("abs", np.abs(y)), ("weight", w)):
                valid = valid_x & np.isfinite(z[:, None])
                clean = np.where(valid, x, 0.0)
                clean_z = np.where(valid, z[:, None], 0.0)
                feature_stats[f"{label}_n"] += valid.sum(axis=0)
                feature_stats[f"{label}_sx"] += clean.sum(axis=0)
                feature_stats[f"{label}_sxx"] += (clean * clean).sum(axis=0)
                feature_stats[f"{label}_sy"] += (clean_z * valid).sum(axis=0)
                feature_stats[f"{label}_syy"] += ((clean_z * clean_z) * valid).sum(axis=0)
                feature_stats[f"{label}_sxy"] += (clean * clean_z).sum(axis=0)
                block_stats = {"target": block_target, "abs": block_abs, "weight": block_weight}[label]
                for key, value in (("n", valid.sum(axis=0)), ("sx", clean.sum(axis=0)), ("sxx", (clean * clean).sum(axis=0)), ("sy", (clean_z * valid).sum(axis=0)), ("syy", ((clean_z * clean_z) * valid).sum(axis=0)), ("sxy", (clean * clean_z).sum(axis=0))):
                    block_stats[key][block] += value

            r = frame[responder_names].to_numpy(dtype=np.float64, copy=False)
            valid_r = np.isfinite(r)
            responder_stats["count"] += len(frame)
            responder_stats["missing"] += (~valid_r).sum(axis=0)
            for label, z in (("target", y), ("abs", np.abs(y)), ("weight", w)):
                valid = valid_r & np.isfinite(z[:, None])
                clean = np.where(valid, r, 0.0)
                clean_z = np.where(valid, z[:, None], 0.0)
                responder_stats[f"{label}_n"] += valid.sum(axis=0)
                responder_stats[f"{label}_sx"] += clean.sum(axis=0)
                responder_stats[f"{label}_sxx"] += (clean * clean).sum(axis=0)
                responder_stats[f"{label}_sy"] += (clean_z * valid).sum(axis=0)
                responder_stats[f"{label}_syy"] += ((clean_z * clean_z) * valid).sum(axis=0)
                responder_stats[f"{label}_sxy"] += (clean * clean_z).sum(axis=0)

    def corr_from(prefix: str, stats: dict[str, np.ndarray]) -> np.ndarray:
        return safe_corr(stats[f"{prefix}_n"], stats[f"{prefix}_sx"], stats[f"{prefix}_sy"], stats[f"{prefix}_sxx"], stats[f"{prefix}_syy"], stats[f"{prefix}_sxy"])

    feature_frame = pd.DataFrame({"feature_name": feature_names, "missing_rate": feature_stats["missing"] / np.maximum(feature_stats["count"], 1), "mean": feature_stats["sum"] / np.maximum(feature_stats["count"] - feature_stats["missing"], 1), "std": np.sqrt(np.maximum(feature_stats["sumsq"] / np.maximum(feature_stats["count"] - feature_stats["missing"], 1) - (feature_stats["sum"] / np.maximum(feature_stats["count"] - feature_stats["missing"], 1)) ** 2, 0.0))})
    for label in ("target", "abs", "weight"):
        feature_frame[f"corr_{label}"] = corr_from(label, feature_stats)
    block_corrs = {}
    for label, stats in (("target", block_target), ("abs", block_abs), ("weight", block_weight)):
        corr = np.vstack([safe_corr(stats["n"][i], stats["sx"][i], stats["sy"][i], stats["sxx"][i], stats["syy"][i], stats["sxy"][i]) for i in range(blocks)])
        block_corrs[label] = corr
        feature_frame[f"{label}_block_mean_corr"] = np.nanmean(corr, axis=0)
        feature_frame[f"{label}_block_std_corr"] = np.nanstd(corr, axis=0)
        feature_frame[f"{label}_block_positive_share"] = np.mean(corr > 0.0, axis=0)
    feature_frame["target_rank_score"] = feature_frame["corr_target"].abs().rank(ascending=False, method="min").astype(int)
    feature_frame["weight_rank_score"] = feature_frame["corr_weight"].abs().rank(ascending=False, method="min").astype(int)
    feature_frame["role_hint"] = np.select(
        [feature_frame["corr_weight"].abs() >= 0.15, feature_frame["corr_target"].abs() >= 0.05, feature_frame["corr_abs"].abs() >= 0.05],
        ["weight_related_candidate", "direction_candidate", "magnitude_or_volatility_candidate"],
        default="weak_univariate_or_nonlinear_candidate",
    )
    feature_frame.sort_values("target_rank_score").to_csv(args.output_dir / "feature_profile.csv", index=False)
    feature_frame.sort_values("weight_rank_score").head(100).to_csv(args.output_dir / "top_weight_related_features.csv", index=False)
    feature_frame.sort_values("target_rank_score").head(100).to_csv(args.output_dir / "top_target_related_features.csv", index=False)

    responder_frame = pd.DataFrame({"responder_name": responder_names, "missing_rate": responder_stats["missing"] / np.maximum(responder_stats["count"], 1)})
    for label in ("target", "abs", "weight"):
        responder_frame[f"corr_{label}"] = corr_from(label, responder_stats)
    responder_frame["target_rank_score"] = responder_frame["corr_target"].abs().rank(ascending=False, method="min").astype(int)
    responder_frame["weight_rank_score"] = responder_frame["corr_weight"].abs().rank(ascending=False, method="min").astype(int)
    responder_frame.sort_values("target_rank_score").to_csv(args.output_dir / "responder_profile.csv", index=False)

    asset_frame = pd.DataFrame([{"asset_id": asset_id, **state} for asset_id, state in sorted(asset_acc.items())])
    asset_frame["mean_weight"] = asset_frame["weight_sum"] / asset_frame["rows"]
    asset_frame["weighted_target_sq_share"] = asset_frame["weight_target_sq"] / max(weight_target_sq_sum, 1e-12)
    asset_frame.to_csv(args.output_dir / "asset_profile.csv", index=False)
    sample = np.concatenate(reservoir, axis=0) if reservoir else np.empty((0, 2))
    target_sample = sample[:, 0] if len(sample) else np.array([])
    weight_sample = sample[:, 1] if len(sample) else np.array([])
    summary = {
        "rows": int(total_rows),
        "assets": int(len(asset_acc)),
        "features": int(len(feature_names)),
        "responders": int(len(responder_names)),
        "target": {"mean": target_sum / max(total_rows, 1), "std_approx": float(np.sqrt(max(target_sumsq / max(total_rows, 1) - (target_sum / max(total_rows, 1)) ** 2, 0.0))), "abs_mean": target_abs_sum / max(total_rows, 1), "sample_quantiles": {str(q): float(np.quantile(target_sample, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)} if len(target_sample) else {}},
        "weight": {"mean": weight_sum / max(total_rows, 1), "std_approx": float(np.sqrt(max(weight_sumsq / max(total_rows, 1) - (weight_sum / max(total_rows, 1)) ** 2, 0.0))), "corr_target": float((weight_target_sum / max(total_rows, 1) - (weight_sum / max(total_rows, 1)) * (target_sum / max(total_rows, 1))) / max(np.sqrt(max(weight_sumsq / max(total_rows, 1) - (weight_sum / max(total_rows, 1)) ** 2, 0.0) * max(target_sumsq / max(total_rows, 1) - (target_sum / max(total_rows, 1)) ** 2, 0.0)), 1e-12)), "weighted_abs_target_mean": weight_abs_target_sum / max(weight_sum, 1e-12), "weighted_target_square_sum": weight_target_sq_sum},
        "block_ranges": [{
            "partition": path.name,
            "time_min": int(min(pq.ParquetFile(path).metadata.row_group(i).column(pq.ParquetFile(path).schema_arrow.names.index("time_id")).statistics.min for i in range(pq.ParquetFile(path).metadata.num_row_groups))),
            "time_max": int(max(pq.ParquetFile(path).metadata.row_group(i).column(pq.ParquetFile(path).schema_arrow.names.index("time_id")).statistics.max for i in range(pq.ParquetFile(path).metadata.num_row_groups))),
        } for path in train_paths],
        "outputs": ["feature_profile.csv", "top_weight_related_features.csv", "top_target_related_features.csv", "responder_profile.csv", "asset_profile.csv", "analysis_summary.json"],
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
