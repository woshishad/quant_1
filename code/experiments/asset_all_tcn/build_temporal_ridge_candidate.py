from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import time
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from final_residual_train_predict_ts_features import (
    add_time_series_and_cross_section_features,
    historical_feature_indices,
)
from final_train_predict import (
    BASE_COLUMNS_TEST,
    BASE_COLUMNS_TRAIN,
    load_feature_ranking,
    parquet_paths,
    read_partitioned_frame,
    reorder_like_sample,
    schema_columns,
    time_range,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit and stream a strict causal temporal Ridge candidate."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--stable-features-file",
        type=Path,
        default=Path(
            "results/asset_all_stable_features_100k/selected_features_stable_top128.csv"
        ),
    )
    parser.add_argument(
        "--validation-audit",
        type=Path,
        default=Path("results/temporal_ridge_candidate_audit_20260824/audit_report.json"),
    )
    parser.add_argument(
        "--base-test-predictions",
        type=Path,
        default=Path(
            "results/auxiliary_stacking_candidate_20260824/final_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/temporal_ridge_candidate_20260824"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/temporal_ridge_candidate_20260824"),
    )
    parser.add_argument("--train-lookback-time-points", type=int, default=75_000)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--engineered-top-k", type=int, default=16)
    parser.add_argument("--market-history-top-k", type=int, default=8)
    parser.add_argument("--lag-steps", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--rolling-windows", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--rolling-min-period-frac", type=float, default=0.25)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--residual-ridge-alpha", type=float, default=10_000.0)
    parser.add_argument("--chunk-time-points", type=int, default=10_000)
    parser.add_argument("--test-end-time", type=int, default=None)
    return parser.parse_args()


def feature_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        disable_lag=False,
        disable_delta=False,
        disable_rolling=False,
        disable_cross_section=True,
        disable_market_history=False,
        lag_steps=[int(value) for value in args.lag_steps],
        rolling_windows=[int(value) for value in args.rolling_windows],
        rolling_min_period_frac=float(args.rolling_min_period_frac),
    )


def fit_normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0).astype(np.float32)
    scale = np.nanstd(values, axis=0).astype(np.float32)
    mean[~np.isfinite(mean)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return mean, scale


def normalize(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    output = values.astype(np.float32, copy=True)
    output -= mean
    output /= scale
    np.nan_to_num(output, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return output


def fit_weighted_ridge(
    values: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    alpha: float,
) -> Ridge:
    sample_weight = weight / max(float(np.mean(weight)), 1e-12)
    model = Ridge(alpha=float(alpha), solver="lsqr", max_iter=500)
    model.fit(values, target, sample_weight=sample_weight)
    return model


def keep_latest_times(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    unique_times = np.unique(frame["time_id"].to_numpy(dtype=np.int64))
    if len(unique_times) <= count:
        return frame.copy()
    keep_times = unique_times[-count:]
    return frame[frame["time_id"].isin(keep_times)].copy()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_zip(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(
        zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.write(csv_path, arcname="submission.csv")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    validation_audit = json.loads(
        args.validation_audit.read_text(encoding="utf-8")
    )
    if validation_audit.get("all_local_promotion_checks_passed") is not True:
        raise ValueError("temporal validation audit did not pass")
    temporal_gamma = float(validation_audit["temporal_gamma"])

    train_paths = parquet_paths(args.raw_data_dir, "train")
    test_paths = parquet_paths(args.raw_data_dir, "test")
    train_min_time, train_max_time = time_range(train_paths)
    test_min_time, available_test_max_time = time_range(test_paths)
    test_max_time = (
        min(available_test_max_time, int(args.test_end_time))
        if args.test_end_time is not None
        else available_test_max_time
    )
    history_size = max(
        max(args.lag_steps or [0]), max(args.rolling_windows or [0])
    )
    train_start_time = max(
        train_min_time,
        train_max_time - int(args.train_lookback_time_points) + 1,
    )
    history_start_time = max(train_min_time, train_start_time - history_size)

    available_columns = schema_columns(train_paths)
    ranking = load_feature_ranking(args.stable_features_file, available_columns)
    raw_features = (
        ranking.head(int(args.top_k))["feature_name"].astype(str).tolist()
    )
    engineered_features = (
        ranking.head(int(args.engineered_top_k))["feature_name"]
        .astype(str)
        .tolist()
    )
    market_history_features = engineered_features[
        : max(0, int(args.market_history_top_k))
    ]
    common_columns = ["row_id", "time_id", "asset_id", *raw_features]
    engineering_args = feature_args(args)

    print(
        f"training temporal Ridge on {train_start_time}..{train_max_time}; "
        f"history starts at {history_start_time}",
        flush=True,
    )
    raw_train = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + raw_features,
        min_time=history_start_time,
        max_time=train_max_time,
    )
    train_frame, model_features = add_time_series_and_cross_section_features(
        raw_train,
        raw_features,
        engineered_features,
        market_history_features,
        engineering_args,
    )
    historical_indices = historical_feature_indices(model_features, len(raw_features))
    historical_features = [model_features[index] for index in historical_indices]
    if not historical_features:
        raise ValueError("no historical features were generated")

    train_mask = train_frame["time_id"].to_numpy(dtype=np.int64) >= train_start_time
    target = train_frame.loc[train_mask, "target"].to_numpy(dtype=np.float32)
    weight = train_frame.loc[train_mask, "weight"].to_numpy(dtype=np.float32)

    raw_values = train_frame.loc[train_mask, raw_features].to_numpy(dtype=np.float32)
    raw_mean, raw_scale = fit_normalizer(raw_values)
    raw_values = normalize(raw_values, raw_mean, raw_scale)
    current_model = fit_weighted_ridge(
        raw_values, target, weight, args.ridge_alpha
    )
    current_prediction = current_model.predict(raw_values)
    del raw_values
    gc.collect()

    historical_values = train_frame.loc[
        train_mask, historical_features
    ].to_numpy(dtype=np.float32)
    historical_mean, historical_scale = fit_normalizer(historical_values)
    historical_values = normalize(
        historical_values, historical_mean, historical_scale
    )
    temporal_model = fit_weighted_ridge(
        historical_values,
        target - current_prediction,
        weight,
        args.residual_ridge_alpha,
    )
    del historical_values, current_prediction, target, weight
    gc.collect()

    history_raw = keep_latest_times(raw_train[common_columns], history_size)
    del raw_train, train_frame
    gc.collect()

    base_test = pd.read_csv(
        args.base_test_predictions, usecols=["row_id", "prediction"]
    )
    if base_test["row_id"].duplicated().any():
        raise ValueError("base test predictions contain duplicate row_id")
    base_by_row = pd.Series(
        base_test["prediction"].to_numpy(dtype=np.float64),
        index=base_test["row_id"].to_numpy(dtype=np.int64),
    )
    del base_test

    detailed_path = args.results_dir / "final_test_predictions.csv"
    wrote_header = False
    row_id_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    processed_rows = 0
    for chunk_start in range(
        test_min_time, test_max_time + 1, int(args.chunk_time_points)
    ):
        chunk_end = min(
            test_max_time, chunk_start + int(args.chunk_time_points) - 1
        )
        raw_chunk = read_partitioned_frame(
            test_paths,
            BASE_COLUMNS_TEST + raw_features,
            min_time=chunk_start,
            max_time=chunk_end,
        )
        if raw_chunk.empty:
            continue
        combined_raw = pd.concat(
            [history_raw, raw_chunk[common_columns]], ignore_index=True
        )
        chunk_frame, _ = add_time_series_and_cross_section_features(
            combined_raw,
            raw_features,
            engineered_features,
            market_history_features,
            engineering_args,
        )
        current_mask = (
            chunk_frame["time_id"].to_numpy(dtype=np.int64) >= chunk_start
        ) & (chunk_frame["time_id"].to_numpy(dtype=np.int64) <= chunk_end)
        prediction_frame = chunk_frame.loc[current_mask, BASE_COLUMNS_TEST].copy()
        historical_x = chunk_frame.loc[
            current_mask, historical_features
        ].to_numpy(dtype=np.float32)
        historical_x = normalize(
            historical_x, historical_mean, historical_scale
        )
        temporal_signal = temporal_model.predict(historical_x).astype(np.float64)
        row_id = prediction_frame["row_id"].to_numpy(dtype=np.int64)
        base_prediction = base_by_row.reindex(row_id).to_numpy(dtype=np.float64)
        if not np.isfinite(base_prediction).all():
            raise ValueError(f"base predictions missing in chunk {chunk_start}..{chunk_end}")
        prediction = base_prediction + temporal_gamma * temporal_signal
        if not np.isfinite(prediction).all():
            raise ValueError(f"non-finite prediction in chunk {chunk_start}..{chunk_end}")

        prediction_frame["base_prediction"] = base_prediction
        prediction_frame["temporal_residual_signal"] = temporal_signal
        prediction_frame["temporal_gamma"] = temporal_gamma
        prediction_frame["prediction"] = prediction
        prediction_frame.to_csv(
            detailed_path,
            mode="a" if wrote_header else "w",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True
        row_id_parts.append(row_id)
        prediction_parts.append(prediction)
        processed_rows += len(prediction_frame)
        print(
            f"processed {processed_rows} rows through time_id={chunk_end}",
            flush=True,
        )

        history_raw = keep_latest_times(combined_raw, history_size)
        del raw_chunk, combined_raw, chunk_frame, prediction_frame
        del historical_x, temporal_signal, base_prediction, prediction
        gc.collect()

    predicted_row_id = np.concatenate(row_id_parts)
    predicted_values = np.concatenate(prediction_parts)
    if len(np.unique(predicted_row_id)) != len(predicted_row_id):
        raise ValueError("temporal candidate contains duplicate row_id")
    prediction_by_row = pd.DataFrame(
        {"row_id": predicted_row_id, "target": predicted_values}
    )
    sample_path = args.raw_data_dir / "sample_submission.csv"
    submission = reorder_like_sample(prediction_by_row, sample_path)
    if len(submission) == len(prediction_by_row) and submission["target"].isna().any():
        raise ValueError("submission is missing predictions after row_id alignment")
    submission_path = args.results_dir / "submission.csv"
    zip_path = args.results_dir / "submission.zip"
    submission.to_csv(submission_path, index=False)
    save_zip(submission_path, zip_path)

    with (args.model_dir / "temporal_ridge_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "current_model": current_model,
                "temporal_model": temporal_model,
                "raw_features": raw_features,
                "engineered_features": engineered_features,
                "market_history_features": market_history_features,
                "historical_features": historical_features,
                "raw_mean": raw_mean,
                "raw_scale": raw_scale,
                "historical_mean": historical_mean,
                "historical_scale": historical_scale,
                "temporal_gamma": temporal_gamma,
                "history_size": history_size,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    full_test = args.test_end_time is None or test_max_time == available_test_max_time
    expected_rows = len(pd.read_csv(sample_path, usecols=["row_id"]))
    metrics = {
        "strategy": "current_candidate_plus_refit_strict_causal_temporal_ridge_residual",
        "leakage_safe": True,
        "official_test_used_for": "sequential_inference_only",
        "formula": "prediction = current_candidate + temporal_gamma * temporal_ridge_residual",
        "temporal_gamma": temporal_gamma,
        "validation_audit": str(args.validation_audit),
        "validation_status": validation_audit["status"],
        "train_range": [train_start_time, train_max_time],
        "test_range": [test_min_time, test_max_time],
        "history_size": history_size,
        "feature_count": {
            "raw": len(raw_features),
            "asset_history_base": len(engineered_features),
            "market_history_base": len(market_history_features),
            "historical_model_inputs": len(historical_features),
        },
        "rows": int(len(submission)),
        "expected_full_test_rows": expected_rows,
        "full_test_completed": full_test and len(submission) == expected_rows,
        "prediction_stats": {
            "mean": float(submission["target"].mean()),
            "std": float(submission["target"].std()),
            "min": float(submission["target"].min()),
            "max": float(submission["target"].max()),
            "finite": int(np.isfinite(submission["target"]).sum()),
        },
        "elapsed_seconds": time.perf_counter() - started_at,
        "outputs": {
            "detailed_predictions": str(detailed_path),
            "submission_csv": str(submission_path),
            "submission_zip": str(zip_path),
            "model": str(args.model_dir / "temporal_ridge_model.pkl"),
            "submission_csv_sha256": sha256_file(submission_path),
            "submission_zip_sha256": sha256_file(zip_path),
        },
    }
    metrics_path = args.results_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
