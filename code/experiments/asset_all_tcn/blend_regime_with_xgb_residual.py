from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_RAW_DATA_DIR = Path(
    "data/raw/public_release_20260630/public_release_20260630/data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an out-of-time XGBoost residual coefficient on the first half of "
            "calibration, then apply it to the current regime submission."
        )
    )
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--xgb-strategy-zip", type=Path, default=Path("Quant-main.zip"))
    parser.add_argument(
        "--current-calibration",
        type=Path,
        default=Path(
            "results/regime_classification_market_75k_probe/calibration_predictions.csv"
        ),
    )
    parser.add_argument(
        "--current-test-predictions",
        type=Path,
        default=Path(
            "results/final_latest_regime_classification_model/final_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--current-submission",
        type=Path,
        default=Path("results/final_latest_regime_classification_model/submission.csv"),
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR / "sample_submission.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/blend_latest_regime_xgb_residual"),
    )
    parser.add_argument("--beta-min", type=float, default=-0.25)
    parser.add_argument("--beta-max", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--block-count", type=int, default=8)
    parser.add_argument("--save-xgb-predictions", action="store_true")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_zero_mean_r2(
    target: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> float:
    numerator = float(np.sum(weight * np.square(target - prediction)))
    denominator = float(np.sum(weight * np.square(target)))
    return float(1.0 - numerator / max(denominator, 1e-12))


def fit_residual_beta(
    target: np.ndarray,
    base_prediction: np.ndarray,
    residual_signal: np.ndarray,
    weight: np.ndarray,
) -> float:
    denominator = float(np.sum(weight * np.square(residual_signal)))
    if denominator <= 1e-12:
        raise ValueError("XGBoost residual signal has zero weighted variance")
    numerator = float(
        np.sum(weight * residual_signal * (target - base_prediction))
    )
    return numerator / denominator


def score_mask(
    frame: pd.DataFrame, prediction: np.ndarray, mask: np.ndarray
) -> float:
    return weighted_zero_mean_r2(
        frame.loc[mask, "target"].to_numpy(dtype=np.float64),
        prediction[mask],
        frame.loc[mask, "weight"].to_numpy(dtype=np.float64),
    )


def score_time_blocks(
    frame: pd.DataFrame,
    base_prediction: np.ndarray,
    selected_prediction: np.ndarray,
    block_count: int,
) -> list[dict[str, float | int]]:
    time_id = frame["time_id"].to_numpy(dtype=np.int64)
    rows = []
    for block, block_times in enumerate(
        np.array_split(np.unique(time_id), int(block_count))
    ):
        mask = np.isin(time_id, block_times)
        base_score = score_mask(frame, base_prediction, mask)
        selected_score = score_mask(frame, selected_prediction, mask)
        rows.append(
            {
                "block": int(block),
                "time_min": int(block_times[0]),
                "time_max": int(block_times[-1]),
                "rows": int(np.sum(mask)),
                "base_score": base_score,
                "selected_score": selected_score,
                "improvement": selected_score - base_score,
            }
        )
    return rows


def find_strategy_dir(extract_root: Path) -> Path:
    candidates = sorted(extract_root.rglob("xgb_strategy/main.py"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one xgb_strategy/main.py, found {len(candidates)}"
        )
    return candidates[0].parent


@contextmanager
def load_xgb_strategy(archive_path: Path) -> Iterator[object]:
    with tempfile.TemporaryDirectory(prefix="quant_xgb_strategy_") as temp_dir:
        extract_root = Path(temp_dir)
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"invalid XGBoost strategy archive: {archive_path}")
            archive.extractall(extract_root)
        strategy_dir = find_strategy_dir(extract_root)
        module_path = strategy_dir / "main.py"
        module_name = f"_quant_xgb_residual_{abs(hash(module_path))}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"could not import {module_path}")
        module = importlib.util.module_from_spec(spec)
        old_path = list(sys.path)
        sys.path.insert(0, str(strategy_dir))
        try:
            spec.loader.exec_module(module)
            yield module.Model()
        finally:
            sys.path = old_path


def parquet_time_bounds(parquet_file: pq.ParquetFile) -> tuple[int, int] | None:
    metadata = parquet_file.metadata
    column_index = None
    for index in range(metadata.num_columns):
        if metadata.schema.column(index).name == "time_id":
            column_index = index
            break
    if column_index is None:
        return None
    minimums = []
    maximums = []
    for row_group in range(metadata.num_row_groups):
        statistics = metadata.row_group(row_group).column(column_index).statistics
        if statistics is None or not statistics.has_min_max:
            return None
        minimums.append(int(statistics.min))
        maximums.append(int(statistics.max))
    return min(minimums), max(maximums)


def predict_parquet_files(
    model: object,
    paths: list[Path],
    batch_size: int,
    min_time_id: int | None = None,
    max_time_id: int | None = None,
) -> pd.DataFrame:
    key_columns = ["row_id", "time_id", "asset_id"]
    feature_columns = list(model.raw_feature_columns)
    read_columns = key_columns + feature_columns
    outputs = []
    predicted_rows = 0
    for file_index, path in enumerate(paths, start=1):
        parquet_file = pq.ParquetFile(path)
        bounds = parquet_time_bounds(parquet_file)
        if bounds is not None:
            if min_time_id is not None and bounds[1] < min_time_id:
                continue
            if max_time_id is not None and bounds[0] > max_time_id:
                continue
        print(
            f"XGB predict file {file_index}/{len(paths)}: {path.name}, bounds={bounds}",
            flush=True,
        )
        for batch in parquet_file.iter_batches(
            columns=read_columns, batch_size=int(batch_size)
        ):
            frame = batch.to_pandas(split_blocks=True, self_destruct=True)
            mask = np.ones(len(frame), dtype=bool)
            if min_time_id is not None:
                mask &= frame["time_id"].to_numpy(dtype=np.int64) >= min_time_id
            if max_time_id is not None:
                mask &= frame["time_id"].to_numpy(dtype=np.int64) <= max_time_id
            if not np.any(mask):
                continue
            frame = frame.loc[mask].reset_index(drop=True)
            prediction = np.asarray(model.predict(frame), dtype=np.float64)
            if len(prediction) != len(frame) or not np.all(np.isfinite(prediction)):
                raise ValueError(f"invalid XGBoost prediction for {path}")
            output = frame[key_columns].copy()
            output["xgb_prediction"] = prediction
            outputs.append(output)
            predicted_rows += len(output)
        print(f"XGB predicted rows: {predicted_rows}", flush=True)
    if not outputs:
        raise ValueError("no rows were predicted by the XGBoost strategy")
    result = pd.concat(outputs, ignore_index=True)
    if result["row_id"].duplicated().any():
        raise ValueError("duplicate row_id in XGBoost predictions")
    return result.sort_values("row_id", kind="mergesort").reset_index(drop=True)


def align_calibration(current_path: Path, xgb: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(
        current_path,
        usecols=["row_id", "time_id", "asset_id", "target", "weight", "prediction"],
    ).rename(columns={"prediction": "base_prediction"})
    current = current.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    merged = current.merge(
        xgb,
        on="row_id",
        how="inner",
        suffixes=("", "_xgb"),
        validate="one_to_one",
    )
    if len(merged) != len(current):
        raise ValueError("XGBoost calibration predictions do not cover current calibration")
    for key in ["time_id", "asset_id"]:
        if not np.array_equal(
            merged[key].to_numpy(), merged[f"{key}_xgb"].to_numpy()
        ):
            raise ValueError(f"calibration {key} mismatch")
    return merged.drop(columns=["time_id_xgb", "asset_id_xgb"])


def align_test(
    current_test_path: Path,
    current_submission_path: Path,
    xgb: pd.DataFrame,
) -> pd.DataFrame:
    keys = pd.read_csv(
        current_test_path, usecols=["row_id", "time_id", "asset_id"]
    )
    current = pd.read_csv(current_submission_path).rename(
        columns={"target": "base_prediction"}
    )
    current = keys.merge(current, on="row_id", how="inner", validate="one_to_one")
    merged = current.merge(
        xgb,
        on="row_id",
        how="inner",
        suffixes=("", "_xgb"),
        validate="one_to_one",
    )
    if len(merged) != len(current):
        raise ValueError("XGBoost test predictions do not cover current submission")
    for key in ["time_id", "asset_id"]:
        if not np.array_equal(
            merged[key].to_numpy(), merged[f"{key}_xgb"].to_numpy()
        ):
            raise ValueError(f"test {key} mismatch")
    return merged.drop(columns=["time_id_xgb", "asset_id_xgb"])


def save_submission_zip(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")


def prediction_stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "finite_count": int(np.sum(np.isfinite(values))),
        "null_count": int(np.sum(~np.isfinite(values))),
    }


def main() -> None:
    args = parse_args()
    if args.beta_min > args.beta_max:
        raise ValueError("--beta-min must be <= --beta-max")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    current_calibration = pd.read_csv(
        args.current_calibration, usecols=["time_id"]
    )
    unique_calibration_times = np.unique(
        current_calibration["time_id"].to_numpy(dtype=np.int64)
    )
    calibration_min = int(unique_calibration_times[0])
    calibration_max = int(unique_calibration_times[-1])
    split_time = int(unique_calibration_times[len(unique_calibration_times) // 2])
    del current_calibration

    train_paths = sorted((args.raw_data_dir / "train").glob("*.parquet"))
    test_paths = sorted((args.raw_data_dir / "test").glob("*.parquet"))
    if not train_paths or not test_paths:
        raise FileNotFoundError(f"missing parquet data under {args.raw_data_dir}")

    with load_xgb_strategy(args.xgb_strategy_zip) as xgb_model:
        xgb_calibration = predict_parquet_files(
            xgb_model,
            train_paths,
            args.batch_size,
            min_time_id=calibration_min,
            max_time_id=calibration_max,
        )
        calibration = align_calibration(args.current_calibration, xgb_calibration)
        fit_mask = calibration["time_id"].to_numpy(dtype=np.int64) < split_time
        holdout_mask = ~fit_mask
        target = calibration["target"].to_numpy(dtype=np.float64)
        weight = calibration["weight"].to_numpy(dtype=np.float64)
        base_prediction = calibration["base_prediction"].to_numpy(dtype=np.float64)
        xgb_signal = calibration["xgb_prediction"].to_numpy(dtype=np.float64)
        raw_beta = fit_residual_beta(
            target[fit_mask],
            base_prediction[fit_mask],
            xgb_signal[fit_mask],
            weight[fit_mask],
        )
        selected_beta = float(np.clip(raw_beta, args.beta_min, args.beta_max))
        selected_calibration = base_prediction + selected_beta * xgb_signal

        base_first_score = score_mask(calibration, base_prediction, fit_mask)
        base_holdout_score = score_mask(calibration, base_prediction, holdout_mask)
        base_full_score = weighted_zero_mean_r2(target, base_prediction, weight)
        selected_first_score = score_mask(
            calibration, selected_calibration, fit_mask
        )
        selected_holdout_score = score_mask(
            calibration, selected_calibration, holdout_mask
        )
        selected_full_score = weighted_zero_mean_r2(
            target, selected_calibration, weight
        )
        xgb_first_score = score_mask(calibration, xgb_signal, fit_mask)
        xgb_holdout_score = score_mask(calibration, xgb_signal, holdout_mask)
        xgb_full_score = weighted_zero_mean_r2(target, xgb_signal, weight)
        block_scores = score_time_blocks(
            calibration,
            base_prediction,
            selected_calibration,
            args.block_count,
        )

        calibration_output = calibration[
            ["row_id", "time_id", "asset_id", "target", "weight"]
        ].copy()
        calibration_output["base_prediction"] = base_prediction
        calibration_output["xgb_prediction"] = xgb_signal
        calibration_output["selected_beta"] = selected_beta
        calibration_output["prediction"] = selected_calibration
        calibration_output["error"] = selected_calibration - target
        calibration_output.to_csv(
            args.results_dir / "calibration_predictions.csv", index=False
        )
        pd.DataFrame(block_scores).to_csv(
            args.results_dir / "block_scores.csv", index=False
        )
        if args.save_xgb_predictions:
            xgb_calibration.to_csv(
                args.results_dir / "xgb_calibration_predictions.csv", index=False
            )

        xgb_test = predict_parquet_files(
            xgb_model, test_paths, args.batch_size
        )

    test = align_test(
        args.current_test_predictions, args.current_submission, xgb_test
    )
    base_test_prediction = test["base_prediction"].to_numpy(dtype=np.float64)
    xgb_test_signal = test["xgb_prediction"].to_numpy(dtype=np.float64)
    final_prediction = base_test_prediction + selected_beta * xgb_test_signal
    if not np.all(np.isfinite(final_prediction)):
        raise ValueError("final prediction contains NaN or infinite values")

    test_output = test[
        ["row_id", "time_id", "asset_id", "base_prediction", "xgb_prediction"]
    ].copy()
    test_output["selected_beta"] = selected_beta
    test_output["prediction"] = final_prediction
    test_output.to_csv(args.results_dir / "final_test_predictions.csv", index=False)
    if args.save_xgb_predictions:
        xgb_test.to_csv(args.results_dir / "xgb_test_predictions.csv", index=False)

    sample = pd.read_csv(args.sample_submission, usecols=["row_id"])
    prediction_frame = pd.DataFrame(
        {"row_id": test["row_id"], "target": final_prediction}
    )
    submission = sample.merge(
        prediction_frame, on="row_id", how="left", validate="one_to_one"
    )
    if submission["target"].isna().any() or not np.all(
        np.isfinite(submission["target"].to_numpy(dtype=np.float64))
    ):
        raise ValueError("submission contains missing or non-finite target values")
    submission_path = args.results_dir / "submission.csv"
    zip_path = args.results_dir / "submission.zip"
    submission.to_csv(submission_path, index=False)
    save_submission_zip(submission_path, zip_path)

    ranking = pd.DataFrame(
        [
            {
                "name": "final_latest_regime_classification_model",
                "label": "regime_soft_conditional_beta_0p20",
                "selected_residual_beta": 0.0,
                "selection_rule": "existing first-half selected regime model",
                "full_calibration": base_full_score,
                "holdout_half": base_holdout_score,
                "submission_zip": str(
                    Path("results/final_latest_regime_classification_model/submission.zip")
                ),
            },
            {
                "name": args.results_dir.name,
                "label": "regime_plus_overtime_xgb_residual",
                "selected_residual_beta": selected_beta,
                "selection_rule": "fit residual beta on calibration first half",
                "full_calibration": selected_full_score,
                "holdout_half": selected_holdout_score,
                "submission_zip": str(zip_path),
            },
        ]
    )
    ranking.to_csv(args.results_dir / "candidate_ranking.csv", index=False)

    reproduce_command = (
        f"{sys.executable} -u code/experiments/asset_all_tcn/"
        f"blend_regime_with_xgb_residual.py --results-dir {args.results_dir}"
    )
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_safe": True,
        "strategy": "regime_composite_with_overtime_xgb_residual",
        "selection_rule": (
            "fit one XGBoost residual beta on the first half of calibration; "
            "freeze it for the second half and official test"
        ),
        "selected_shrink": None,
        "selected_residual_beta": selected_beta,
        "raw_fitted_residual_beta": raw_beta,
        "residual_beta_bounds": [float(args.beta_min), float(args.beta_max)],
        "current_calibration_file": str(args.current_calibration),
        "current_test_file": str(args.current_test_predictions),
        "xgb_strategy_zip": str(args.xgb_strategy_zip),
        "xgb_strategy_zip_sha256": sha256_file(args.xgb_strategy_zip),
        "xgb_training_guard": {
            "train_partitions": "0-6",
            "validation_partitions": "7-8",
            "residual_fit_time": [calibration_min, split_time - 1],
            "residual_holdout_time": [split_time, calibration_max],
            "official_test_used_for": "prediction_only",
        },
        "calibration": {
            "rows": int(len(calibration)),
            "time_min": calibration_min,
            "time_max": calibration_max,
            "split_time": split_time,
            "base_first_half_score": base_first_score,
            "base_holdout_half_score": base_holdout_score,
            "base_full_score": base_full_score,
            "selected_first_half_score": selected_first_score,
            "selected_holdout_half_score": selected_holdout_score,
            "selected_full_score": selected_full_score,
            "holdout_improvement": selected_holdout_score - base_holdout_score,
            "full_improvement": selected_full_score - base_full_score,
            "xgb_first_half_score": xgb_first_score,
            "xgb_holdout_half_score": xgb_holdout_score,
            "xgb_full_score": xgb_full_score,
        },
        "raw_full_calibration_score": base_full_score,
        "selected_full_calibration_score": selected_full_score,
        "block_scores": block_scores,
        "test_prediction_stats": prediction_stats(final_prediction),
        "submission": {
            "rows": int(len(submission)),
            "columns": submission.columns.tolist(),
            "row_order_matches_sample": bool(
                submission["row_id"].equals(sample["row_id"])
            ),
            "csv_sha256": sha256_file(submission_path),
            "zip_sha256": sha256_file(zip_path),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "reproduce_command": reproduce_command,
        "output_files": {
            "calibration_predictions": str(
                args.results_dir / "calibration_predictions.csv"
            ),
            "block_scores": str(args.results_dir / "block_scores.csv"),
            "final_test_predictions": str(
                args.results_dir / "final_test_predictions.csv"
            ),
            "submission": str(submission_path),
            "submission_zip": str(zip_path),
            "candidate_ranking": str(
                args.results_dir / "candidate_ranking.csv"
            ),
        },
    }
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
