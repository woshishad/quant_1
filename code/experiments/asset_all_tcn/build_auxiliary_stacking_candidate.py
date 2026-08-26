from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from train_auxiliary_stacking import (
    AuxiliaryMLP,
    normalize_aux_inputs,
    predict_auxiliary_mlp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a feature -> predicted auxiliary state -> target candidate submission."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("results/auxiliary_stacking_20260824"),
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=Path("results/auxiliary_stacking_audit_20260824/audit_report.json"),
    )
    parser.add_argument(
        "--base-test-predictions",
        type=Path,
        default=Path("results/blend_latest_regime_xgb_residual/final_test_predictions.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/auxiliary_stacking_candidate_20260824"),
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--predict-batch-size", type=int, default=16_384)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_component(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    output = values.astype(np.float32, copy=True)
    output -= mean.astype(np.float32)
    output /= scale.astype(np.float32)
    np.nan_to_num(output, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
    np.clip(output, -10.0, 10.0, out=output)
    return output


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def target_prediction(
    values: np.ndarray,
    ridge: object,
    booster: lgb.Booster,
    selection: dict[str, float],
) -> np.ndarray:
    ridge_prediction = np.asarray(ridge.predict(values), dtype=np.float64)
    residual_prediction = np.asarray(booster.predict(values), dtype=np.float64)
    return float(selection["shrink"]) * (
        ridge_prediction + float(selection["residual_weight"]) * residual_prediction
    )


def save_submission_zip(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(
        zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.write(csv_path, arcname="submission.csv")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    summary = json.loads((args.model_dir / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    external_audit = audit["latest_external_current_best"][
        "current_best_plus_selected_refit_delta"
    ]
    selected_variant = external_audit["selected_variant"]
    if selected_variant != "predicted_aux_only":
        raise ValueError(
            f"candidate builder expects predicted_aux_only, got {selected_variant}"
        )
    gamma = float(external_audit["gamma_fit_first_half"])

    checkpoint = torch.load(
        args.model_dir / "auxiliary_mlp.pt",
        map_location="cpu",
        weights_only=False,
    )
    feature_names = list(checkpoint["feature_names"])
    auxiliary_names = list(checkpoint["auxiliary_names"])
    auxiliary_mlp = AuxiliaryMLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    auxiliary_mlp.load_state_dict(checkpoint["state_dict"])
    auxiliary_mlp.to(device).eval()
    auxiliary_ridge = load_pickle(args.model_dir / "auxiliary_ridge.pkl")

    auxiliary_normalizers = np.load(args.model_dir / "auxiliary_normalizers.npz")
    input_mean = auxiliary_normalizers["input_mean"]
    input_scale = auxiliary_normalizers["input_scale"]
    component_normalizers = np.load(
        args.model_dir / "target_component_normalizers.npz"
    )

    raw_ridge = load_pickle(args.model_dir / "target_refit_raw_only_ridge.pkl")
    raw_booster = lgb.Booster(
        model_file=str(args.model_dir / "target_refit_raw_only_residual_lgbm.txt")
    )
    auxiliary_target_ridge = load_pickle(
        args.model_dir / "target_refit_predicted_aux_only_ridge.pkl"
    )
    auxiliary_target_booster = lgb.Booster(
        model_file=str(
            args.model_dir / "target_refit_predicted_aux_only_residual_lgbm.txt"
        )
    )
    raw_selection = summary["target_refit_variants"]["raw_only"]["frozen_selection"]
    auxiliary_selection = summary["target_refit_variants"]["predicted_aux_only"][
        "frozen_selection"
    ]
    target_feature_names = list(summary["features"]["target_feature_names"])

    sample_path = args.data_root / "sample_submission.csv"
    sample = pd.read_csv(sample_path, usecols=["row_id"])
    base = pd.read_csv(
        args.base_test_predictions, usecols=["row_id", "prediction"]
    )
    if base["row_id"].duplicated().any():
        raise ValueError("base test predictions contain duplicate row_id")
    if not base["row_id"].equals(sample["row_id"]):
        raise ValueError("base test prediction row order differs from sample_submission")
    base_by_row = pd.Series(
        base["prediction"].to_numpy(dtype=np.float64),
        index=base["row_id"].to_numpy(dtype=np.int64),
    )
    del base

    test_paths = sorted((args.data_root / "test").glob("test_partition_*.parquet"))
    if not test_paths:
        raise FileNotFoundError(args.data_root / "test")
    columns = ["row_id", "time_id", "asset_id", *feature_names]
    prediction_path = args.results_dir / "final_test_predictions.csv"
    all_row_ids: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    output_header = True
    processed_rows = 0
    started_at = time.perf_counter()

    for path in test_paths:
        parquet_file = pq.ParquetFile(path)
        print(f"predicting {path.name}: {parquet_file.metadata.num_rows} rows", flush=True)
        for batch in parquet_file.iter_batches(
            batch_size=args.batch_size, columns=columns
        ):
            frame = batch.to_pandas()
            row_id = frame["row_id"].to_numpy(dtype=np.int64, copy=True)
            asset_id = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
            raw_feature_values = frame[target_feature_names].to_numpy(
                dtype=np.float32, copy=True
            )
            raw_x = normalize_component(
                raw_feature_values,
                component_normalizers["raw_mean"],
                component_normalizers["raw_scale"],
            )
            raw_prediction = target_prediction(
                raw_x, raw_ridge, raw_booster, raw_selection
            )

            auxiliary_x = normalize_aux_inputs(
                frame[feature_names].to_numpy(dtype=np.float32, copy=True),
                asset_id,
                input_mean,
                input_scale,
                int(checkpoint["input_dim"]) - len(feature_names),
            )
            ridge_auxiliary = np.asarray(
                auxiliary_ridge.predict(auxiliary_x), dtype=np.float32
            )
            mlp_auxiliary = predict_auxiliary_mlp(
                auxiliary_mlp, auxiliary_x, args.predict_batch_size, device
            )
            ridge_component = normalize_component(
                ridge_auxiliary,
                component_normalizers["ridge_aux_mean"],
                component_normalizers["ridge_aux_scale"],
            )
            mlp_component = normalize_component(
                mlp_auxiliary,
                component_normalizers["mlp_aux_mean"],
                component_normalizers["mlp_aux_scale"],
            )
            auxiliary_target_x = np.concatenate(
                [ridge_component, mlp_component], axis=1
            )
            two_stage_prediction = target_prediction(
                auxiliary_target_x,
                auxiliary_target_ridge,
                auxiliary_target_booster,
                auxiliary_selection,
            )
            auxiliary_delta = two_stage_prediction - raw_prediction
            base_prediction = base_by_row.reindex(row_id).to_numpy(dtype=np.float64)
            if not np.isfinite(base_prediction).all():
                raise ValueError(f"base predictions missing for rows in {path}")
            candidate_prediction = base_prediction + gamma * auxiliary_delta
            if not np.isfinite(candidate_prediction).all():
                raise ValueError(f"non-finite candidate predictions in {path}")

            output = pd.DataFrame(
                {
                    "row_id": row_id,
                    "time_id": frame["time_id"].to_numpy(dtype=np.int64),
                    "asset_id": asset_id,
                    "base_prediction": base_prediction,
                    "raw_auxiliary_control_prediction": raw_prediction,
                    "two_stage_prediction": two_stage_prediction,
                    "auxiliary_delta": auxiliary_delta,
                    "selected_gamma": gamma,
                    "prediction": candidate_prediction,
                }
            )
            output.to_csv(
                prediction_path,
                mode="w" if output_header else "a",
                header=output_header,
                index=False,
            )
            output_header = False
            all_row_ids.append(row_id)
            all_predictions.append(candidate_prediction)
            processed_rows += len(output)
            print(f"processed {processed_rows} rows", flush=True)
            del frame, output, raw_x, raw_feature_values, auxiliary_x
            del ridge_auxiliary, mlp_auxiliary, ridge_component, mlp_component
            del auxiliary_target_x, raw_prediction, two_stage_prediction
            del auxiliary_delta, base_prediction, candidate_prediction
            gc.collect()

    predicted_row_ids = np.concatenate(all_row_ids)
    predicted_values = np.concatenate(all_predictions)
    if len(predicted_row_ids) != len(sample):
        raise ValueError(
            f"predicted {len(predicted_row_ids)} rows, expected {len(sample)}"
        )
    if len(np.unique(predicted_row_ids)) != len(predicted_row_ids):
        raise ValueError("candidate predictions contain duplicate row_id")
    prediction_by_row = pd.Series(predicted_values, index=predicted_row_ids)
    submission = sample.copy()
    submission["target"] = prediction_by_row.reindex(
        sample["row_id"].to_numpy(dtype=np.int64)
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(submission["target"].to_numpy()).all():
        raise ValueError("submission contains missing or non-finite target")
    submission_path = args.results_dir / "submission.csv"
    zip_path = args.results_dir / "submission.zip"
    submission.to_csv(submission_path, index=False)
    save_submission_zip(submission_path, zip_path)

    elapsed = time.perf_counter() - started_at
    metrics = {
        "strategy": "current_best_plus_feature_predicted_weight_responders_delta",
        "leakage_safe": True,
        "official_test_used_for": "inference_only",
        "forbidden_test_inputs": ["target", "weight", "responder_00..responder_46"],
        "test_inputs": ["row_id", "time_id", "asset_id", "feature_000..feature_322"],
        "auxiliary_outputs": auxiliary_names,
        "selected_auxiliary_variant": selected_variant,
        "formula": "prediction = current_best + gamma * (two_stage_prediction - raw_auxiliary_control_prediction)",
        "selected_gamma": gamma,
        "gamma_selection": {
            "fit_range": [868480, 878479],
            "frozen_holdout_range": [878480, 888479],
            "fit_gain": external_audit["gain_fit_first_half"],
            "frozen_holdout_gain": external_audit["gain_holdout_second_half"],
            "full_gain": external_audit["gain_full"],
            "base_full_score": external_audit["base_full"],
            "candidate_full_score": external_audit["candidate_full"],
            "holdout_optimal_gamma_diagnostic": external_audit[
                "gamma_optimal_second_half_diagnostic"
            ],
        },
        "promotion_status": "candidate_pending_online_validation_with_5_of_8_positive_time_blocks",
        "rows": int(len(submission)),
        "prediction_stats": {
            "mean": float(submission["target"].mean()),
            "std": float(submission["target"].std()),
            "min": float(submission["target"].min()),
            "max": float(submission["target"].max()),
        },
        "elapsed_seconds": elapsed,
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "outputs": {
            "final_test_predictions": str(prediction_path),
            "submission_csv": str(submission_path),
            "submission_zip": str(zip_path),
            "submission_csv_sha256": sha256_file(submission_path),
            "submission_zip_sha256": sha256_file(zip_path),
        },
    }
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
