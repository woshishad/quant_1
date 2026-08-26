from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from audit_submission_package import (
    DEFAULT_SAMPLE_SUBMISSION,
    sha256_file,
    validate_submission_csv,
    validate_zip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit the strict causal temporal candidate."
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("results/temporal_ridge_candidate_20260824"),
    )
    parser.add_argument(
        "--base-test-predictions",
        type=Path,
        default=Path(
            "results/auxiliary_stacking_candidate_20260824/final_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--validation-audit",
        type=Path,
        default=Path("results/temporal_ridge_candidate_audit_20260824/audit_report.json"),
    )
    parser.add_argument(
        "--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION
    )
    return parser.parse_args()


def check(passed: bool, detail: str) -> dict[str, bool | str]:
    return {"passed": bool(passed), "detail": detail}


def main() -> None:
    args = parse_args()
    metrics_path = args.candidate_dir / "metrics.json"
    detailed_path = args.candidate_dir / "final_test_predictions.csv"
    submission_path = args.candidate_dir / "submission.csv"
    zip_path = args.candidate_dir / "submission.zip"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validation_audit = json.loads(
        args.validation_audit.read_text(encoding="utf-8")
    )

    _, csv_audit = validate_submission_csv(
        submission_path, args.sample_submission
    )
    zip_audit = validate_zip(zip_path, submission_path)
    expected_columns = [
        "row_id",
        "time_id",
        "asset_id",
        "base_prediction",
        "temporal_residual_signal",
        "temporal_gamma",
        "prediction",
    ]
    detailed = pd.read_csv(detailed_path, usecols=expected_columns)
    base = pd.read_csv(
        args.base_test_predictions, usecols=["row_id", "prediction"]
    )
    sample = pd.read_csv(args.sample_submission, usecols=["row_id"])

    detailed_row_id = detailed["row_id"].to_numpy(dtype=np.int64)
    base_row_id = base["row_id"].to_numpy(dtype=np.int64)
    sample_row_id = sample["row_id"].to_numpy(dtype=np.int64)
    base_prediction = detailed["base_prediction"].to_numpy(dtype=np.float64)
    signal = detailed["temporal_residual_signal"].to_numpy(dtype=np.float64)
    gamma = detailed["temporal_gamma"].to_numpy(dtype=np.float64)
    prediction = detailed["prediction"].to_numpy(dtype=np.float64)
    formula_prediction = base_prediction + gamma * signal
    formula_error = float(np.max(np.abs(formula_prediction - prediction)))
    base_error = float(
        np.max(np.abs(base["prediction"].to_numpy(dtype=np.float64) - base_prediction))
    )
    metric_gamma = float(metrics["temporal_gamma"])
    validation_gamma = float(validation_audit["temporal_gamma"])

    detailed_checks = {
        "detailed_rows_match_sample": check(
            len(detailed) == len(sample),
            f"detailed={len(detailed)}, sample={len(sample)}",
        ),
        "detailed_row_order_matches_sample": check(
            np.array_equal(detailed_row_id, sample_row_id),
            "detailed row_id order must match sample_submission",
        ),
        "base_row_order_matches_detailed": check(
            np.array_equal(base_row_id, detailed_row_id),
            "base and temporal detailed row_id order must match",
        ),
        "base_prediction_exact": check(
            base_error <= 1e-12, f"max_abs_error={base_error}"
        ),
        "final_formula_exact": check(
            formula_error <= 1e-12, f"max_abs_error={formula_error}"
        ),
        "all_detailed_values_finite": check(
            bool(
                np.isfinite(base_prediction).all()
                and np.isfinite(signal).all()
                and np.isfinite(gamma).all()
                and np.isfinite(prediction).all()
            ),
            "base, temporal signal, gamma and prediction must be finite",
        ),
        "gamma_constant_and_frozen_from_validation": check(
            bool(
                np.allclose(gamma, metric_gamma, rtol=0.0, atol=1e-15)
                and abs(metric_gamma - validation_gamma) <= 1e-15
            ),
            f"metrics={metric_gamma}, validation={validation_gamma}",
        ),
        "validation_audit_passed": check(
            validation_audit.get("all_local_promotion_checks_passed") is True,
            f"status={validation_audit.get('status')}",
        ),
        "test_was_sequential_inference_only": check(
            metrics.get("official_test_used_for") == "sequential_inference_only",
            f"official_test_used_for={metrics.get('official_test_used_for')}",
        ),
        "full_test_completed": check(
            metrics.get("full_test_completed") is True,
            f"full_test_completed={metrics.get('full_test_completed')}",
        ),
        "model_artifact_exists": check(
            Path(metrics["outputs"]["model"]).exists(),
            str(metrics["outputs"]["model"]),
        ),
    }
    all_checks = {
        **{name: value["passed"] for name, value in csv_audit["checks"].items()},
        **{name: value["passed"] for name, value in zip_audit["checks"].items()},
        **{name: value["passed"] for name, value in detailed_checks.items()},
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_dir": str(args.candidate_dir),
        "overall_structural_audit_passed": all(all_checks.values()),
        "failed_checks": [name for name, passed in all_checks.items() if not passed],
        "csv_audit": csv_audit,
        "zip_audit": zip_audit,
        "detailed_checks": detailed_checks,
        "hashes": {
            "metrics": sha256_file(metrics_path),
            "final_test_predictions": sha256_file(detailed_path),
            "submission_csv": sha256_file(submission_path),
            "submission_zip": sha256_file(zip_path),
        },
    }
    output_path = args.candidate_dir / "audit_report.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["overall_structural_audit_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
