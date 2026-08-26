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
        description="Independently audit an auxiliary-stacking candidate submission."
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("results/auxiliary_stacking_candidate_20260824"),
    )
    parser.add_argument(
        "--base-test-predictions",
        type=Path,
        default=Path("results/blend_latest_regime_xgb_residual/final_test_predictions.csv"),
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

    _, csv_audit = validate_submission_csv(
        submission_path, args.sample_submission
    )
    zip_audit = validate_zip(zip_path, submission_path)
    detailed = pd.read_csv(detailed_path)
    base = pd.read_csv(
        args.base_test_predictions, usecols=["row_id", "prediction"]
    )
    sample = pd.read_csv(args.sample_submission, usecols=["row_id"])

    expected_columns = [
        "row_id",
        "time_id",
        "asset_id",
        "base_prediction",
        "raw_auxiliary_control_prediction",
        "two_stage_prediction",
        "auxiliary_delta",
        "selected_gamma",
        "prediction",
    ]
    numeric = detailed[expected_columns].to_numpy(dtype=np.float64)
    formula_prediction = detailed["base_prediction"].to_numpy(dtype=np.float64) + (
        detailed["selected_gamma"].to_numpy(dtype=np.float64)
        * detailed["auxiliary_delta"].to_numpy(dtype=np.float64)
    )
    formula_max_abs_error = float(
        np.max(
            np.abs(
                formula_prediction
                - detailed["prediction"].to_numpy(dtype=np.float64)
            )
        )
    )
    base_max_abs_error = float(
        np.max(
            np.abs(
                base["prediction"].to_numpy(dtype=np.float64)
                - detailed["base_prediction"].to_numpy(dtype=np.float64)
            )
        )
    )
    auxiliary_delta_max_abs_error = float(
        np.max(
            np.abs(
                detailed["two_stage_prediction"].to_numpy(dtype=np.float64)
                - detailed[
                    "raw_auxiliary_control_prediction"
                ].to_numpy(dtype=np.float64)
                - detailed["auxiliary_delta"].to_numpy(dtype=np.float64)
            )
        )
    )
    forbidden_columns = [
        name
        for name in detailed.columns
        if name == "target" or name == "weight" or name.startswith("responder_")
    ]
    gamma_values = detailed["selected_gamma"].to_numpy(dtype=np.float64)
    selected_gamma = float(metrics["selected_gamma"])

    detailed_checks = {
        "detailed_columns_exact": check(
            detailed.columns.tolist() == expected_columns,
            f"columns={detailed.columns.tolist()}",
        ),
        "detailed_rows_match_sample": check(
            len(detailed) == len(sample),
            f"detailed={len(detailed)}, sample={len(sample)}",
        ),
        "detailed_row_order_matches_sample": check(
            detailed["row_id"].equals(sample["row_id"]),
            "detailed row_id order must match sample_submission",
        ),
        "base_row_order_matches_detailed": check(
            base["row_id"].equals(detailed["row_id"]),
            "base and candidate row_id order must match",
        ),
        "all_detailed_values_finite": check(
            bool(np.isfinite(numeric).all()), "all detailed numeric values must be finite"
        ),
        "base_prediction_exact": check(
            base_max_abs_error <= 1e-12,
            f"max_abs_error={base_max_abs_error}",
        ),
        "auxiliary_delta_formula_exact": check(
            auxiliary_delta_max_abs_error <= 1e-12,
            f"max_abs_error={auxiliary_delta_max_abs_error}",
        ),
        "final_formula_exact": check(
            formula_max_abs_error <= 1e-12,
            f"max_abs_error={formula_max_abs_error}",
        ),
        "gamma_constant_and_matches_metrics": check(
            bool(np.allclose(gamma_values, selected_gamma, rtol=0.0, atol=1e-15)),
            f"metrics_gamma={selected_gamma}",
        ),
        "no_forbidden_test_columns": check(
            not forbidden_columns, f"forbidden_columns={forbidden_columns}"
        ),
        "test_used_for_inference_only": check(
            metrics.get("official_test_used_for") == "inference_only",
            f"official_test_used_for={metrics.get('official_test_used_for')}",
        ),
        "candidate_not_mislabeled_as_promoted": check(
            str(metrics.get("promotion_status", "")).startswith("candidate_"),
            f"promotion_status={metrics.get('promotion_status')}",
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
        "model_promotion_status": metrics.get("promotion_status"),
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
