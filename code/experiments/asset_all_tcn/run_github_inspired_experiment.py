from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROFILES = {
    "long_horizon_rolling": {
        "description": (
            "Strictly causal 32/256/1000-step asset and market rolling features "
            "with a regularized Ridge residual."
        ),
        "sources": [
            "https://github.com/evgeniavolkova/kagglejanestreet",
            "https://github.com/osyuksel/kaggle-optiver-2024",
        ],
        "training_args": [
            "--rolling-windows",
            "32",
            "256",
            "1000",
            "--disable-lag",
            "--disable-delta",
            "--disable-cross-section",
            "--ridge-raw-only",
            "--residual-model",
            "ridge",
            "--residual-feature-set",
            "historical",
            "--residual-ridge-alpha",
            "10000",
        ],
    },
    "cross_sectional_rank": {
        "description": (
            "Same-time market mean, z-score, de-meaned value, and percentile rank "
            "features with a regularized Ridge residual."
        ),
        "sources": [
            "https://github.com/evgeniavolkova/kagglejanestreet",
            "https://github.com/osyuksel/kaggle-optiver-2024",
        ],
        "training_args": [
            "--disable-lag",
            "--disable-delta",
            "--disable-rolling",
            "--disable-market-history",
            "--ridge-raw-only",
            "--residual-model",
            "ridge",
            "--residual-feature-set",
            "engineered",
            "--residual-ridge-alpha",
            "10000",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and audit a leakage-safe experiment inspired by public repositories."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=Path(
            "results/temporal_ridge_candidate_audit_20260824/"
            "validation_predictions.csv"
        ),
    )
    parser.add_argument(
        "--reuse-training-output",
        action="store_true",
        help="Skip training and audit an existing calibration output directory.",
    )
    return parser.parse_args()


def build_training_command(
    python_executable: str,
    trainer_path: Path,
    profile_name: str,
    results_dir: Path,
    model_dir: Path,
) -> list[str]:
    profile = PROFILES[profile_name]
    return [
        python_executable,
        "-u",
        str(trainer_path),
        "--results-dir",
        str(results_dir),
        "--model-dir",
        str(model_dir),
        "--skip-test-prediction",
        "--no-save-models",
        *profile["training_args"],
    ]


def build_audit_command(
    python_executable: str,
    auditor_path: Path,
    run_id: str,
    base_predictions: Path,
    results_dir: Path,
    audit_dir: Path,
) -> list[str]:
    return [
        python_executable,
        "-u",
        str(auditor_path),
        "--candidate-name",
        run_id,
        "--base-predictions",
        str(base_predictions),
        "--signal-predictions",
        str(results_dir / "calibration_predictions.csv"),
        "--signal-metrics",
        str(results_dir / "metrics.json"),
        "--results-dir",
        str(audit_dir),
    ]


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"github_inspired_{args.profile}"
    results_dir = args.results_root / run_id
    model_dir = args.models_root / run_id
    audit_dir = args.results_root / f"{run_id}_audit"
    script_dir = Path(__file__).resolve().parent
    trainer_path = script_dir / "final_residual_train_predict_ts_features.py"
    auditor_path = script_dir / "audit_frozen_signal_candidate.py"

    training_command = build_training_command(
        sys.executable,
        trainer_path,
        args.profile,
        results_dir,
        model_dir,
    )
    audit_command = build_audit_command(
        sys.executable,
        auditor_path,
        run_id,
        args.base_predictions,
        results_dir,
        audit_dir,
    )

    if args.reuse_training_output:
        required = [
            results_dir / "calibration_predictions.csv",
            results_dir / "metrics.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing reused training outputs: {missing}")
    else:
        if results_dir.exists() or model_dir.exists():
            raise FileExistsError(
                "training output already exists; choose another --run-id or use "
                "--reuse-training-output"
            )
        subprocess.run(training_command, check=True)

    subprocess.run(audit_command, check=True)
    audit_report = json.loads(
        (audit_dir / "audit_report.json").read_text(encoding="utf-8")
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "profile": args.profile,
        "description": PROFILES[args.profile]["description"],
        "sources": PROFILES[args.profile]["sources"],
        "training_output_reused": bool(args.reuse_training_output),
        "training_command": training_command,
        "audit_command": audit_command,
        "results_dir": str(results_dir),
        "audit_dir": str(audit_dir),
        "status": audit_report["status"],
        "all_local_promotion_checks_passed": audit_report[
            "all_local_promotion_checks_passed"
        ],
    }
    manifest_path = audit_dir / "integration_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
