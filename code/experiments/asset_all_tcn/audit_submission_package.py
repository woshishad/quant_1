from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SAMPLE_SUBMISSION = Path(
    "data/raw/public_release_20260630/public_release_20260630/data/sample_submission.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计最终提交目录，确认 submission 和来源实验一致且可提交。")
    parser.add_argument("--final-dir", type=Path, default=Path("results/final_recommended_submission"))
    parser.add_argument("--source-dir", type=Path, default=Path("results/blend_final_120k_global_shrink_recency_weighted_8blocks"))
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION)
    parser.add_argument("--output", type=Path, default=None, help="默认写到 final-dir/audit_report.json")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bool_check(value: bool, detail: str) -> dict:
    return {"passed": bool(value), "detail": detail}


def validate_submission_csv(submission_path: Path, sample_path: Path) -> tuple[pd.DataFrame, dict]:
    """校验官方提交 CSV 的形状、列名、row_id 顺序和 target 数值质量。"""
    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path, usecols=["row_id"])
    target = submission["target"] if "target" in submission.columns else pd.Series(dtype=float)
    finite_mask = np.isfinite(target.to_numpy(dtype=np.float64, copy=False)) if len(target) else np.array([])

    checks = {
        "csv_exists": bool_check(submission_path.exists(), str(submission_path)),
        "shape_matches_sample": bool_check(
            submission.shape == (len(sample), 2),
            f"submission shape={submission.shape}, sample rows={len(sample)}",
        ),
        "columns_are_row_id_target": bool_check(
            submission.columns.tolist() == ["row_id", "target"],
            f"columns={submission.columns.tolist()}",
        ),
        "row_id_order_matches_sample": bool_check(
            "row_id" in submission.columns and submission["row_id"].equals(sample["row_id"]),
            "row_id must exactly follow sample_submission order",
        ),
        "row_id_unique": bool_check(
            "row_id" in submission.columns and int(submission["row_id"].duplicated().sum()) == 0,
            f"duplicate row_id count={int(submission['row_id'].duplicated().sum()) if 'row_id' in submission.columns else 'missing'}",
        ),
        "target_has_no_null": bool_check(
            "target" in submission.columns and int(target.isna().sum()) == 0,
            f"null target count={int(target.isna().sum()) if 'target' in submission.columns else 'missing target column'}",
        ),
        "target_all_finite": bool_check(
            "target" in submission.columns and int(finite_mask.sum()) == len(submission),
            f"finite target count={int(finite_mask.sum()) if len(finite_mask) else 0}",
        ),
        "target_not_constant": bool_check(
            "target" in submission.columns and float(target.std()) > 0.0,
            f"target std={float(target.std()) if 'target' in submission.columns else 'missing'}",
        ),
    }
    stats = {
        "rows": int(len(submission)),
        "columns": submission.columns.tolist(),
        "target_mean": float(target.mean()) if "target" in submission.columns else None,
        "target_std": float(target.std()) if "target" in submission.columns else None,
        "target_min": float(target.min()) if "target" in submission.columns else None,
        "target_max": float(target.max()) if "target" in submission.columns else None,
    }
    return submission, {"checks": checks, "stats": stats}


def validate_zip(zip_path: Path, submission_csv_path: Path) -> dict:
    """校验 zip 是否可读，并确认 zip 内的 submission.csv 与外部 CSV 完全一致。"""
    checks = {"zip_exists": bool_check(zip_path.exists(), str(zip_path))}
    members: list[str] = []
    extracted_csv_sha256 = None
    zip_test_result = None

    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.namelist()
            zip_test_result = archive.testzip()
            if "submission.csv" in members:
                extracted = archive.read("submission.csv")
                extracted_csv_sha256 = sha256_bytes(extracted)
    checks.update(
        {
            "zip_is_valid": bool_check(zip_path.exists() and zipfile.is_zipfile(zip_path), "zipfile.is_zipfile"),
            "zip_test_passed": bool_check(zip_test_result is None, f"testzip result={zip_test_result}"),
            "zip_contains_submission_csv": bool_check("submission.csv" in members, f"members={members}"),
            "zip_csv_matches_external_csv": bool_check(
                extracted_csv_sha256 == sha256_file(submission_csv_path) if extracted_csv_sha256 else False,
                "sha256(zip/submission.csv) == sha256(final/submission.csv)",
            ),
        }
    )
    return {
        "checks": checks,
        "members": members,
        "zip_test_result": zip_test_result,
        "zip_sha256": sha256_file(zip_path) if zip_path.exists() else None,
        "extracted_submission_csv_sha256": extracted_csv_sha256,
    }


def compare_source(final_dir: Path, source_dir: Path) -> dict:
    """确认 final_recommended_submission 是从指定来源实验目录固化出来的。"""
    final_csv = final_dir / "submission.csv"
    final_zip = final_dir / "submission.zip"
    source_csv = source_dir / "submission.csv"
    source_zip = source_dir / "submission.zip"
    source_metrics = source_dir / "metrics.json"
    copied_metrics = final_dir / "source_metrics.json"

    checks = {
        "source_dir_exists": bool_check(source_dir.exists(), str(source_dir)),
        "source_csv_exists": bool_check(source_csv.exists(), str(source_csv)),
        "source_zip_exists": bool_check(source_zip.exists(), str(source_zip)),
        "final_csv_matches_source_csv": bool_check(
            final_csv.exists() and source_csv.exists() and sha256_file(final_csv) == sha256_file(source_csv),
            "final submission.csv should be byte-identical to source submission.csv",
        ),
        "final_zip_matches_source_zip": bool_check(
            final_zip.exists() and source_zip.exists() and sha256_file(final_zip) == sha256_file(source_zip),
            "final submission.zip should be byte-identical to source submission.zip",
        ),
        "copied_metrics_matches_source_metrics": bool_check(
            copied_metrics.exists() and source_metrics.exists() and sha256_file(copied_metrics) == sha256_file(source_metrics),
            "source_metrics.json should be copied from source metrics.json",
        ),
    }
    return {
        "checks": checks,
        "source_hashes": {
            "source_csv_sha256": sha256_file(source_csv) if source_csv.exists() else None,
            "source_zip_sha256": sha256_file(source_zip) if source_zip.exists() else None,
            "source_metrics_sha256": sha256_file(source_metrics) if source_metrics.exists() else None,
        },
    }


def audit_metadata(final_dir: Path, source_dir: Path) -> dict:
    """检查报告、候选排名、复现命令和 metrics 中的关键防泄露字段。"""
    validation_report_path = final_dir / "validation_report.json"
    source_metrics_path = final_dir / "source_metrics.json"
    ranking_path = final_dir / "candidate_ranking.csv"
    reproduce_path = final_dir / "reproduce_command.ps1"

    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8")) if source_metrics_path.exists() else {}
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8")) if validation_report_path.exists() else {}
    ranking = pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame()
    source_name = source_dir.name
    ranking_has_source = bool((ranking.get("name", pd.Series(dtype=str)) == source_name).any()) if not ranking.empty else False

    checks = {
        "validation_report_exists": bool_check(validation_report_path.exists(), str(validation_report_path)),
        "source_metrics_exists": bool_check(source_metrics_path.exists(), str(source_metrics_path)),
        "candidate_ranking_exists": bool_check(ranking_path.exists(), str(ranking_path)),
        "reproduce_command_exists": bool_check(reproduce_path.exists(), str(reproduce_path)),
        "ranking_contains_source_candidate": bool_check(ranking_has_source, f"source candidate={source_name}"),
        "source_metrics_leakage_safe": bool_check(
            bool(source_metrics.get("leakage_safe")) is True,
            f"leakage_safe={source_metrics.get('leakage_safe')}",
        ),
        "source_metrics_has_strategy_or_guard": bool_check(
            any(
                source_metrics.get(key) is not None
                for key in ["strategy", "selected_strategy", "official_test_used_for", "future_function_guard"]
            ),
            (
                f"strategy={source_metrics.get('strategy')}, "
                f"selected_strategy={source_metrics.get('selected_strategy')}, "
                f"official_test_used_for={source_metrics.get('official_test_used_for')}"
            ),
        ),
        "selected_shrink_matches_report": bool_check(
            source_metrics.get("selected_shrink") == validation_report.get("selected_shrink"),
            f"metrics={source_metrics.get('selected_shrink')}, report={validation_report.get('selected_shrink')}",
        ),
    }
    return {
        "checks": checks,
        "selected_shrink": source_metrics.get("selected_shrink"),
        "selection_rule": source_metrics.get("selection_rule"),
        "full_calibration_score": source_metrics.get("selected_full_calibration_score"),
    }


def flatten_checks(*sections: dict) -> dict[str, bool]:
    flattened = {}
    for section in sections:
        for check_name, check in section.get("checks", {}).items():
            flattened[check_name] = bool(check.get("passed"))
    return flattened


def main() -> None:
    args = parse_args()
    final_dir = args.final_dir
    output_path = args.output or (final_dir / "audit_report.json")

    submission_csv = final_dir / "submission.csv"
    submission_zip = final_dir / "submission.zip"
    submission, csv_audit = validate_submission_csv(submission_csv, args.sample_submission)
    zip_audit = validate_zip(submission_zip, submission_csv)
    source_audit = compare_source(final_dir, args.source_dir)
    metadata_audit = audit_metadata(final_dir, args.source_dir)
    all_checks = flatten_checks(csv_audit, zip_audit, source_audit, metadata_audit)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "final_dir": str(final_dir),
        "source_dir": str(args.source_dir),
        "sample_submission": str(args.sample_submission),
        "overall_passed": all(all_checks.values()),
        "failed_checks": [name for name, passed in all_checks.items() if not passed],
        "submission_csv_sha256": sha256_file(submission_csv) if submission_csv.exists() else None,
        "submission_zip_sha256": sha256_file(submission_zip) if submission_zip.exists() else None,
        "csv_audit": csv_audit,
        "zip_audit": zip_audit,
        "source_audit": source_audit,
        "metadata_audit": metadata_audit,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
