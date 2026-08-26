from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from audit_submission_package import (
    DEFAULT_SAMPLE_SUBMISSION,
    audit_metadata,
    compare_source,
    flatten_checks,
    sha256_file,
    validate_submission_csv,
    validate_zip,
)


DEFAULT_SOURCE_DIR = Path("results/blend_final_120k_global_shrink_recency_weighted_8blocks")
DEFAULT_FINAL_DIR = Path("results/final_recommended_submission")
DEFAULT_CANDIDATE_RANKING = Path("results/final_submission_candidate_ranking.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把某个实验目录固化成最终推荐提交目录，并生成校验、hash 和复现说明。")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="来源实验目录，必须包含 submission.csv/zip 和 metrics.json")
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR, help="最终提交目录")
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION)
    parser.add_argument("--candidate-ranking", type=Path, default=DEFAULT_CANDIDATE_RANKING)
    parser.add_argument("--environment", type=str, default="quant-competition-wsl")
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def copy_required_files(source_dir: Path, final_dir: Path, candidate_ranking: Path) -> list[str]:
    """复制最终提交必须依赖的文件；重复执行时直接覆盖同名文件，保持幂等。"""
    final_dir.mkdir(parents=True, exist_ok=True)
    required_pairs = {
        source_dir / "submission.csv": final_dir / "submission.csv",
        source_dir / "submission.zip": final_dir / "submission.zip",
        source_dir / "metrics.json": final_dir / "source_metrics.json",
    }
    copied = []
    for src, dst in required_pairs.items():
        require_file(src)
        shutil.copy2(src, dst)
        copied.append(str(dst))

    if candidate_ranking.exists():
        shutil.copy2(candidate_ranking, final_dir / "candidate_ranking.csv")
        copied.append(str(final_dir / "candidate_ranking.csv"))

    # 诊断文件不是所有实验都有；存在就复制，方便之后复盘为什么选这个提交。
    optional_names = [
        "block_scores.csv",
        "rolling_scores.csv",
        "shrink_curve.csv",
        "shrink_curve.png",
        "best_shrink_by_block.png",
        "rolling_validation.png",
        "calibration_score_by_asset.png",
        "strategy_selection.csv",
        "strategy_selection.png",
        "submission_distribution.png",
    ]
    for name in optional_names:
        src = source_dir / name
        if src.exists():
            dst = final_dir / name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def load_metrics(final_dir: Path) -> dict:
    metrics_path = final_dir / "source_metrics.json"
    require_file(metrics_path)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def build_reproduce_command(metrics: dict, source_dir: Path, args: argparse.Namespace) -> str:
    """根据 metrics 自动写一个复现命令；当前主路径是全局 shrink 稳定性脚本。"""
    if metrics.get("reproduce_command"):
        command = str(metrics["reproduce_command"])
        # Older result metadata contains a Windows command. Normalize it when
        # packaging a result from the WSL environment.
        command = command.replace(r"D:\conda-envs\quant-competition-sim\python.exe", args.python)
        command = command.replace(".\\", "").replace("\\", "/")
        return f"# Run from the project root in WSL\n{command}\n"
    if metrics.get("strategy") == "single_model_global_shrink":
        return f"""# 在项目根目录 E:\\量化大赛 下运行
{args.python} -u .\\code\\experiments\\asset_all_tcn\\calibrate_global_shrink_stability.py `
  --calibration .\\{metrics.get("calibration_file")} `
  --test .\\{metrics.get("test_file")} `
  --results-dir .\\{source_dir} `
  --min-shrink 0.80 `
  --max-shrink 1.80 `
  --shrink-step 0.01 `
  --block-count {metrics.get("block_count", 8)} `
  --selection-rule {metrics.get("selection_rule", "holdout_recency_weighted_best")}
"""
    return f"""# 来源实验不是 single_model_global_shrink，请查看 source_metrics.json 和原始训练脚本。
# source_dir: {source_dir}
"""


def build_validation_report(final_dir: Path, source_dir: Path, sample_submission: Path, environment: str) -> dict:
    """生成轻量 validation_report；完整逐项审计由 audit_report.json 记录。"""
    metrics = load_metrics(final_dir)
    _, csv_audit = validate_submission_csv(final_dir / "submission.csv", sample_submission)
    zip_audit = validate_zip(final_dir / "submission.zip", final_dir / "submission.csv")

    selected_shrink = metrics.get("selected_shrink")
    selected_residual_beta = metrics.get("selected_residual_beta")
    full_score = metrics.get("selected_full_calibration_score", metrics.get("calibration_score"))
    raw_score = metrics.get("raw_full_calibration_score")
    calibration = metrics.get("calibration", {})
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "environment": environment,
        "recommended_source_dir": str(source_dir),
        "recommended_submission": str(final_dir / "submission.zip"),
        "strategy": metrics.get("strategy", metrics.get("selected_strategy")),
        "selected_shrink": selected_shrink,
        "selected_residual_beta": selected_residual_beta,
        "selection_rule": metrics.get("selection_rule"),
        "full_calibration_score": full_score,
        "base_full_calibration_score": raw_score,
        "holdout_half_calibration_score": calibration.get("selected_holdout_half_score"),
        "base_holdout_half_calibration_score": calibration.get("base_holdout_half_score"),
        "raw_120k_full_calibration_score": raw_score,
        "full_calibration_best": metrics.get("full_calibration_best"),
        "submission_validation": {
            "shape": [csv_audit["stats"]["rows"], len(csv_audit["stats"]["columns"])],
            "columns": csv_audit["stats"]["columns"],
            "row_order_matches_sample": csv_audit["checks"]["row_id_order_matches_sample"]["passed"],
            "null_target_count": int("0" if csv_audit["checks"]["target_has_no_null"]["passed"] else -1),
            "finite_target_count": csv_audit["stats"]["rows"] if csv_audit["checks"]["target_all_finite"]["passed"] else None,
            "target_mean": csv_audit["stats"]["target_mean"],
            "target_std": csv_audit["stats"]["target_std"],
            "target_min": csv_audit["stats"]["target_min"],
            "target_max": csv_audit["stats"]["target_max"],
            "zip_is_valid": zip_audit["checks"]["zip_is_valid"]["passed"],
            "zip_members": zip_audit["members"],
            "zip_testzip_result": zip_audit["zip_test_result"],
        },
        "hashes": {
            "submission_csv_sha256": sha256_file(final_dir / "submission.csv"),
            "submission_zip_sha256": sha256_file(final_dir / "submission.zip"),
            "source_metrics_sha256": sha256_file(final_dir / "source_metrics.json"),
        },
        "key_files": {
            "submission_csv": str(final_dir / "submission.csv"),
            "submission_zip": str(final_dir / "submission.zip"),
            "validation_report": str(final_dir / "validation_report.json"),
            "audit_report": str(final_dir / "audit_report.json"),
            "candidate_ranking": str(final_dir / "candidate_ranking.csv"),
            "source_metrics": str(final_dir / "source_metrics.json"),
        },
    }
    (final_dir / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_embedded_audit(final_dir: Path, source_dir: Path, sample_submission: Path) -> dict:
    """复用 audit_submission_package 的检查函数，直接生成 audit_report.json。"""
    _, csv_audit = validate_submission_csv(final_dir / "submission.csv", sample_submission)
    zip_audit = validate_zip(final_dir / "submission.zip", final_dir / "submission.csv")
    source_audit = compare_source(final_dir, source_dir)
    metadata_audit = audit_metadata(final_dir, source_dir)
    checks = flatten_checks(csv_audit, zip_audit, source_audit, metadata_audit)
    audit_report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "final_dir": str(final_dir),
        "source_dir": str(source_dir),
        "sample_submission": str(sample_submission),
        "overall_passed": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "submission_csv_sha256": sha256_file(final_dir / "submission.csv"),
        "submission_zip_sha256": sha256_file(final_dir / "submission.zip"),
        "csv_audit": csv_audit,
        "zip_audit": zip_audit,
        "source_audit": source_audit,
        "metadata_audit": metadata_audit,
    }
    (final_dir / "audit_report.json").write_text(json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not audit_report["overall_passed"]:
        raise RuntimeError(f"final package audit failed: {audit_report['failed_checks']}")
    return audit_report


def write_readme(final_dir: Path, source_dir: Path, validation_report: dict, audit_report: dict) -> None:
    readme = f"""# Final Recommended Submission

Recommended file to submit:

```text
{(final_dir / "submission.zip").as_posix()}
```

Source experiment:

```text
{source_dir.as_posix()}
```

Strategy: `{validation_report.get("strategy")}`

- selected shrink: `{validation_report.get("selected_shrink")}`
- selected residual beta: `{validation_report.get("selected_residual_beta")}`
- selection rule: `{validation_report.get("selection_rule")}`
- full calibration score: `{validation_report.get("full_calibration_score")}`
- base full calibration score: `{validation_report.get("base_full_calibration_score")}`
- holdout-half calibration score: `{validation_report.get("holdout_half_calibration_score")}`
- base holdout-half calibration score: `{validation_report.get("base_holdout_half_calibration_score")}`
- full calibration best shrink/score: `{validation_report.get("full_calibration_best")}`

Submission validation:

- shape: `{validation_report["submission_validation"]["shape"]}`
- columns: `{validation_report["submission_validation"]["columns"]}`
- row order matches sample: `{validation_report["submission_validation"]["row_order_matches_sample"]}`
- null targets: `{validation_report["submission_validation"]["null_target_count"]}`
- finite targets: `{validation_report["submission_validation"]["finite_target_count"]}`
- zip valid: `{validation_report["submission_validation"]["zip_is_valid"]}`

Hashes:

- submission.csv sha256: `{validation_report["hashes"]["submission_csv_sha256"]}`
- submission.zip sha256: `{validation_report["hashes"]["submission_zip_sha256"]}`

Audit:

- overall passed: `{audit_report["overall_passed"]}`
- failed checks: `{audit_report["failed_checks"]}`

Useful files:

- `validation_report.json`: machine-readable validation and hashes
- `audit_report.json`: full package audit; checks csv, zip, source consistency, candidate ranking, and leakage-safe metadata
- `candidate_ranking.csv`: candidate comparison table
- `source_metrics.json`: metrics copied from the source experiment
- `reproduce_command.ps1`: command to regenerate the source experiment
- diagnostic plots such as `shrink_curve.png`, `best_shrink_by_block.png`, `rolling_validation.png`
"""
    (final_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    copied_files = copy_required_files(args.source_dir, args.final_dir, args.candidate_ranking)
    metrics = load_metrics(args.final_dir)
    reproduce_command = build_reproduce_command(metrics, args.source_dir, args)
    (args.final_dir / "reproduce_command.ps1").write_text(reproduce_command, encoding="utf-8")
    validation_report = build_validation_report(args.final_dir, args.source_dir, args.sample_submission, args.environment)
    audit_report = run_embedded_audit(args.final_dir, args.source_dir, args.sample_submission)
    write_readme(args.final_dir, args.source_dir, validation_report, audit_report)
    print(
        json.dumps(
            {
                "final_dir": str(args.final_dir),
                "source_dir": str(args.source_dir),
                "copied_file_count": len(copied_files),
                "submission_zip": str(args.final_dir / "submission.zip"),
                "selected_shrink": validation_report.get("selected_shrink"),
                "selected_residual_beta": validation_report.get("selected_residual_beta"),
                "audit_overall_passed": audit_report["overall_passed"],
                "failed_checks": audit_report["failed_checks"],
                "submission_zip_sha256": validation_report["hashes"]["submission_zip_sha256"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
