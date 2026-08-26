from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_FINAL_DIR = Path("results/final_recommended_submission")
DEFAULT_EXTRA_FILES = [
    Path("code/experiments/asset_all_tcn/audit_submission_package.py"),
    Path("code/experiments/asset_all_tcn/package_final_submission.py"),
    Path("code/experiments/asset_all_tcn/calibrate_global_shrink_stability.py"),
    Path("code/experiments/asset_all_tcn/audit_final_submission.ps1"),
    Path("code/experiments/asset_all_tcn/package_final_submission.ps1"),
    Path("docs/final_submission_model_card.md"),
    Path("docs/final_submission_quickstart.md"),
    Path("docs/asset_all_final_pipeline.md"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成最终提交包及关键脚本/文档的文件级 manifest。")
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--include-extra-files", action="store_true", default=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, role: str) -> dict:
    """把一个文件记录成稳定的 manifest 行，便于之后检查是否被改过。"""
    stat = path.stat()
    return {
        "role": role,
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256_file(path),
    }


def missing_record(path: Path, role: str) -> dict:
    return {
        "role": role,
        "path": str(path),
        "exists": False,
        "size_bytes": None,
        "modified_time": None,
        "sha256": None,
    }


def collect_final_dir_files(final_dir: Path, excluded_paths: set[Path]) -> list[dict]:
    records = []
    for path in sorted(final_dir.iterdir(), key=lambda value: value.name.lower()):
        # manifest 不能记录自身，否则每次写入都会让自己的 hash 立刻过期。
        if path.is_file() and path.resolve() not in excluded_paths:
            records.append(file_record(path, "final_submission_package"))
    return records


def collect_extra_files(include_extra_files: bool) -> list[dict]:
    if not include_extra_files:
        return []
    records = []
    for path in DEFAULT_EXTRA_FILES:
        if path.exists():
            records.append(file_record(path, "pipeline_code_or_doc"))
        else:
            records.append(missing_record(path, "pipeline_code_or_doc"))
    return records


def get_git_status() -> str:
    """记录当前 git 状态摘要；仓库状态很乱时也不影响 manifest 生成。"""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout.strip()
    except Exception as exc:  # pragma: no cover - 只做附加诊断
        return f"git status unavailable: {exc}"


def get_cuda_info(python_exe: str) -> dict:
    """用项目指定环境查询 CUDA，确认记录的是用户实际训练环境。"""
    code = (
        "import json, torch; "
        "print(json.dumps({"
        "'torch_version': torch.__version__, "
        "'cuda_available': torch.cuda.is_available(), "
        "'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'"
        "}, ensure_ascii=False))"
    )
    try:
        result = subprocess.run(
            [python_exe, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        return {"error": result.stderr.strip(), "returncode": result.returncode}
    except Exception as exc:  # pragma: no cover - 只做附加诊断
        return {"error": str(exc)}


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    final_dir = args.final_dir
    output_json = args.output_json or (final_dir / "MANIFEST.json")
    output_csv = args.output_csv or (final_dir / "MANIFEST.csv")
    if not final_dir.exists():
        raise FileNotFoundError(final_dir)

    excluded_paths = {
        output_json.resolve(),
        output_csv.resolve(),
        (final_dir / "MANIFEST_VERIFY.json").resolve(),
    }
    records = collect_final_dir_files(final_dir, excluded_paths) + collect_extra_files(args.include_extra_files)
    audit_report = load_json_if_exists(final_dir / "audit_report.json")
    validation_report = load_json_if_exists(final_dir / "validation_report.json")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "final_dir": str(final_dir),
        "python_exe": args.python_exe,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "cuda": get_cuda_info(args.python_exe),
        "audit_summary": {
            "overall_passed": audit_report.get("overall_passed"),
            "failed_checks": audit_report.get("failed_checks"),
            "submission_zip_sha256": audit_report.get("submission_zip_sha256"),
        },
        "validation_summary": {
            "recommended_submission": validation_report.get("recommended_submission"),
            "selected_shrink": validation_report.get("selected_shrink"),
            "full_calibration_score": validation_report.get("full_calibration_score"),
        },
        "git_status_short": get_git_status(),
        "file_count": len(records),
        "missing_file_count": int(sum(not record["exists"] for record in records)),
        "files": records,
    }

    output_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(records).to_csv(output_csv, index=False, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_csv": str(output_csv),
                "file_count": manifest["file_count"],
                "missing_file_count": manifest["missing_file_count"],
                "submission_zip_sha256": manifest["audit_summary"]["submission_zip_sha256"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
