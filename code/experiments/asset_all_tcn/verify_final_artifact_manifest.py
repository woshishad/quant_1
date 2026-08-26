from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


DEFAULT_MANIFEST = Path("results/final_recommended_submission/MANIFEST.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验最终提交 manifest 中记录的文件 hash 是否仍然一致。")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_record(record: dict) -> dict:
    path = Path(record["path"])
    expected_exists = bool(record["exists"])
    actual_exists = path.exists()
    result = {
        "path": record["path"],
        "role": record.get("role"),
        "expected_exists": expected_exists,
        "actual_exists": actual_exists,
        "passed": False,
        "reason": "",
        "expected_sha256": record.get("sha256"),
        "actual_sha256": None,
    }
    if expected_exists != actual_exists:
        result["reason"] = "existence mismatch"
        return result
    if not expected_exists:
        result["passed"] = True
        result["reason"] = "missing as expected"
        return result
    actual_sha256 = sha256_file(path)
    result["actual_sha256"] = actual_sha256
    if actual_sha256 != record.get("sha256"):
        result["reason"] = "sha256 mismatch"
        return result
    result["passed"] = True
    result["reason"] = "ok"
    return result


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = [verify_record(record) for record in manifest.get("files", [])]
    failed = [check for check in checks if not check["passed"]]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "overall_passed": len(failed) == 0,
        "checked_file_count": len(checks),
        "failed_count": len(failed),
        "failed": failed,
        "manifest_submission_zip_sha256": manifest.get("audit_summary", {}).get("submission_zip_sha256"),
    }
    output = args.output or (args.manifest.parent / "MANIFEST_VERIFY.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
