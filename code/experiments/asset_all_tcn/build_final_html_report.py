from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_FINAL_DIR = Path("results/final_recommended_submission")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成最终提交包的静态 HTML 总览报告。")
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def table_from_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    rows = []
    for _, row in frame[columns].iterrows():
        cells = "".join(f"<td>{fmt(row[column])}</td>" for column in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def status_badge(passed: bool) -> str:
    klass = "ok" if passed else "bad"
    text = "PASS" if passed else "FAIL"
    return f'<span class="badge {klass}">{text}</span>'


def main() -> None:
    args = parse_args()
    final_dir = args.final_dir
    output = args.output or (final_dir / "report.html")
    if not final_dir.exists():
        raise FileNotFoundError(final_dir)

    audit = load_json(final_dir / "audit_report.json")
    validation = load_json(final_dir / "validation_report.json")
    manifest_verify = load_json(final_dir / "MANIFEST_VERIFY.json")
    manifest = load_json(final_dir / "MANIFEST.json")
    metrics = load_json(final_dir / "source_metrics.json")
    ranking = pd.read_csv(final_dir / "candidate_ranking.csv")
    block_scores = pd.read_csv(final_dir / "block_scores.csv")
    rolling_scores = pd.read_csv(final_dir / "rolling_scores.csv")

    target_stats = validation.get("submission_validation", {})
    hashes = validation.get("hashes", {})
    selected_shrink = validation.get("selected_shrink", metrics.get("selected_shrink"))
    full_score = validation.get("full_calibration_score", metrics.get("selected_full_calibration_score"))
    raw_score = validation.get("raw_120k_full_calibration_score", metrics.get("raw_full_calibration_score"))

    ranking_html = table_from_frame(
        ranking,
        ["label", "selected_shrink", "selection_rule", "full_calibration", "holdout_half", "submission_zip"],
    )
    block_html = table_from_frame(
        block_scores,
        ["block", "time_min", "time_max", "rows", "raw_score", "best_shrink", "best_score"],
    )
    rolling_html = table_from_frame(
        rolling_scores,
        ["fit_blocks", "holdout_block", "fit_best_shrink", "fit_score", "holdout_score"],
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Final Submission Report</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; color: #1f2937; background: #f7f7f4; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1, h2 {{ color: #111827; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }}
    .card {{ background: #fff; border: 1px solid #ddd8cc; border-radius: 6px; padding: 14px; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .small {{ color: #6b7280; font-size: 13px; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 3px 9px; font-weight: 700; font-size: 12px; }}
    .ok {{ background: #d1fae5; color: #065f46; }}
    .bad {{ background: #fee2e2; color: #991b1b; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; margin: 12px 0 22px; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #ece9df; }}
    img {{ max-width: 100%; border: 1px solid #ddd8cc; border-radius: 6px; background: #fff; }}
    code {{ background: #eeeae0; padding: 2px 4px; border-radius: 3px; }}
    .mono {{ font-family: Consolas, monospace; word-break: break-all; }}
  </style>
</head>
<body>
<main>
  <h1>Final Recommended Submission Report</h1>
  <p class="small">Generated at {html.escape(datetime.now().isoformat(timespec="seconds"))}</p>

  <section class="grid">
    <div class="card"><div>提交审计</div><div class="metric">{status_badge(bool(audit.get("overall_passed")))}</div><div class="small">failed_checks={html.escape(str(audit.get("failed_checks", [])))}</div></div>
    <div class="card"><div>Manifest 校验</div><div class="metric">{status_badge(bool(manifest_verify.get("overall_passed")))}</div><div class="small">checked={manifest_verify.get("checked_file_count")}, failed={manifest_verify.get("failed_count")}</div></div>
    <div class="card"><div>Selected shrink</div><div class="metric">{fmt(selected_shrink, 4)}</div><div class="small">{html.escape(str(validation.get("selection_rule", metrics.get("selection_rule"))))}</div></div>
    <div class="card"><div>Full calibration score</div><div class="metric">{fmt(full_score, 9)}</div><div class="small">raw 120k={fmt(raw_score, 9)}</div></div>
  </section>

  <h2>Final File</h2>
  <p>Upload: <code>results/final_recommended_submission/submission.zip</code></p>
  <p class="mono">submission.zip sha256: {html.escape(str(hashes.get("submission_zip_sha256", audit.get("submission_zip_sha256"))))}</p>
  <div class="grid">
    <div class="card"><div>Rows</div><div class="metric">{target_stats.get("shape", [""])[0] if target_stats.get("shape") else target_stats.get("rows", "")}</div></div>
    <div class="card"><div>Target mean</div><div class="metric">{fmt(target_stats.get("target_mean"), 6)}</div></div>
    <div class="card"><div>Target std</div><div class="metric">{fmt(target_stats.get("target_std"), 6)}</div></div>
    <div class="card"><div>Target range</div><div class="metric">{fmt(target_stats.get("target_min"), 3)} / {fmt(target_stats.get("target_max"), 3)}</div></div>
  </div>

  <h2>Candidate Ranking</h2>
  {ranking_html}

  <h2>Shrink Stability</h2>
  <div class="grid">
    <div><img src="shrink_curve.png" alt="shrink curve"></div>
    <div><img src="best_shrink_by_block.png" alt="best shrink by block"></div>
    <div><img src="rolling_validation.png" alt="rolling validation"></div>
  </div>

  <h2>Block Scores</h2>
  {block_html}

  <h2>Rolling Validation</h2>
  {rolling_html}

  <h2>Reproducibility</h2>
  <ul>
    <li>Manifest files: <code>MANIFEST.json</code>, <code>MANIFEST.csv</code></li>
    <li>Manifest file count: <code>{manifest.get("file_count")}</code>, missing: <code>{manifest.get("missing_file_count")}</code></li>
    <li>CUDA: <code>{html.escape(str(manifest.get("cuda", {})))}</code></li>
    <li>Model card: <code>MODEL_CARD.md</code></li>
  </ul>
</main>
</body>
</html>
"""
    output.write_text(html_text, encoding="utf-8")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
