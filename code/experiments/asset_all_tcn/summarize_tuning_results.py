from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize leakage-safe walk-forward tuning results.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/asset_all_walk_forward_tuning_summary.csv"),
    )
    return parser.parse_args()


def parse_time_range(text: str | None) -> tuple[int | None, int | None]:
    if not text:
        return None, None
    numbers = [int(value) for value in re.findall(r"\d+", text)]
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    if len(numbers) == 1:
        return None, numbers[0]
    return None, None


def infer_fold(test_start: int | None, test_end: int | None) -> str:
    # 这里按我们当前的 walk-forward 协议命名：fold0/1/2 分别对应三个未来 holdout 段。
    if (test_start, test_end) == (40_000, 59_999):
        return "fold0"
    if (test_start, test_end) == (60_000, 79_999):
        return "fold1"
    if (test_start, test_end) == (80_000, 99_999):
        return "fold2"
    return "unknown"


def protocol_name(metrics: dict) -> str:
    if metrics.get("blend_learned_on") == "calibration_predictions_only":
        return "calibration_blend"
    if metrics.get("catboost_config"):
        return "ridge_catboost_gpu"
    if metrics.get("model_scope") == "per_asset":
        return "per_asset_models"
    if "best_candidate" in metrics:
        return "global_tabular"
    return "other"


def extract_best_candidate(metrics: dict) -> dict:
    candidate = metrics.get("best_candidate") or {}
    return {
        "top_k": candidate.get("top_k"),
        "lgbm_num_leaves": candidate.get("lgbm_num_leaves"),
        "catboost_depth": candidate.get("catboost_depth"),
        "cal_score": candidate.get("cal_score"),
        "cal_ridge_weight": candidate.get("cal_ridge_weight"),
        "cal_lgbm_weight": candidate.get("cal_lgbm_weight"),
        "cal_catboost_weight": candidate.get("cal_catboost_weight"),
        "cal_shrink": candidate.get("cal_shrink"),
        "cal_shrink_mean": candidate.get("cal_shrink_mean"),
        "cal_shrink_max": candidate.get("cal_shrink_max"),
    }


def row_from_metrics(metrics_path: Path, results_root: Path) -> dict | None:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"skip unreadable metrics: {metrics_path} ({exc})")
        return None

    guard = metrics.get("future_function_guard", {})
    _, fit_train_end = parse_time_range(guard.get("feature_screen_fit_train"))
    cal_start, cal_end = parse_time_range(
        guard.get("selection_and_shrink_calibration") or guard.get("blend_weight_calibration")
    )
    test_start, test_end = parse_time_range(guard.get("holdout_test_only_for_final_score"))
    blend_info = metrics.get("blend_info") or {}
    weight_bounds = metrics.get("weight_bounds") or {}
    policy = metrics.get("candidate_selection_policy") or {}
    catboost_config = metrics.get("catboost_config") or {}

    row = {
        "experiment": metrics_path.parent.name,
        "metrics_path": str(metrics_path.relative_to(results_root.parent)),
        "protocol": protocol_name(metrics),
        "fold": infer_fold(test_start, test_end),
        "fit_train_end": fit_train_end,
        "cal_start": cal_start,
        "cal_end": cal_end,
        "test_start": test_start,
        "test_end": test_end,
        "test_score": metrics.get("test_score"),
        "negative_asset_count": metrics.get("negative_asset_count"),
        "left_test_score": metrics.get("left_test_score"),
        "right_test_score": metrics.get("right_test_score"),
        "blend_left_weight": blend_info.get("left_weight"),
        "blend_right_weight": blend_info.get("right_weight"),
        "min_left_weight": weight_bounds.get("min_left_weight"),
        "max_left_weight": weight_bounds.get("max_left_weight"),
        "feature_source": metrics.get("feature_source"),
        "fixed_features_file": metrics.get("fixed_features_file"),
        "selected_feature_count": metrics.get("selected_feature_count"),
        "lgbm_seeds": ",".join(str(seed) for seed in policy.get("lgbm_seeds", [])),
        "lgbm_subsample": policy.get("lgbm_subsample"),
        "lgbm_colsample_bytree": policy.get("lgbm_colsample_bytree"),
        "catboost_task_type": catboost_config.get("task_type"),
        "catboost_iterations": catboost_config.get("iterations"),
        "catboost_depth_candidates": None,
        "leakage_safe": metrics.get("leakage_safe"),
    }
    row.update(extract_best_candidate(metrics))
    return row


def should_include(metrics_path: Path) -> bool:
    name = metrics_path.parent.name
    prefixes = (
        "asset_all_walk_forward_tabular_100k",
        "asset_all_walk_forward_asset_models_100k",
        "asset_all_walk_forward_catboost_gpu",
        "blend_global",
    )
    return name.startswith(prefixes)


def main() -> None:
    args = parse_args()
    rows = []
    for metrics_path in sorted(args.results_root.glob("*/metrics.json")):
        if not should_include(metrics_path):
            continue
        row = row_from_metrics(metrics_path, args.results_root)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError(f"no metrics.json found under {args.results_root}")

    frame = pd.DataFrame(rows)
    # 同一 fold 内按分数降序排，便于直接看每段时间里当前最强配置。
    frame = frame.sort_values(["fold", "test_score"], ascending=[True, False], na_position="last")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {len(frame)} rows to {args.output}")
    print(frame[["fold", "experiment", "protocol", "test_score", "negative_asset_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
