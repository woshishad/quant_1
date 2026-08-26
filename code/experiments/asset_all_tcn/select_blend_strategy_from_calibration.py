from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from blend_predictions_from_calibration import (
    DEFAULT_SAMPLE_SUBMISSION,
    make_submission,
    merge_calibration,
    merge_test,
    numeric_grid,
    prediction_stats,
    score_by_asset,
    weighted_zero_mean_r2,
    write_zip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在 calibration 内部再做时间切分：前半段拟合融合参数，后半段选择融合策略，"
            "最后用选中的策略在完整 calibration 上重新拟合参数并生成官方 test 提交。"
        )
    )
    parser.add_argument("--left-calibration", type=Path, required=True)
    parser.add_argument("--right-calibration", type=Path, required=True)
    parser.add_argument("--left-test", type=Path, required=True)
    parser.add_argument("--right-test", type=Path, required=True)
    parser.add_argument("--left-name", type=str, default="lookback120k")
    parser.add_argument("--right-name", type=str, default="lookback240k")
    parser.add_argument("--prediction-column", type=str, default="prediction")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION)
    parser.add_argument("--min-left-weight", type=float, default=0.80)
    parser.add_argument("--max-left-weight", type=float, default=1.00)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--min-shrink", type=float, default=0.80)
    parser.add_argument("--max-shrink", type=float, default=1.60)
    parser.add_argument("--shrink-step", type=float, default=0.01)
    return parser.parse_args()


def score_frame(frame: pd.DataFrame, prediction_column: str) -> float:
    return weighted_zero_mean_r2(
        frame["target"].to_numpy(dtype=np.float64),
        frame[prediction_column].to_numpy(dtype=np.float64),
        frame["weight"].to_numpy(dtype=np.float64),
    )


def search_shrink(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    shrink_grid: np.ndarray,
) -> dict[str, float]:
    """只搜索 shrink，用于检查“幅度缩放”是否比复杂融合更稳。"""
    best = {"score": -np.inf, "shrink": float(shrink_grid[0])}
    for shrink in shrink_grid:
        score = weighted_zero_mean_r2(y_true, shrink * prediction, weight)
        if score > best["score"]:
            best = {"score": float(score), "shrink": float(shrink)}
    return best


def search_blend(
    y_true: np.ndarray,
    left_pred: np.ndarray,
    right_pred: np.ndarray,
    weight: np.ndarray,
    left_weight_grid: np.ndarray,
    shrink_grid: np.ndarray,
) -> dict[str, float]:
    """搜索 left/right 融合权重和 shrink；所有选择只发生在 calibration 内部。"""
    best = {
        "score": -np.inf,
        "left_weight": float(left_weight_grid[0]),
        "right_weight": float(1.0 - left_weight_grid[0]),
        "shrink": float(shrink_grid[0]),
    }
    for left_weight in left_weight_grid:
        base_prediction = left_weight * left_pred + (1.0 - left_weight) * right_pred
        for shrink in shrink_grid:
            score = weighted_zero_mean_r2(y_true, shrink * base_prediction, weight)
            if score > best["score"]:
                best = {
                    "score": float(score),
                    "left_weight": float(left_weight),
                    "right_weight": float(1.0 - left_weight),
                    "shrink": float(shrink),
                }
    return best


def fit_strategy(
    strategy: str,
    frame: pd.DataFrame,
    left_col: str,
    right_col: str,
    left_weight_grid: np.ndarray,
    shrink_grid: np.ndarray,
) -> dict:
    """根据策略名字在给定时间段上拟合参数。"""
    y_true = frame["target"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    left_pred = frame[left_col].to_numpy(dtype=np.float64)
    right_pred = frame[right_col].to_numpy(dtype=np.float64)

    if strategy == "left_only":
        return {"strategy": strategy, "params": {"left_weight": 1.0, "right_weight": 0.0, "shrink": 1.0}}
    if strategy == "right_only":
        return {"strategy": strategy, "params": {"left_weight": 0.0, "right_weight": 1.0, "shrink": 1.0}}
    if strategy == "left_global_shrink":
        best = search_shrink(y_true, left_pred, weight, shrink_grid)
        return {"strategy": strategy, "params": {"left_weight": 1.0, "right_weight": 0.0, "shrink": best["shrink"], "fit_score": best["score"]}}
    if strategy == "global_blend_shrink":
        best = search_blend(y_true, left_pred, right_pred, weight, left_weight_grid, shrink_grid)
        return {"strategy": strategy, "params": best}

    if strategy in {"per_asset_left_shrink", "per_asset_blend_shrink"}:
        rows = []
        for asset_id, asset_frame in frame.groupby("asset_id", sort=True):
            asset_y = asset_frame["target"].to_numpy(dtype=np.float64)
            asset_w = asset_frame["weight"].to_numpy(dtype=np.float64)
            asset_left = asset_frame[left_col].to_numpy(dtype=np.float64)
            asset_right = asset_frame[right_col].to_numpy(dtype=np.float64)
            if strategy == "per_asset_left_shrink":
                best = search_shrink(asset_y, asset_left, asset_w, shrink_grid)
                row = {
                    "asset_id": int(asset_id),
                    "left_weight": 1.0,
                    "right_weight": 0.0,
                    "shrink": float(best["shrink"]),
                    "fit_score": float(best["score"]),
                }
            else:
                best = search_blend(asset_y, asset_left, asset_right, asset_w, left_weight_grid, shrink_grid)
                row = {
                    "asset_id": int(asset_id),
                    "left_weight": float(best["left_weight"]),
                    "right_weight": float(best["right_weight"]),
                    "shrink": float(best["shrink"]),
                    "fit_score": float(best["score"]),
                }
            rows.append(row)
        return {"strategy": strategy, "params_by_asset": rows}

    raise ValueError(f"unknown strategy: {strategy}")


def apply_strategy(frame: pd.DataFrame, fitted: dict, left_col: str, right_col: str) -> pd.DataFrame:
    """把已拟合好的策略应用到 calibration 或官方 test。"""
    result = frame.copy()
    strategy = fitted["strategy"]

    if strategy in {"left_only", "right_only", "left_global_shrink", "global_blend_shrink"}:
        params = fitted["params"]
        result["blend_left_weight"] = float(params["left_weight"])
        result["blend_right_weight"] = float(params["right_weight"])
        result["blend_shrink"] = float(params["shrink"])
        result["prediction"] = result["blend_shrink"] * (
            result["blend_left_weight"] * result[left_col] + result["blend_right_weight"] * result[right_col]
        )
        return result

    if strategy in {"per_asset_left_shrink", "per_asset_blend_shrink"}:
        if "asset_id" not in result.columns:
            raise ValueError(f"{strategy} 需要 test 文件包含 asset_id，请传 final_test_predictions.csv")
        result["blend_left_weight"] = np.nan
        result["blend_right_weight"] = np.nan
        result["blend_shrink"] = np.nan
        result["prediction"] = np.nan
        for params in fitted["params_by_asset"]:
            mask = result["asset_id"] == params["asset_id"]
            result.loc[mask, "blend_left_weight"] = float(params["left_weight"])
            result.loc[mask, "blend_right_weight"] = float(params["right_weight"])
            result.loc[mask, "blend_shrink"] = float(params["shrink"])
            result.loc[mask, "prediction"] = float(params["shrink"]) * (
                float(params["left_weight"]) * result.loc[mask, left_col]
                + float(params["right_weight"]) * result.loc[mask, right_col]
            )
        if result["prediction"].isna().any():
            missing_assets = sorted(result.loc[result["prediction"].isna(), "asset_id"].dropna().unique().tolist())
            raise ValueError(f"{strategy} 有未覆盖 asset: {missing_assets[:10]}")
        return result

    raise ValueError(f"unknown strategy: {strategy}")


def split_calibration_by_time(cal_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用时间切 calibration，保证后半段永远晚于前半段。"""
    unique_times = np.sort(cal_frame["time_id"].unique())
    if len(unique_times) < 2:
        raise ValueError("calibration time_id 太少，无法做时间切分")
    split_time = int(unique_times[len(unique_times) // 2 - 1])
    fit_frame = cal_frame[cal_frame["time_id"] <= split_time].copy()
    holdout_frame = cal_frame[cal_frame["time_id"] > split_time].copy()
    if fit_frame.empty or holdout_frame.empty:
        raise ValueError("calibration 时间切分后出现空段")
    return fit_frame, holdout_frame


def strategy_summary(
    strategy: str,
    fit_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    left_col: str,
    right_col: str,
    left_weight_grid: np.ndarray,
    shrink_grid: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    fitted = fit_strategy(strategy, fit_frame, left_col, right_col, left_weight_grid, shrink_grid)
    fit_pred = apply_strategy(fit_frame, fitted, left_col, right_col)
    holdout_pred = apply_strategy(holdout_frame, fitted, left_col, right_col)
    row = {
        "strategy": strategy,
        "fit_half_score": score_frame(fit_pred, "prediction"),
        "holdout_half_score": score_frame(holdout_pred, "prediction"),
        "fit_time_min": int(fit_frame["time_id"].min()),
        "fit_time_max": int(fit_frame["time_id"].max()),
        "holdout_time_min": int(holdout_frame["time_id"].min()),
        "holdout_time_max": int(holdout_frame["time_id"].max()),
        "fitted_on_fit_half": json.dumps(fitted, ensure_ascii=False),
    }
    return row, holdout_pred


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    left_weight_grid = numeric_grid(args.min_left_weight, args.max_left_weight, args.step)
    shrink_grid = numeric_grid(args.min_shrink, args.max_shrink, args.shrink_step)
    cal_frame = merge_calibration(args)
    test_frame = merge_test(args)
    left_col = f"prediction_{args.left_name}"
    right_col = f"prediction_{args.right_name}"
    fit_frame, holdout_frame = split_calibration_by_time(cal_frame)

    # 策略从简单到复杂都放进来，最后只按后半段 holdout 分数选。
    strategies = [
        "left_only",
        "right_only",
        "left_global_shrink",
        "global_blend_shrink",
        "per_asset_left_shrink",
        "per_asset_blend_shrink",
    ]
    rows = []
    holdout_predictions = {}
    for strategy in strategies:
        row, holdout_pred = strategy_summary(
            strategy,
            fit_frame,
            holdout_frame,
            left_col,
            right_col,
            left_weight_grid,
            shrink_grid,
        )
        rows.append(row)
        holdout_predictions[strategy] = holdout_pred

    strategy_table = pd.DataFrame(rows).sort_values("holdout_half_score", ascending=False).reset_index(drop=True)
    selected_strategy = str(strategy_table.loc[0, "strategy"])

    # 策略选完后，用完整 calibration 重新拟合该策略的参数，再应用到官方 test。
    final_fitted = fit_strategy(selected_strategy, cal_frame, left_col, right_col, left_weight_grid, shrink_grid)
    final_cal = apply_strategy(cal_frame, final_fitted, left_col, right_col)
    final_test = apply_strategy(test_frame, final_fitted, left_col, right_col)
    final_cal["error"] = final_cal["prediction"] - final_cal["target"]

    submission = make_submission(final_test, args.sample_submission)
    submission_path = args.results_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    write_zip(submission_path, args.results_dir / "submission.zip")

    final_cal.to_csv(args.results_dir / "calibration_predictions.csv", index=False)
    final_test.to_csv(args.results_dir / "final_test_predictions.csv", index=False)
    strategy_table.to_csv(args.results_dir / "strategy_selection.csv", index=False, encoding="utf-8-sig")
    if "params_by_asset" in final_fitted:
        pd.DataFrame(final_fitted["params_by_asset"]).to_csv(args.results_dir / "final_params_by_asset.csv", index=False)

    row_order_matches_sample = None
    if args.sample_submission.exists():
        sample = pd.read_csv(args.sample_submission, usecols=["row_id"])
        row_order_matches_sample = bool(submission["row_id"].equals(sample["row_id"]))

    metrics = {
        "leakage_safe": True,
        "strategy_selected_on": "second_half_of_calibration_only",
        "final_parameters_refit_on": "full_calibration_after_strategy_selection",
        "official_test_used_for": "prediction_output_only",
        "selected_strategy": selected_strategy,
        "input_files": {
            "left_calibration": str(args.left_calibration),
            "right_calibration": str(args.right_calibration),
            "left_test": str(args.left_test),
            "right_test": str(args.right_test),
        },
        "grid": {
            "min_left_weight": args.min_left_weight,
            "max_left_weight": args.max_left_weight,
            "step": args.step,
            "min_shrink": args.min_shrink,
            "max_shrink": args.max_shrink,
            "shrink_step": args.shrink_step,
        },
        "calibration_split": {
            "fit_time_range": [int(fit_frame["time_id"].min()), int(fit_frame["time_id"].max())],
            "holdout_time_range": [int(holdout_frame["time_id"].min()), int(holdout_frame["time_id"].max())],
        },
        "strategy_selection": strategy_table.to_dict(orient="records"),
        "final_fitted": final_fitted,
        "full_calibration_score": score_frame(final_cal, "prediction"),
        "full_calibration_score_by_asset": score_by_asset(final_cal, "prediction"),
        "selected_holdout_score": float(strategy_table.loc[0, "holdout_half_score"]),
        "left_full_calibration_score": weighted_zero_mean_r2(
            cal_frame["target"].to_numpy(dtype=np.float64),
            cal_frame[left_col].to_numpy(dtype=np.float64),
            cal_frame["weight"].to_numpy(dtype=np.float64),
        ),
        "right_full_calibration_score": weighted_zero_mean_r2(
            cal_frame["target"].to_numpy(dtype=np.float64),
            cal_frame[right_col].to_numpy(dtype=np.float64),
            cal_frame["weight"].to_numpy(dtype=np.float64),
        ),
        "test_prediction_stats": prediction_stats(final_test["prediction"]),
        "submission": {
            "path": str(submission_path),
            "zip_path": str(args.results_dir / "submission.zip"),
            "rows": int(len(submission)),
            "columns": submission.columns.tolist(),
            "row_order_matches_sample": row_order_matches_sample,
            "target_stats": prediction_stats(submission["target"]),
        },
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
