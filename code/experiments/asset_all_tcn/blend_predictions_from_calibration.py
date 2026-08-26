from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SAMPLE_SUBMISSION = Path(
    "data/raw/public_release_20260630/public_release_20260630/data/sample_submission.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只用 train 内部 calibration 段学习两个提交/预测文件的融合权重，再应用到官方 test。"
    )
    parser.add_argument("--left-calibration", type=Path, required=True, help="左侧模型的 calibration_predictions.csv")
    parser.add_argument("--right-calibration", type=Path, required=True, help="右侧模型的 calibration_predictions.csv")
    parser.add_argument("--left-test", type=Path, required=True, help="左侧模型的 final_test_predictions.csv 或 submission.csv")
    parser.add_argument("--right-test", type=Path, required=True, help="右侧模型的 final_test_predictions.csv 或 submission.csv")
    parser.add_argument("--left-name", type=str, default="left", help="左侧模型名字，会写入输出列名")
    parser.add_argument("--right-name", type=str, default="right", help="右侧模型名字，会写入输出列名")
    parser.add_argument("--prediction-column", type=str, default="prediction", help="输入文件里作为预测值使用的列名")
    parser.add_argument("--results-dir", type=Path, required=True, help="融合结果输出目录")
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION, help="用于校验 row_id 顺序")
    parser.add_argument("--blend-mode", choices=["global", "per_asset"], default="global", help="全局权重或逐 asset 权重")
    parser.add_argument("--min-left-weight", type=float, default=0.0)
    parser.add_argument("--max-left-weight", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.01, help="融合权重搜索步长")
    parser.add_argument("--min-shrink", type=float, default=1.0, help="融合后整体缩放系数下界")
    parser.add_argument("--max-shrink", type=float, default=1.0, help="融合后整体缩放系数上界")
    parser.add_argument("--shrink-step", type=float, default=0.01, help="缩放系数搜索步长")
    return parser.parse_args()


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    """比赛里常用的 zero-mean weighted R2：分母不减均值，负数表示不如预测 0。"""
    denominator = float(np.sum(weight * y_true * y_true))
    if denominator <= 0.0:
        return 0.0
    numerator = float(np.sum(weight * (y_true - y_pred) ** 2))
    return 1.0 - numerator / denominator


def score_by_asset(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    """逐标的计算分数，用来检查是不是某几个 asset 把总分拖下去了。"""
    scores: dict[str, float] = {}
    for asset, asset_frame in frame.groupby("asset_id", sort=True):
        scores[str(int(asset))] = weighted_zero_mean_r2(
            asset_frame["target"].to_numpy(dtype=np.float64),
            asset_frame[prediction_column].to_numpy(dtype=np.float64),
            asset_frame["weight"].to_numpy(dtype=np.float64),
        )
    return scores


def numeric_grid(start: float, stop: float, step: float) -> np.ndarray:
    """避免浮点步长累积误差导致 1.0 这种右端点被漏掉。"""
    if step <= 0:
        raise ValueError("grid step must be positive")
    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    values = start + step * np.arange(count, dtype=np.float64)
    if values.size == 0 or values[-1] < stop - 1e-12:
        values = np.append(values, stop)
    return np.round(values, 12)


def read_prediction_file(path: Path, prediction_column: str, required_context: list[str]) -> pd.DataFrame:
    """按需读取列，减少 300 万行 test 文件的内存占用。"""
    header = pd.read_csv(path, nrows=0)
    columns = set(header.columns)
    missing = [column for column in ["row_id", prediction_column, *required_context] if column not in columns]
    if missing:
        raise ValueError(f"{path} 缺少必要列: {missing}")

    optional_context = [column for column in ["time_id", "asset_id"] if column in columns and column not in required_context]
    usecols = ["row_id", *required_context, *optional_context, prediction_column]
    return pd.read_csv(path, usecols=usecols)


def merge_calibration(args: argparse.Namespace) -> pd.DataFrame:
    """合并 calibration 预测；target/weight 只来自左侧，右侧只取预测列。"""
    required = ["time_id", "asset_id", "target", "weight"]
    left = read_prediction_file(args.left_calibration, args.prediction_column, required)
    right = read_prediction_file(args.right_calibration, args.prediction_column, [])
    left = left.rename(columns={args.prediction_column: f"prediction_{args.left_name}"})
    right = right.rename(columns={args.prediction_column: f"prediction_{args.right_name}"})
    frame = left.merge(right[["row_id", f"prediction_{args.right_name}"]], on="row_id", how="inner", validate="one_to_one")
    if len(frame) != len(left):
        raise ValueError("calibration 合并后行数变化，请检查两个 calibration 文件的 row_id 是否一致")
    return frame


def merge_test(args: argparse.Namespace) -> pd.DataFrame:
    """合并官方 test 预测；这里没有 target，所以只负责生成最终 prediction。"""
    left = read_prediction_file(args.left_test, args.prediction_column, [])
    right = read_prediction_file(args.right_test, args.prediction_column, [])
    left = left.rename(columns={args.prediction_column: f"prediction_{args.left_name}"})
    right = right.rename(columns={args.prediction_column: f"prediction_{args.right_name}"})
    frame = left.merge(right[["row_id", f"prediction_{args.right_name}"]], on="row_id", how="inner", validate="one_to_one")
    if len(frame) != len(left):
        raise ValueError("test 合并后行数变化，请检查两个 test 文件的 row_id 是否一致")
    return frame


def search_weight_and_shrink(
    y_true: np.ndarray,
    left_pred: np.ndarray,
    right_pred: np.ndarray,
    weight: np.ndarray,
    left_weight_grid: np.ndarray,
    shrink_grid: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """在 calibration 上网格搜索融合权重和 shrink；官方 test 不参与选择。"""
    history: list[dict[str, float]] = []
    best = {"score": -np.inf, "left_weight": float(left_weight_grid[0]), "shrink": float(shrink_grid[0])}

    for left_weight in left_weight_grid:
        base_prediction = left_weight * left_pred + (1.0 - left_weight) * right_pred
        for shrink in shrink_grid:
            prediction = shrink * base_prediction
            score = weighted_zero_mean_r2(y_true, prediction, weight)
            row = {"left_weight": float(left_weight), "right_weight": float(1.0 - left_weight), "shrink": float(shrink), "score": float(score)}
            history.append(row)
            if score > best["score"]:
                best = row.copy()
    return best, history


def apply_global_blend(
    cal_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    args: argparse.Namespace,
    left_weight_grid: np.ndarray,
    shrink_grid: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    left_col = f"prediction_{args.left_name}"
    right_col = f"prediction_{args.right_name}"
    best, history = search_weight_and_shrink(
        cal_frame["target"].to_numpy(dtype=np.float64),
        cal_frame[left_col].to_numpy(dtype=np.float64),
        cal_frame[right_col].to_numpy(dtype=np.float64),
        cal_frame["weight"].to_numpy(dtype=np.float64),
        left_weight_grid,
        shrink_grid,
    )

    # 学到的权重固定应用到 calibration 和官方 test，方便审计和复现实验。
    for frame in (cal_frame, test_frame):
        frame["blend_left_weight"] = best["left_weight"]
        frame["blend_right_weight"] = best["right_weight"]
        frame["blend_shrink"] = best["shrink"]
        frame["prediction"] = best["shrink"] * (best["left_weight"] * frame[left_col] + best["right_weight"] * frame[right_col])

    return {"mode": "global", **best}, pd.DataFrame(history)


def apply_per_asset_blend(
    cal_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    args: argparse.Namespace,
    left_weight_grid: np.ndarray,
    shrink_grid: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    if "asset_id" not in test_frame.columns:
        raise ValueError("per_asset 融合需要 test 文件里包含 asset_id，请传 final_test_predictions.csv 而不是 submission.csv")

    left_col = f"prediction_{args.left_name}"
    right_col = f"prediction_{args.right_name}"
    weights_by_asset = []
    history_rows = []
    for column in ["blend_left_weight", "blend_right_weight", "blend_shrink", "prediction"]:
        cal_frame[column] = np.nan
        test_frame[column] = np.nan

    for asset, cal_asset in cal_frame.groupby("asset_id", sort=True):
        best, history = search_weight_and_shrink(
            cal_asset["target"].to_numpy(dtype=np.float64),
            cal_asset[left_col].to_numpy(dtype=np.float64),
            cal_asset[right_col].to_numpy(dtype=np.float64),
            cal_asset["weight"].to_numpy(dtype=np.float64),
            left_weight_grid,
            shrink_grid,
        )
        best["asset_id"] = int(asset)
        weights_by_asset.append(best)
        for row in history:
            row["asset_id"] = int(asset)
            history_rows.append(row)

        # 每个 asset 使用自己在 calibration 上学到的权重。
        for frame in (cal_frame, test_frame):
            mask = frame["asset_id"] == asset
            frame.loc[mask, "blend_left_weight"] = best["left_weight"]
            frame.loc[mask, "blend_right_weight"] = best["right_weight"]
            frame.loc[mask, "blend_shrink"] = best["shrink"]
            frame.loc[mask, "prediction"] = best["shrink"] * (
                best["left_weight"] * frame.loc[mask, left_col] + best["right_weight"] * frame.loc[mask, right_col]
            )

    total_score = weighted_zero_mean_r2(
        cal_frame["target"].to_numpy(dtype=np.float64),
        cal_frame["prediction"].to_numpy(dtype=np.float64),
        cal_frame["weight"].to_numpy(dtype=np.float64),
    )
    return {"mode": "per_asset", "cal_score": float(total_score), "by_asset_weights": weights_by_asset}, pd.DataFrame(history_rows)


def time_split_blend_diagnostic(
    cal_frame: pd.DataFrame,
    args: argparse.Namespace,
    left_weight_grid: np.ndarray,
    shrink_grid: np.ndarray,
) -> dict:
    """把 calibration 再切成前半/后半：前半学权重，后半只评估，专门用来观察过拟合。"""
    left_col = f"prediction_{args.left_name}"
    right_col = f"prediction_{args.right_name}"
    unique_times = np.sort(cal_frame["time_id"].unique())
    if len(unique_times) < 2:
        return {"available": False, "reason": "calibration time_id count < 2"}

    split_time = int(unique_times[len(unique_times) // 2 - 1])
    fit_frame = cal_frame[cal_frame["time_id"] <= split_time].copy()
    holdout_frame = cal_frame[cal_frame["time_id"] > split_time].copy()
    if fit_frame.empty or holdout_frame.empty:
        return {"available": False, "reason": "empty fit or holdout after split"}

    if args.blend_mode == "global":
        best, _ = search_weight_and_shrink(
            fit_frame["target"].to_numpy(dtype=np.float64),
            fit_frame[left_col].to_numpy(dtype=np.float64),
            fit_frame[right_col].to_numpy(dtype=np.float64),
            fit_frame["weight"].to_numpy(dtype=np.float64),
            left_weight_grid,
            shrink_grid,
        )
        for frame in (fit_frame, holdout_frame):
            frame["prediction"] = best["shrink"] * (
                best["left_weight"] * frame[left_col] + best["right_weight"] * frame[right_col]
            )
        weights = {"mode": "global", **best}
    else:
        weights_by_asset = []
        fit_frame["prediction"] = np.nan
        holdout_frame["prediction"] = np.nan
        for asset, fit_asset in fit_frame.groupby("asset_id", sort=True):
            best, _ = search_weight_and_shrink(
                fit_asset["target"].to_numpy(dtype=np.float64),
                fit_asset[left_col].to_numpy(dtype=np.float64),
                fit_asset[right_col].to_numpy(dtype=np.float64),
                fit_asset["weight"].to_numpy(dtype=np.float64),
                left_weight_grid,
                shrink_grid,
            )
            best["asset_id"] = int(asset)
            weights_by_asset.append(best)
            for frame in (fit_frame, holdout_frame):
                mask = frame["asset_id"] == asset
                frame.loc[mask, "prediction"] = best["shrink"] * (
                    best["left_weight"] * frame.loc[mask, left_col]
                    + best["right_weight"] * frame.loc[mask, right_col]
                )
        weights = {"mode": "per_asset", "by_asset_weights": weights_by_asset}

    fit_score = weighted_zero_mean_r2(
        fit_frame["target"].to_numpy(dtype=np.float64),
        fit_frame["prediction"].to_numpy(dtype=np.float64),
        fit_frame["weight"].to_numpy(dtype=np.float64),
    )
    holdout_score = weighted_zero_mean_r2(
        holdout_frame["target"].to_numpy(dtype=np.float64),
        holdout_frame["prediction"].to_numpy(dtype=np.float64),
        holdout_frame["weight"].to_numpy(dtype=np.float64),
    )
    return {
        "available": True,
        "fit_time_range": [int(fit_frame["time_id"].min()), int(fit_frame["time_id"].max())],
        "holdout_time_range": [int(holdout_frame["time_id"].min()), int(holdout_frame["time_id"].max())],
        "weights_learned_on_fit_half": weights,
        "fit_half_score": float(fit_score),
        "holdout_half_score": float(holdout_score),
        "holdout_score_by_asset": score_by_asset(holdout_frame, "prediction"),
    }


def make_submission(test_frame: pd.DataFrame, sample_submission_path: Path) -> pd.DataFrame:
    """严格按 sample_submission 的 row_id 顺序生成提交，避免排序问题。"""
    prediction_frame = test_frame[["row_id", "prediction"]].rename(columns={"prediction": "target"})
    if sample_submission_path.exists():
        sample = pd.read_csv(sample_submission_path, usecols=["row_id"])
        submission = sample.merge(prediction_frame, on="row_id", how="left", validate="one_to_one")
        if submission["target"].isna().any():
            missing_count = int(submission["target"].isna().sum())
            raise ValueError(f"submission 有 {missing_count} 行缺失预测")
        return submission
    return prediction_frame


def write_zip(csv_path: Path, zip_path: Path) -> None:
    """比赛平台通常接受 zip，这里顺手把 submission.csv 压进去。"""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(csv_path, arcname=csv_path.name)


def prediction_stats(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "null_count": int(np.isnan(array).sum()),
        "finite_count": int(np.isfinite(array).sum()),
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.min_left_weight < 0 or args.max_left_weight > 1 or args.min_left_weight > args.max_left_weight:
        raise ValueError("left weight bounds must satisfy 0 <= min <= max <= 1")
    if args.min_shrink < 0 or args.max_shrink < args.min_shrink:
        raise ValueError("shrink bounds must satisfy 0 <= min <= max")

    left_weight_grid = numeric_grid(args.min_left_weight, args.max_left_weight, args.step)
    shrink_grid = numeric_grid(args.min_shrink, args.max_shrink, args.shrink_step)
    cal_frame = merge_calibration(args)
    test_frame = merge_test(args)

    left_col = f"prediction_{args.left_name}"
    right_col = f"prediction_{args.right_name}"
    if args.blend_mode == "global":
        blend_info, search_history = apply_global_blend(cal_frame, test_frame, args, left_weight_grid, shrink_grid)
    else:
        blend_info, search_history = apply_per_asset_blend(cal_frame, test_frame, args, left_weight_grid, shrink_grid)
    time_split_diagnostic = time_split_blend_diagnostic(cal_frame, args, left_weight_grid, shrink_grid)

    cal_frame["error"] = cal_frame["prediction"] - cal_frame["target"]
    submission = make_submission(test_frame, args.sample_submission)

    cal_score = weighted_zero_mean_r2(
        cal_frame["target"].to_numpy(dtype=np.float64),
        cal_frame["prediction"].to_numpy(dtype=np.float64),
        cal_frame["weight"].to_numpy(dtype=np.float64),
    )
    left_cal_score = weighted_zero_mean_r2(
        cal_frame["target"].to_numpy(dtype=np.float64),
        cal_frame[left_col].to_numpy(dtype=np.float64),
        cal_frame["weight"].to_numpy(dtype=np.float64),
    )
    right_cal_score = weighted_zero_mean_r2(
        cal_frame["target"].to_numpy(dtype=np.float64),
        cal_frame[right_col].to_numpy(dtype=np.float64),
        cal_frame["weight"].to_numpy(dtype=np.float64),
    )

    cal_frame.to_csv(args.results_dir / "calibration_predictions.csv", index=False)
    test_frame.to_csv(args.results_dir / "final_test_predictions.csv", index=False)
    search_history.to_csv(args.results_dir / "blend_weight_search.csv", index=False)
    if args.blend_mode == "per_asset":
        pd.DataFrame(blend_info["by_asset_weights"]).to_csv(args.results_dir / "blend_weights_by_asset.csv", index=False)

    submission_path = args.results_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    write_zip(submission_path, args.results_dir / "submission.zip")

    row_order_matches_sample = None
    if args.sample_submission.exists():
        sample = pd.read_csv(args.sample_submission, usecols=["row_id"])
        row_order_matches_sample = bool(submission["row_id"].equals(sample["row_id"]))

    metrics = {
        "leakage_safe": True,
        "blend_learned_on": "calibration_predictions_only",
        "official_test_used_for": "prediction_output_only",
        "input_files": {
            "left_calibration": str(args.left_calibration),
            "right_calibration": str(args.right_calibration),
            "left_test": str(args.left_test),
            "right_test": str(args.right_test),
        },
        "left_name": args.left_name,
        "right_name": args.right_name,
        "blend_mode": args.blend_mode,
        "weight_grid": {
            "min_left_weight": args.min_left_weight,
            "max_left_weight": args.max_left_weight,
            "step": args.step,
            "min_shrink": args.min_shrink,
            "max_shrink": args.max_shrink,
            "shrink_step": args.shrink_step,
        },
        "blend_info": blend_info,
        "time_split_diagnostic": time_split_diagnostic,
        "calibration_score": float(cal_score),
        "left_calibration_score": float(left_cal_score),
        "right_calibration_score": float(right_cal_score),
        "calibration_score_by_asset": score_by_asset(cal_frame, "prediction"),
        "test_prediction_stats": prediction_stats(test_frame["prediction"]),
        "left_right_test_prediction_correlation": float(test_frame[left_col].corr(test_frame[right_col])),
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
