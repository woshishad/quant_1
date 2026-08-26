from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from final_train_predict import reorder_like_sample
from walk_forward_tabular import (
    apply_shrink,
    calibrate_shrink_info,
    score_candidate_on_calibration,
    summarize_shrink_info,
    weighted_zero_mean_r2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 time_id 对预测做横截面中性化，并在 calibration 上搜索中性化强度和 shrink。"
    )
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--prediction-column", type=str, default="prediction")
    parser.add_argument("--alpha-min", type=float, default=-1.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--shrink-mode", choices=["none", "global", "per_asset"], default="per_asset")
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[1.0, 1.2, 1.4])
    parser.add_argument("--candidate-score-mode", choices=["full", "mean_halves", "min_halves"], default="full")
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data/sample_submission.csv"),
    )
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def save_zip(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=csv_path.name)


def neutralize_by_time(frame: pd.DataFrame, prediction_column: str, alpha: float) -> np.ndarray:
    """每个 time_id 内减去 alpha 倍横截面均值；alpha=0 等于不处理。"""
    prediction = frame[prediction_column].to_numpy(dtype=np.float64)
    time_mean = frame.groupby("time_id", sort=False)[prediction_column].transform("mean").to_numpy(dtype=np.float64)
    return prediction - float(alpha) * time_mean


def score_time_blocks(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    block_count: int,
) -> dict[str, float]:
    unique_times = np.unique(time_id)
    values = []
    for chunk in np.array_split(unique_times, block_count):
        mask = (time_id >= int(chunk[0])) & (time_id <= int(chunk[-1]))
        values.append(float(weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])))
    scores = np.asarray(values, dtype=np.float64)
    return {
        f"block{block_count}_mean": float(np.mean(scores)),
        f"block{block_count}_min": float(np.min(scores)),
        f"block{block_count}_negative_count": int(np.sum(scores < 0.0)),
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    cal = pd.read_csv(args.calibration_predictions)
    required = {"time_id", "asset_id", "target", "weight", args.prediction_column}
    missing = sorted(required - set(cal.columns))
    if missing:
        raise ValueError(f"calibration 文件缺少列: {missing}")

    y = cal["target"].to_numpy(dtype=np.float64)
    w = cal["weight"].to_numpy(dtype=np.float64)
    asset = cal["asset_id"].to_numpy(dtype=np.int64)
    time_id = cal["time_id"].to_numpy(dtype=np.int64)

    rows = []
    best: dict | None = None
    alpha_values = np.arange(args.alpha_min, args.alpha_max + 1e-12, args.alpha_step)
    for alpha in alpha_values:
        raw_prediction = neutralize_by_time(cal, args.prediction_column, float(alpha))
        if args.shrink_mode == "none":
            shrink_candidates = [{"mode": "none", "cap": None, "global": 1.0, "by_asset": {}}]
        else:
            shrink_candidates = [
                calibrate_shrink_info(y, raw_prediction, w, asset, args.shrink_mode, float(cap))
                for cap in args.shrink_cap_candidates
            ]
        for shrink_info in shrink_candidates:
            prediction = raw_prediction if args.shrink_mode == "none" else apply_shrink(raw_prediction, asset, shrink_info)
            score_info = score_candidate_on_calibration(y, prediction, w, time_id, args.candidate_score_mode)
            shrink_summary = (
                {"cal_shrink": 1.0, "cal_shrink_min": 1.0, "cal_shrink_mean": 1.0, "cal_shrink_max": 1.0}
                if args.shrink_mode == "none"
                else summarize_shrink_info(shrink_info)
            )
            row = {
                "alpha": float(alpha),
                "selection_score": float(score_info["selection_score"]),
                "full_score": float(score_info["full_score"]),
                "first_half_score": float(score_info["first_half_score"]),
                "second_half_score": float(score_info["second_half_score"]),
                "shrink_mode": args.shrink_mode,
                "shrink_cap": None if shrink_info.get("cap") is None else float(shrink_info["cap"]),
                "shrink": float(shrink_summary["cal_shrink"]),
                "shrink_min": float(shrink_summary["cal_shrink_min"]),
                "shrink_mean": float(shrink_summary["cal_shrink_mean"]),
                "shrink_max": float(shrink_summary["cal_shrink_max"]),
                "prediction_std": float(np.std(prediction)),
                "raw_prediction_std": float(np.std(raw_prediction)),
                "shrink_info": json.dumps(shrink_info, ensure_ascii=False, default=json_default),
                **score_time_blocks(y, prediction, w, time_id, 4),
                **score_time_blocks(y, prediction, w, time_id, 8),
            }
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = {
                    **row,
                    "_prediction": prediction,
                    "_raw_prediction": raw_prediction,
                    "_shrink_info": shrink_info,
                }

    if best is None:
        raise ValueError("没有找到可用中性化候选")

    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(
        args.results_dir / "neutralization_candidate_metrics.csv",
        index=False,
    )
    cal_output = cal.copy()
    cal_output["neutralized_raw_prediction"] = np.asarray(best["_raw_prediction"], dtype=np.float32)
    cal_output["prediction"] = np.asarray(best["_prediction"], dtype=np.float32)
    cal_output["error"] = cal_output["prediction"] - cal_output["target"]
    cal_output.to_csv(args.results_dir / "neutralized_calibration_predictions.csv", index=False)

    output_files = {
        "candidate_metrics": str(args.results_dir / "neutralization_candidate_metrics.csv"),
        "calibration_predictions": str(args.results_dir / "neutralized_calibration_predictions.csv"),
    }
    test_stats = None
    if args.test_predictions is not None:
        test = pd.read_csv(args.test_predictions)
        required_test = {"row_id", "time_id", "asset_id", args.prediction_column}
        missing_test = sorted(required_test - set(test.columns))
        if missing_test:
            raise ValueError(f"test 文件缺少列: {missing_test}")
        raw_test = neutralize_by_time(test, args.prediction_column, float(best["alpha"]))
        asset_test = test["asset_id"].to_numpy(dtype=np.int64)
        prediction_test = (
            raw_test
            if args.shrink_mode == "none"
            else apply_shrink(raw_test, asset_test, best["_shrink_info"])
        )
        test_output = test.copy()
        test_output["neutralized_raw_prediction"] = raw_test.astype(np.float32)
        test_output["prediction"] = prediction_test.astype(np.float32)
        test_output.to_csv(args.results_dir / "final_test_predictions.csv", index=False)
        submission = test_output[["row_id", "prediction"]].rename(columns={"prediction": "target"})
        submission = reorder_like_sample(submission, args.sample_submission)
        submission_path = args.results_dir / "submission.csv"
        zip_path = args.results_dir / "submission.zip"
        submission.to_csv(submission_path, index=False)
        save_zip(submission_path, zip_path)
        output_files.update(
            {
                "final_test_predictions": str(args.results_dir / "final_test_predictions.csv"),
                "submission": str(submission_path),
                "submission_zip": str(zip_path),
            }
        )
        test_stats = {
            "mean": float(np.mean(prediction_test)),
            "std": float(np.std(prediction_test)),
            "min": float(np.min(prediction_test)),
            "max": float(np.max(prediction_test)),
            "null_count": int(np.sum(~np.isfinite(prediction_test))),
            "finite_count": int(np.sum(np.isfinite(prediction_test))),
        }

    summary = {
        "leakage_safe": True,
        "official_test_used": False,
        "method": "time_id cross-sectional prediction neutralization",
        "source_calibration_predictions": str(args.calibration_predictions),
        "source_test_predictions": None if args.test_predictions is None else str(args.test_predictions),
        "prediction_column": args.prediction_column,
        "candidate_score_mode": args.candidate_score_mode,
        "best_candidate": {key: value for key, value in best.items() if not key.startswith("_")},
        "test_prediction_stats": test_stats,
        "output_files": output_files,
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
