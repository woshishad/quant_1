from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from final_train_predict import reorder_like_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按固定权重融合多个 official test prediction 文件，并生成 submission.zip。"
    )
    parser.add_argument("--prediction-files", type=Path, nargs="+", required=True)
    parser.add_argument("--prediction-names", type=str, nargs="+", required=True)
    parser.add_argument("--prediction-columns", type=str, nargs="+", default=None)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--global-shrink", type=float, default=1.0)
    parser.add_argument("--results-dir", type=Path, default=None)
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


def make_results_dir(path: Path | None) -> Path:
    if path is not None:
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"test_prediction_blend_{timestamp}"


def load_prediction_file(path: Path, name: str, prediction_column: str) -> pd.DataFrame:
    """读取 test 预测文件，只保留提交和融合必要字段。"""
    use_columns = ["row_id", "time_id", "asset_id", prediction_column]
    frame = pd.read_csv(path, usecols=use_columns)
    frame = frame.rename(columns={prediction_column: f"prediction_{name}"})
    return frame.sort_values("row_id", kind="mergesort").reset_index(drop=True)


def merge_prediction_files(files: list[Path], names: list[str], columns: list[str]) -> pd.DataFrame:
    """按 row_id 对齐多个 test 预测；time_id/asset_id 用来做一致性校验。"""
    merged = None
    for path, name, column in zip(files, names, columns):
        frame = load_prediction_file(path, name, column)
        if merged is None:
            merged = frame
            continue
        suffix = f"_{name}"
        merged = merged.merge(frame, on="row_id", how="inner", suffixes=("", suffix))
        if not np.array_equal(merged["time_id"].to_numpy(), merged[f"time_id{suffix}"].to_numpy()):
            raise ValueError(f"{name} 与第一个预测文件的 time_id 不一致")
        if not np.array_equal(merged["asset_id"].to_numpy(), merged[f"asset_id{suffix}"].to_numpy()):
            raise ValueError(f"{name} 与第一个预测文件的 asset_id 不一致")
        merged = merged.drop(columns=[f"time_id{suffix}", f"asset_id{suffix}"])
    if merged is None or merged.empty:
        raise ValueError("没有读到可融合的 test 预测")
    return merged


def save_zip(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=csv_path.name)


def main() -> None:
    args = parse_args()
    if len(args.prediction_files) != len(args.prediction_names):
        raise ValueError("--prediction-files 和 --prediction-names 数量必须一致")
    if len(args.weights) != len(args.prediction_files):
        raise ValueError("--weights 数量必须和 --prediction-files 一致")
    prediction_columns = args.prediction_columns or ["prediction"] * len(args.prediction_files)
    if len(prediction_columns) != len(args.prediction_files):
        raise ValueError("--prediction-columns 数量必须和 --prediction-files 一致")
    weight_sum = float(np.sum(args.weights))
    if weight_sum <= 0:
        raise ValueError("--weights 之和必须大于 0")
    normalized_weights = np.asarray(args.weights, dtype=np.float64) / weight_sum

    args.results_dir = make_results_dir(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_prediction_files(args.prediction_files, args.prediction_names, prediction_columns)
    matrix = np.column_stack(
        [merged[f"prediction_{name}"].to_numpy(dtype=np.float64) for name in args.prediction_names]
    )
    raw_prediction = matrix @ normalized_weights
    final_prediction = float(args.global_shrink) * raw_prediction

    output = merged[["row_id", "time_id", "asset_id"]].copy()
    for name in args.prediction_names:
        output[f"prediction_{name}"] = merged[f"prediction_{name}"].to_numpy(dtype=np.float32)
    output["raw_prediction"] = raw_prediction.astype(np.float32)
    output["global_shrink"] = float(args.global_shrink)
    output["prediction"] = final_prediction.astype(np.float32)
    output.to_csv(args.results_dir / "final_test_predictions.csv", index=False)

    submission = output[["row_id", "prediction"]].rename(columns={"prediction": "target"})
    submission = reorder_like_sample(submission, args.sample_submission)
    submission_path = args.results_dir / "submission.csv"
    zip_path = args.results_dir / "submission.zip"
    submission.to_csv(submission_path, index=False)
    save_zip(submission_path, zip_path)

    metrics = {
        "official_test_used_for_training": False,
        "prediction_files": [str(path) for path in args.prediction_files],
        "prediction_names": args.prediction_names,
        "prediction_columns": prediction_columns,
        "weights_input": [float(weight) for weight in args.weights],
        "weights_normalized": {
            name: float(weight) for name, weight in zip(args.prediction_names, normalized_weights)
        },
        "global_shrink": float(args.global_shrink),
        "rows": int(len(output)),
        "time_min": int(output["time_id"].min()),
        "time_max": int(output["time_id"].max()),
        "prediction_stats": {
            "mean": float(np.mean(final_prediction)),
            "std": float(np.std(final_prediction)),
            "min": float(np.min(final_prediction)),
            "max": float(np.max(final_prediction)),
            "null_count": int(np.sum(~np.isfinite(final_prediction))),
            "finite_count": int(np.sum(np.isfinite(final_prediction))),
        },
        "output_files": {
            "final_test_predictions": str(args.results_dir / "final_test_predictions.csv"),
            "submission": str(submission_path),
            "submission_zip": str(zip_path),
        },
    }
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
