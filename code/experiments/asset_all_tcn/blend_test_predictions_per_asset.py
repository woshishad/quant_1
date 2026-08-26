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
        description="按 asset_id 使用不同融合权重和 shrink，生成 official test submission。"
    )
    parser.add_argument("--prediction-files", type=Path, nargs="+", required=True)
    parser.add_argument("--prediction-names", type=str, nargs="+", required=True)
    parser.add_argument("--prediction-columns", type=str, nargs="+", default=None)
    parser.add_argument("--per-asset-params", type=Path, required=True)
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
    return Path("results") / f"test_prediction_per_asset_blend_{timestamp}"


def load_prediction_file(path: Path, name: str, prediction_column: str) -> pd.DataFrame:
    """读取单个 test 预测文件，统一预测列名，方便后面按 row_id 合并。"""
    frame = pd.read_csv(path, usecols=["row_id", "time_id", "asset_id", prediction_column])
    return frame.rename(columns={prediction_column: f"prediction_{name}"}).sort_values(
        "row_id",
        kind="mergesort",
    )


def merge_prediction_files(files: list[Path], names: list[str], columns: list[str]) -> pd.DataFrame:
    """按 row_id 对齐多个模型输出，并校验 time_id/asset_id 完全一致。"""
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
    prediction_columns = args.prediction_columns or ["prediction"] * len(args.prediction_files)
    if len(prediction_columns) != len(args.prediction_files):
        raise ValueError("--prediction-columns 数量必须和 --prediction-files 一致")

    args.results_dir = make_results_dir(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    params = pd.read_csv(args.per_asset_params)
    required_columns = {"asset_id", "shrink"} | {f"weight_{name}" for name in args.prediction_names}
    missing = sorted(required_columns - set(params.columns))
    if missing:
        raise ValueError(f"per-asset 参数文件缺少列: {missing}")

    merged = merge_prediction_files(args.prediction_files, args.prediction_names, prediction_columns)
    prediction = np.zeros(len(merged), dtype=np.float64)
    raw_prediction = np.zeros(len(merged), dtype=np.float64)
    asset_values = merged["asset_id"].to_numpy(dtype=np.int64)

    for _, row in params.iterrows():
        asset = int(row["asset_id"])
        mask = asset_values == asset
        if not np.any(mask):
            continue
        raw = np.zeros(int(mask.sum()), dtype=np.float64)
        for name in args.prediction_names:
            raw += float(row[f"weight_{name}"]) * merged.loc[mask, f"prediction_{name}"].to_numpy(dtype=np.float64)
        raw_prediction[mask] = raw
        prediction[mask] = float(row["shrink"]) * raw

    missing_asset_count = int(np.sum((raw_prediction == 0.0) & ~np.isin(asset_values, params["asset_id"].astype(int))))
    if missing_asset_count:
        raise ValueError(f"存在没有 per-asset 参数的预测行: {missing_asset_count}")

    output = merged[["row_id", "time_id", "asset_id"]].copy()
    for name in args.prediction_names:
        output[f"prediction_{name}"] = merged[f"prediction_{name}"].to_numpy(dtype=np.float32)
    output["raw_prediction"] = raw_prediction.astype(np.float32)
    output["prediction"] = prediction.astype(np.float32)
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
        "per_asset_params": str(args.per_asset_params),
        "rows": int(len(output)),
        "time_min": int(output["time_id"].min()),
        "time_max": int(output["time_id"].max()),
        "prediction_stats": {
            "mean": float(np.mean(prediction)),
            "std": float(np.std(prediction)),
            "min": float(np.min(prediction)),
            "max": float(np.max(prediction)),
            "null_count": int(np.sum(~np.isfinite(prediction))),
            "finite_count": int(np.sum(np.isfinite(prediction))),
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
