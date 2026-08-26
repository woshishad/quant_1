from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export smoke validation predictions and charts.")
    parser.add_argument("--release-root", type=Path, default=Path("data/public_release_smoke"))
    parser.add_argument("--model-path", type=Path, default=Path("models/linear_window_smoke/linear_model.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/linear_window_smoke"))
    parser.add_argument("--valid-time-fraction", type=float, default=0.2)
    return parser.parse_args()


def add_baseline_to_path() -> None:
    # 复用官方 baseline 的数据读取、时序切分和滚动特征构造，避免报告脚本和训练逻辑不一致。
    baseline_dir = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "public_release_20260630"
        / "public_release_20260630"
        / "examples"
        / "linear_window_strategy"
    )
    sys.path.insert(0, str(baseline_dir))


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    denominator = np.sum(weight * y_true * y_true)
    if denominator <= 0:
        return 0.0
    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    return float(1.0 - numerator / denominator)


def predict_with_saved_linear_model(matrix: pd.DataFrame, payload: dict) -> np.ndarray:
    # 训练脚本保存的是截距 + 标准化后特征系数，这里按同一公式恢复预测。
    mean = np.asarray(payload["mean"], dtype=np.float64)
    scale = np.asarray(payload["scale"], dtype=np.float64)
    coef = np.asarray(payload["coef"], dtype=np.float64)
    x = (matrix - mean) / scale
    x = np.column_stack([np.ones(len(x), dtype=np.float64), x.to_numpy(dtype=np.float64)])
    return x @ coef


def save_scatter_plot(report: pd.DataFrame, output_path: Path, score: float) -> None:
    plt.figure(figsize=(8, 6))
    sample = report.sample(n=min(len(report), 5000), random_state=42)
    plt.scatter(sample["target"], sample["prediction"], s=6, alpha=0.25)
    limit = float(np.nanmax(np.abs(sample[["target", "prediction"]].to_numpy())))
    limit = max(limit, 1e-6)
    plt.plot([-limit, limit], [-limit, limit], color="black", linewidth=1)
    plt.title(f"Validation Target vs Prediction, weighted R2={score:.4f}")
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_time_plot(report: pd.DataFrame, output_path: Path) -> None:
    by_time = report.groupby("time_id", as_index=False).agg(
        target_mean=("target", "mean"),
        prediction_mean=("prediction", "mean"),
    )
    plt.figure(figsize=(10, 5))
    plt.plot(by_time["time_id"], by_time["target_mean"], label="target mean", linewidth=1.5)
    plt.plot(by_time["time_id"], by_time["prediction_mean"], label="prediction mean", linewidth=1.5)
    plt.title("Validation Mean by time_id")
    plt.xlabel("time_id")
    plt.ylabel("mean value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_coef_plot(payload: dict, output_path: Path, top_n: int = 30) -> None:
    coef = np.asarray(payload["coef"][1:], dtype=np.float64)
    names = np.asarray(payload["derived_columns"], dtype=object)
    order = np.argsort(np.abs(coef))[-top_n:]
    names = names[order]
    values = coef[order]

    plt.figure(figsize=(10, 8))
    colors = np.where(values >= 0, "#2ca02c", "#d62728")
    plt.barh(np.arange(len(values)), values, color=colors)
    plt.yticks(np.arange(len(values)), names)
    plt.title(f"Top {top_n} Linear Coefficients by Absolute Value")
    plt.xlabel("coefficient")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    add_baseline_to_path()
    from train import build_training_matrix, load_train_frame, time_series_split

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.model_path.read_text(encoding="utf-8"))

    # 使用和训练阶段一致的时序后 20% 作为验证集。
    frame = load_train_frame(args.release_root)
    _, valid = time_series_split(frame, args.valid_time_fraction)
    matrix = build_training_matrix(valid, payload["feature_columns"], int(payload["window_size"]))

    prediction = predict_with_saved_linear_model(matrix, payload)
    target = pd.to_numeric(valid["target"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    weight = pd.to_numeric(valid["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    score = weighted_zero_mean_r2(target, prediction, weight)

    report = pd.DataFrame(
        {
            "row_id": valid["row_id"].to_numpy(),
            "time_id": valid["time_id"].to_numpy(),
            "asset_id": valid["asset_id"].to_numpy(),
            "target": target,
            "prediction": prediction,
            "weight": weight,
            "error": prediction - target,
        }
    )
    report.to_csv(args.output_dir / "validation_predictions.csv", index=False)

    summary = {
        "model_path": str(args.model_path),
        "release_root": str(args.release_root),
        "validation_rows": int(len(report)),
        "weighted_zero_mean_r2": score,
        "saved_files": [
            "validation_predictions.csv",
            "target_vs_prediction.png",
            "mean_by_time.png",
            "top_coefficients.png",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_scatter_plot(report, args.output_dir / "target_vs_prediction.png", score)
    save_time_plot(report, args.output_dir / "mean_by_time.png")
    save_coef_plot(payload, args.output_dir / "top_coefficients.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
