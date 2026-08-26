from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic prediction blend analysis. Oracle rows use holdout target.")
    parser.add_argument("--left-predictions", type=Path, required=True)
    parser.add_argument("--right-predictions", type=Path, required=True)
    parser.add_argument("--left-name", type=str, default="left")
    parser.add_argument("--right-name", type=str, default="right")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.01)
    return parser.parse_args()


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.sum(weight * y_true * y_true))
    if denominator <= 0.0:
        return 0.0
    return 1.0 - float(np.sum(weight * (y_true - y_pred) ** 2)) / denominator


def score_by_asset(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    scores = {}
    for asset, asset_frame in frame.groupby("asset_id", sort=True):
        scores[str(int(asset))] = weighted_zero_mean_r2(
            asset_frame["target"].to_numpy(dtype=np.float64),
            asset_frame[prediction_column].to_numpy(dtype=np.float64),
            asset_frame["weight"].to_numpy(dtype=np.float64),
        )
    return scores


def find_oracle_blend(
    y_true: np.ndarray,
    left_pred: np.ndarray,
    right_pred: np.ndarray,
    weight: np.ndarray,
    step: float,
) -> tuple[float, float]:
    # 注意：oracle blend 用了 holdout target，只能用于诊断互补性，不能作为无泄漏最终分数。
    best_weight = 0.0
    best_score = -np.inf
    for left_weight in np.arange(0.0, 1.0 + 1e-12, step):
        prediction = left_weight * left_pred + (1.0 - left_weight) * right_pred
        score = weighted_zero_mean_r2(y_true, prediction, weight)
        if score > best_score:
            best_weight = float(left_weight)
            best_score = float(score)
    return best_weight, best_score


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    left = pd.read_csv(args.left_predictions, usecols=["row_id", "time_id", "asset_id", "target", "weight", "prediction"])
    right = pd.read_csv(args.right_predictions, usecols=["row_id", "prediction"])
    frame = left.merge(right, on="row_id", suffixes=(f"_{args.left_name}", f"_{args.right_name}"))
    left_col = f"prediction_{args.left_name}"
    right_col = f"prediction_{args.right_name}"

    y_true = frame["target"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    left_pred = frame[left_col].to_numpy(dtype=np.float64)
    right_pred = frame[right_col].to_numpy(dtype=np.float64)

    global_w, global_oracle_score = find_oracle_blend(y_true, left_pred, right_pred, weight, args.step)
    frame["oracle_global_blend"] = global_w * left_pred + (1.0 - global_w) * right_pred

    rows = []
    oracle_asset_prediction = np.zeros(len(frame), dtype=np.float64)
    for asset, asset_frame in frame.groupby("asset_id", sort=True):
        index = asset_frame.index.to_numpy()
        asset_y = asset_frame["target"].to_numpy(dtype=np.float64)
        asset_w = asset_frame["weight"].to_numpy(dtype=np.float64)
        asset_left = asset_frame[left_col].to_numpy(dtype=np.float64)
        asset_right = asset_frame[right_col].to_numpy(dtype=np.float64)
        asset_blend_w, asset_oracle_score = find_oracle_blend(asset_y, asset_left, asset_right, asset_w, args.step)
        oracle_asset_prediction[index] = asset_blend_w * asset_left + (1.0 - asset_blend_w) * asset_right
        rows.append(
            {
                "asset_id": int(asset),
                f"{args.left_name}_score": weighted_zero_mean_r2(asset_y, asset_left, asset_w),
                f"{args.right_name}_score": weighted_zero_mean_r2(asset_y, asset_right, asset_w),
                "oracle_left_weight": asset_blend_w,
                "oracle_score": asset_oracle_score,
            }
        )
    frame["oracle_by_asset_blend"] = oracle_asset_prediction

    asset_frame = pd.DataFrame(rows)
    asset_frame.to_csv(args.results_dir / "oracle_blend_by_asset.csv", index=False)

    metrics = {
        "diagnostic_only": True,
        "uses_holdout_target_for_oracle": True,
        "left_name": args.left_name,
        "right_name": args.right_name,
        "row_count": int(len(frame)),
        "scores": {
            args.left_name: weighted_zero_mean_r2(y_true, left_pred, weight),
            args.right_name: weighted_zero_mean_r2(y_true, right_pred, weight),
            "oracle_global_blend": weighted_zero_mean_r2(
                y_true, frame["oracle_global_blend"].to_numpy(dtype=np.float64), weight
            ),
            "oracle_by_asset_blend": weighted_zero_mean_r2(
                y_true, frame["oracle_by_asset_blend"].to_numpy(dtype=np.float64), weight
            ),
        },
        "oracle_global_left_weight": global_w,
        "score_by_asset": {
            args.left_name: score_by_asset(frame, left_col),
            args.right_name: score_by_asset(frame, right_col),
            "oracle_global_blend": score_by_asset(frame, "oracle_global_blend"),
            "oracle_by_asset_blend": score_by_asset(frame, "oracle_by_asset_blend"),
        },
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
