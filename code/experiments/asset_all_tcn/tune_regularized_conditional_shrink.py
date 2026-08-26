from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from walk_forward_tabular import (
    apply_shrink,
    calibrate_shrink_info,
    score_candidate_on_calibration,
    weighted_zero_mean_r2,
)
from market_mean_ts_model import score_time_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对市场状态条件 shrink 做强正则和前向稳定性选择。"
    )
    parser.add_argument(
        "--input-predictions",
        type=Path,
        default=Path(
            "results/asset_all_market_regime_experts_75k_probe/"
            "calibration_predictions.csv"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/regularized_conditional_shrink_75k_probe"),
    )
    parser.add_argument("--beta-min", type=float, default=-0.15)
    parser.add_argument("--beta-max", type=float, default=0.05)
    parser.add_argument("--beta-step", type=float, default=0.01)
    parser.add_argument("--shrink-cap", type=float, default=1.6)
    parser.add_argument(
        "--regularization-strengths",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0],
    )
    parser.add_argument("--scale-floor", type=float, default=0.75)
    parser.add_argument("--scale-cap", type=float, default=1.25)
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


def make_edges(regime: np.ndarray, quantiles: list[float]) -> np.ndarray:
    edges = np.quantile(np.abs(regime), quantiles).astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    for index in range(1, len(edges) - 1):
        if edges[index] <= edges[index - 1]:
            edges[index] = np.nextafter(edges[index - 1], np.inf)
    return edges


def bucket_ids(regime: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(np.abs(regime), edges[1:-1], right=False).astype(np.int16)


def fit_raw_scales(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    regime: np.ndarray,
    quantiles: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """先求每个幅度桶的闭式最优 scale；随后再向 1 强正则。"""
    edges = make_edges(regime, quantiles)
    buckets = bucket_ids(regime, edges)
    scales = []
    for bucket in range(len(edges) - 1):
        mask = buckets == bucket
        denominator = float(np.sum(weight[mask] * prediction[mask] ** 2))
        if denominator <= 1e-18:
            scales.append(1.0)
            continue
        numerator = float(np.sum(weight[mask] * y_true[mask] * prediction[mask]))
        scales.append(numerator / denominator)
    return edges, np.asarray(scales, dtype=np.float64)


def regularize_scales(
    raw_scales: np.ndarray,
    strength: float,
    floor: float,
    cap: float,
) -> np.ndarray:
    """strength=0 等价于不做条件 shrink；strength=1 使用完整桶内估计。"""
    scales = 1.0 + float(strength) * (raw_scales - 1.0)
    return np.clip(scales, float(floor), float(cap))


def apply_scales(
    prediction: np.ndarray,
    regime: np.ndarray,
    edges: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    buckets = bucket_ids(regime, edges)
    return prediction * scales[buckets], buckets


def score_by_abs_target_bucket(
    y_true: np.ndarray,
    base_prediction: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
) -> pd.DataFrame:
    abs_target = np.abs(y_true)
    edges = np.unique(np.quantile(abs_target, np.linspace(0.0, 1.0, 11)))
    buckets = np.digitize(abs_target, edges[1:-1], right=False)
    rows = []
    for bucket in range(len(edges) - 1):
        mask = buckets == bucket
        rows.append(
            {
                "bucket": int(bucket),
                "abs_target_min": float(edges[bucket]),
                "abs_target_max": float(edges[bucket + 1]),
                "rows": int(mask.sum()),
                "base_r2": float(
                    weighted_zero_mean_r2(y_true[mask], base_prediction[mask], weight[mask])
                ),
                "prediction_r2": float(
                    weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_bucket_scores(frame: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(frame))
    width = 0.38
    plt.figure(figsize=(10, 4.5))
    plt.bar(x - width / 2, frame["base_r2"], width=width, label="base")
    plt.bar(x + width / 2, frame["prediction_r2"], width=width, label="regularized")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, [f"Q{index + 1}" for index in range(len(frame))])
    plt.xlabel("|target| decile")
    plt.ylabel("weighted zero-mean R2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    use_columns = [
        "time_id",
        "asset_id",
        "target",
        "weight",
        "base_prediction",
        "market_prediction",
    ]
    frame = pd.read_csv(args.input_predictions, usecols=use_columns)
    frame = frame.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    y_true = frame["target"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    asset_id = frame["asset_id"].to_numpy(dtype=np.int64)
    time_id = frame["time_id"].to_numpy(dtype=np.int64)
    base_prediction = frame["base_prediction"].to_numpy(dtype=np.float64)
    regime = frame["market_prediction"].to_numpy(dtype=np.float64)

    unique_times = np.unique(time_id)
    split_time = int(unique_times[len(unique_times) // 2])
    first_mask = time_id < split_time
    second_mask = ~first_mask
    quantile_configs = {
        "q50_q80_q95": [0.0, 0.50, 0.80, 0.95, 1.0],
        "q80_q95": [0.0, 0.80, 0.95, 1.0],
        "q70": [0.0, 0.70, 1.0],
        "q75": [0.0, 0.75, 1.0],
        "q80": [0.0, 0.80, 1.0],
        "q85": [0.0, 0.85, 1.0],
        "q90": [0.0, 0.90, 1.0],
        "q50": [0.0, 0.50, 1.0],
    }

    rows: list[dict] = []
    payloads: dict[str, dict] = {}
    for beta in np.arange(args.beta_min, args.beta_max + 1e-12, args.beta_step):
        raw_prediction = base_prediction + float(beta) * regime
        shrink_info = calibrate_shrink_info(
            y_true,
            raw_prediction,
            weight,
            asset_id,
            "per_asset",
            float(args.shrink_cap),
        )
        plain_prediction = apply_shrink(raw_prediction, asset_id, shrink_info)
        plain_info = score_candidate_on_calibration(
            y_true,
            plain_prediction,
            weight,
            time_id,
            "full",
        )

        for config_name, quantiles in quantile_configs.items():
            # 完整 calibration 参数用于最终落盘；前半段参数只用于 forward 稳定性评价。
            full_edges, full_raw_scales = fit_raw_scales(
                y_true, plain_prediction, weight, regime, quantiles
            )
            forward_edges, forward_raw_scales = fit_raw_scales(
                y_true[first_mask],
                plain_prediction[first_mask],
                weight[first_mask],
                regime[first_mask],
                quantiles,
            )
            for strength in args.regularization_strengths:
                full_scales = regularize_scales(
                    full_raw_scales,
                    strength,
                    args.scale_floor,
                    args.scale_cap,
                )
                forward_scales = regularize_scales(
                    forward_raw_scales,
                    strength,
                    args.scale_floor,
                    args.scale_cap,
                )
                prediction, buckets = apply_scales(
                    plain_prediction, regime, full_edges, full_scales
                )
                forward_second, _ = apply_scales(
                    plain_prediction[second_mask],
                    regime[second_mask],
                    forward_edges,
                    forward_scales,
                )
                full_score = float(weighted_zero_mean_r2(y_true, prediction, weight))
                forward_second_score = float(
                    weighted_zero_mean_r2(
                        y_true[second_mask],
                        forward_second,
                        weight[second_mask],
                    )
                )
                robust_score = min(
                    float(plain_info["first_half_score"]),
                    forward_second_score,
                )
                key = f"{beta:.8f}|{config_name}|{float(strength):.8f}"
                rows.append(
                    {
                        "beta": float(beta),
                        "bucket_config": config_name,
                        "regularization_strength": float(strength),
                        "plain_full_score": float(plain_info["full_score"]),
                        "plain_first_half_score": float(plain_info["first_half_score"]),
                        "plain_second_half_score": float(plain_info["second_half_score"]),
                        "conditional_full_score": full_score,
                        "forward_second_score": forward_second_score,
                        "robust_score": robust_score,
                        "prediction_std": float(np.std(prediction)),
                        "full_scales": json.dumps(full_scales.tolist()),
                        "forward_scales": json.dumps(forward_scales.tolist()),
                    }
                )
                payloads[key] = {
                    "prediction": prediction,
                    "plain_prediction": plain_prediction,
                    "buckets": buckets,
                    "full_edges": full_edges,
                    "full_scales": full_scales,
                    "forward_edges": forward_edges,
                    "forward_scales": forward_scales,
                    "shrink_info": shrink_info,
                }

    metrics_frame = pd.DataFrame(rows)
    metrics_frame = metrics_frame.sort_values(
        ["robust_score", "conditional_full_score"], ascending=False
    ).reset_index(drop=True)
    best_row = metrics_frame.iloc[0]
    best_key = (
        f"{float(best_row['beta']):.8f}|{best_row['bucket_config']}|"
        f"{float(best_row['regularization_strength']):.8f}"
    )
    best = payloads[best_key]

    output = frame.copy()
    output["plain_prediction"] = best["plain_prediction"]
    output["conditional_bucket"] = best["buckets"]
    output["prediction"] = best["prediction"]
    output["error"] = output["prediction"] - output["target"]
    bucket_scores = score_by_abs_target_bucket(
        y_true,
        base_prediction,
        best["prediction"],
        weight,
    )

    metrics_frame.to_csv(args.results_dir / "candidate_metrics.csv", index=False)
    metrics_frame.head(100).to_csv(args.results_dir / "candidate_top100.csv", index=False)
    output.to_csv(args.results_dir / "calibration_predictions.csv", index=False)
    bucket_scores.to_csv(args.results_dir / "score_by_abs_target_bucket.csv", index=False)
    plot_bucket_scores(bucket_scores, args.results_dir / "score_by_abs_target_bucket.png")

    base_score = float(weighted_zero_mean_r2(y_true, base_prediction, weight))
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_predictions": str(args.input_predictions),
        "selection_rule": "max min(plain first-half, forward conditional second-half)",
        "base": {
            "full_score": base_score,
            "prediction_std": float(np.std(base_prediction)),
        },
        "best": {
            "beta": float(best_row["beta"]),
            "bucket_config": str(best_row["bucket_config"]),
            "regularization_strength": float(best_row["regularization_strength"]),
            "plain_full_score": float(best_row["plain_full_score"]),
            "plain_first_half_score": float(best_row["plain_first_half_score"]),
            "plain_second_half_score": float(best_row["plain_second_half_score"]),
            "conditional_full_score": float(best_row["conditional_full_score"]),
            "forward_second_score": float(best_row["forward_second_score"]),
            "robust_score": float(best_row["robust_score"]),
            "improvement_over_base": float(best_row["conditional_full_score"] - base_score),
            "full_edges": np.asarray(best["full_edges"]).tolist(),
            "full_scales": np.asarray(best["full_scales"]).tolist(),
            "forward_scales": np.asarray(best["forward_scales"]).tolist(),
            "prediction_std": float(np.std(best["prediction"])),
            **score_time_blocks(y_true, best["prediction"], weight, time_id, 4),
            **score_time_blocks(y_true, best["prediction"], weight, time_id, 8),
        },
        "output_files": {
            "candidate_metrics": str(args.results_dir / "candidate_metrics.csv"),
            "calibration_predictions": str(args.results_dir / "calibration_predictions.csv"),
            "score_by_abs_target_bucket": str(
                args.results_dir / "score_by_abs_target_bucket.csv"
            ),
            "score_plot": str(args.results_dir / "score_by_abs_target_bucket.png"),
        },
    }
    with (args.results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=json_default)
    print(json.dumps(result["best"], ensure_ascii=False, indent=2, default=json_default))
    print(f"Saved outputs to {args.results_dir}")


if __name__ == "__main__":
    main()
