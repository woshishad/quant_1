from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a candidate signal coefficient on an early interval and audit the "
            "frozen coefficient against the current validation baseline."
        )
    )
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=Path(
            "results/temporal_ridge_candidate_audit_20260824/"
            "validation_predictions.csv"
        ),
    )
    parser.add_argument("--base-column", default="prediction")
    parser.add_argument("--signal-predictions", type=Path, required=True)
    parser.add_argument("--signal-column", default="residual_prediction")
    parser.add_argument("--signal-metrics", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", default="external_signal_candidate")
    parser.add_argument("--coefficient-fit-start", type=int, default=868_480)
    parser.add_argument("--coefficient-fit-end", type=int, default=872_480)
    parser.add_argument("--calibration-start", type=int, default=873_480)
    parser.add_argument("--calibration-end", type=int, default=877_480)
    parser.add_argument("--holdout-start", type=int, default=878_480)
    parser.add_argument("--holdout-end", type=int, default=888_480)
    parser.add_argument("--minimum-purge-gap", type=int, default=1_000)
    parser.add_argument("--gamma-bound", type=float, default=2.0)
    parser.add_argument("--holdout-blocks", type=int, default=8)
    parser.add_argument("--minimum-positive-blocks", type=int, default=5)
    return parser.parse_args()


def weighted_zero_mean_r2(
    target: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> float:
    denominator = float(np.sum(weight * target * target))
    if denominator <= 1e-18:
        return 0.0
    error = float(np.sum(weight * (target - prediction) ** 2))
    return 1.0 - error / denominator


def fit_gamma(
    target: np.ndarray,
    base: np.ndarray,
    signal: np.ndarray,
    weight: np.ndarray,
    bound: float,
) -> float:
    denominator = float(np.sum(weight * signal * signal))
    if denominator <= 1e-18:
        return 0.0
    gamma = float(np.sum(weight * signal * (target - base)) / denominator)
    return float(np.clip(gamma, -bound, bound))


def score_gain(
    target: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    base_score = weighted_zero_mean_r2(target[mask], base[mask], weight[mask])
    candidate_score = weighted_zero_mean_r2(
        target[mask], candidate[mask], weight[mask]
    )
    return {
        "rows": int(mask.sum()),
        "base_score": base_score,
        "candidate_score": candidate_score,
        "gain": candidate_score - base_score,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol(args: argparse.Namespace) -> None:
    ranges = [
        ("coefficient_fit", args.coefficient_fit_start, args.coefficient_fit_end),
        ("calibration", args.calibration_start, args.calibration_end),
        ("holdout", args.holdout_start, args.holdout_end),
    ]
    for name, start, end in ranges:
        if start >= end:
            raise ValueError(f"invalid {name} range [{start}, {end})")
    for (_, _, previous_end), (name, next_start, _) in zip(ranges, ranges[1:]):
        if next_start - previous_end < args.minimum_purge_gap:
            raise ValueError(f"purge gap before {name} is too short")


def main() -> None:
    args = parse_args()
    validate_protocol(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    base_columns = [
        "row_id",
        "time_id",
        "asset_id",
        "target",
        "weight",
        args.base_column,
    ]
    base = pd.read_csv(args.base_predictions, usecols=base_columns)
    signal = pd.read_csv(
        args.signal_predictions, usecols=["row_id", args.signal_column]
    )
    if base["row_id"].duplicated().any() or signal["row_id"].duplicated().any():
        raise ValueError("row_id must be unique in base and signal predictions")
    if len(base) != len(signal) or set(base["row_id"]) != set(signal["row_id"]):
        raise ValueError("base and signal row_id sets must match exactly")

    frame = base.merge(signal, on="row_id", how="left", validate="one_to_one")
    numeric_columns = [
        "time_id",
        "asset_id",
        "target",
        "weight",
        args.base_column,
        args.signal_column,
    ]
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("base or signal predictions contain non-finite values")

    signal_metrics = json.loads(args.signal_metrics.read_text(encoding="utf-8"))
    metrics_rows = signal_metrics.get("rows", {})
    future_guard = signal_metrics.get("future_function_guard", {})
    source_checks = {
        "signal_metrics_leakage_safe": signal_metrics.get("leakage_safe") is True,
        "official_test_not_used_for_training": signal_metrics.get(
            "official_test_used_for_training"
        )
        is False,
        "signal_generated_without_test_prediction": metrics_rows.get(
            "test_predicted"
        )
        is None
        and future_guard.get("official_test_prediction_only") is None,
        "row_alignment_exact": len(frame) == len(base),
    }
    if not all(source_checks.values()):
        failed = [name for name, passed in source_checks.items() if not passed]
        raise ValueError(f"candidate source checks failed: {failed}")

    target = frame["target"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    time_id = frame["time_id"].to_numpy(dtype=np.int64)
    base_prediction = frame[args.base_column].to_numpy(dtype=np.float64)
    candidate_signal = frame[args.signal_column].to_numpy(dtype=np.float64)

    fit_mask = (time_id >= args.coefficient_fit_start) & (
        time_id < args.coefficient_fit_end
    )
    calibration_mask = (time_id >= args.calibration_start) & (
        time_id < args.calibration_end
    )
    holdout_mask = (time_id >= args.holdout_start) & (
        time_id < args.holdout_end
    )
    full_mask = np.ones(len(frame), dtype=bool)
    for name, mask in {
        "coefficient_fit": fit_mask,
        "calibration": calibration_mask,
        "holdout": holdout_mask,
    }.items():
        if not np.any(mask):
            raise ValueError(f"{name} interval has no rows")

    gamma = fit_gamma(
        target[fit_mask],
        base_prediction[fit_mask],
        candidate_signal[fit_mask],
        weight[fit_mask],
        args.gamma_bound,
    )
    calibration_optimal_gamma = fit_gamma(
        target[calibration_mask],
        base_prediction[calibration_mask],
        candidate_signal[calibration_mask],
        weight[calibration_mask],
        args.gamma_bound,
    )
    holdout_optimal_gamma = fit_gamma(
        target[holdout_mask],
        base_prediction[holdout_mask],
        candidate_signal[holdout_mask],
        weight[holdout_mask],
        args.gamma_bound,
    )
    candidate_prediction = base_prediction + gamma * candidate_signal
    scores = {
        "coefficient_fit": score_gain(
            target, base_prediction, candidate_prediction, weight, fit_mask
        ),
        "calibration": score_gain(
            target, base_prediction, candidate_prediction, weight, calibration_mask
        ),
        "holdout": score_gain(
            target, base_prediction, candidate_prediction, weight, holdout_mask
        ),
        "full_outer": score_gain(
            target, base_prediction, candidate_prediction, weight, full_mask
        ),
    }

    block_rows = []
    holdout_times = np.unique(time_id[holdout_mask])
    for block, times in enumerate(np.array_split(holdout_times, args.holdout_blocks)):
        if len(times) == 0:
            continue
        mask = np.isin(time_id, times)
        block_rows.append(
            {
                "block": block,
                "time_min": int(times[0]),
                "time_max": int(times[-1]),
                **score_gain(
                    target, base_prediction, candidate_prediction, weight, mask
                ),
            }
        )
    positive_blocks = sum(float(row["gain"]) > 0.0 for row in block_rows)
    promotion_checks = {
        **source_checks,
        "gamma_is_finite_and_not_at_bound": np.isfinite(gamma)
        and abs(gamma) < args.gamma_bound,
        "coefficient_fit_gain_positive": scores["coefficient_fit"]["gain"] > 0.0,
        "calibration_gain_positive": scores["calibration"]["gain"] > 0.0,
        "holdout_gain_positive": scores["holdout"]["gain"] > 0.0,
        "holdout_positive_block_requirement": positive_blocks
        >= args.minimum_positive_blocks,
    }
    all_checks_passed = all(promotion_checks.values())

    validation_predictions = frame[
        ["row_id", "time_id", "asset_id", "target", "weight"]
    ].copy()
    validation_predictions["base_prediction"] = base_prediction
    validation_predictions["candidate_signal"] = candidate_signal
    validation_predictions["frozen_gamma"] = gamma
    validation_predictions["prediction"] = candidate_prediction
    prediction_path = args.results_dir / "validation_predictions.csv"
    validation_predictions.to_csv(prediction_path, index=False)

    report = {
        "candidate_name": args.candidate_name,
        "strategy": "current_validated_baseline_plus_frozen_external_signal",
        "leakage_safe": True,
        "official_test_used": False,
        "base_predictions": str(args.base_predictions),
        "base_column": args.base_column,
        "signal_predictions": str(args.signal_predictions),
        "signal_column": args.signal_column,
        "formula": "prediction = base_prediction + frozen_gamma * candidate_signal",
        "frozen_gamma": gamma,
        "diagnostic_optimal_gamma": {
            "calibration": calibration_optimal_gamma,
            "holdout": holdout_optimal_gamma,
        },
        "protocol": {
            "coefficient_fit": [
                args.coefficient_fit_start,
                args.coefficient_fit_end - 1,
            ],
            "calibration": [args.calibration_start, args.calibration_end - 1],
            "holdout": [args.holdout_start, args.holdout_end - 1],
            "minimum_purge_gap": args.minimum_purge_gap,
        },
        "signal_feature_engineering": signal_metrics.get("feature_engineering"),
        "signal_model": signal_metrics.get("model"),
        "scores": scores,
        "holdout_blocks": block_rows,
        "holdout_positive_blocks": positive_blocks,
        "holdout_block_count": len(block_rows),
        "promotion_checks": promotion_checks,
        "all_local_promotion_checks_passed": all_checks_passed,
        "status": (
            "candidate_passed_local_checks_pending_online_validation"
            if all_checks_passed
            else "candidate_rejected"
        ),
        "outputs": {
            "validation_predictions": str(prediction_path),
            "validation_predictions_sha256": sha256_file(prediction_path),
        },
    }
    report_path = args.results_dir / "audit_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
