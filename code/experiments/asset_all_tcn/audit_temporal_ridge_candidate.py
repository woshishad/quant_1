from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a strictly causal temporal Ridge residual over the current candidate."
    )
    parser.add_argument(
        "--outer-predictions",
        type=Path,
        default=Path("results/auxiliary_stacking_20260824/outer_predictions.csv"),
    )
    parser.add_argument(
        "--auxiliary-audit",
        type=Path,
        default=Path("results/auxiliary_stacking_audit_20260824/audit_report.json"),
    )
    parser.add_argument(
        "--temporal-predictions",
        type=Path,
        default=Path(
            "results/asset_all_residual_ridge_75k_temporal_history_probe_20260824/"
            "calibration_predictions.csv"
        ),
    )
    parser.add_argument(
        "--temporal-metrics",
        type=Path,
        default=Path(
            "results/asset_all_residual_ridge_75k_temporal_history_probe_20260824/"
            "metrics.json"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/temporal_ridge_candidate_audit_20260824"),
    )
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

    temporal_metrics = json.loads(
        args.temporal_metrics.read_text(encoding="utf-8")
    )
    feature_info = temporal_metrics["feature_engineering"]
    model_info = temporal_metrics["model"]
    protocol_checks = {
        "temporal_metrics_leakage_safe": temporal_metrics.get("leakage_safe") is True,
        "official_test_not_used_for_training": temporal_metrics.get(
            "official_test_used_for_training"
        )
        is False,
        "residual_model_is_ridge": model_info.get("residual_model") == "ridge",
        "residual_features_are_historical_only": model_info.get(
            "residual_feature_set"
        )
        == "historical",
        "asset_history_enabled": feature_info.get("engineered_base_feature_count", 0)
        > 0,
        "market_history_enabled": feature_info.get("market_history_feature_count", 0)
        > 0
        and feature_info.get("disable_market_history") is False,
    }
    if not all(protocol_checks.values()):
        failed = [name for name, passed in protocol_checks.items() if not passed]
        raise ValueError(f"temporal protocol checks failed: {failed}")

    outer_columns = [
        "row_id",
        "time_id",
        "asset_id",
        "target",
        "weight",
        "current_best_prediction",
        "prediction_refit_raw_only",
        "prediction_refit_predicted_aux_only",
    ]
    outer = pd.read_csv(args.outer_predictions, usecols=outer_columns)
    temporal = pd.read_csv(
        args.temporal_predictions, usecols=["row_id", "residual_prediction"]
    )
    frame = outer.merge(temporal, on="row_id", how="left", validate="one_to_one")
    if frame["residual_prediction"].isna().any():
        raise ValueError("temporal predictions do not cover all outer rows")

    auxiliary_audit = json.loads(
        args.auxiliary_audit.read_text(encoding="utf-8")
    )
    auxiliary_selection = auxiliary_audit["latest_external_current_best"][
        "current_best_plus_selected_refit_delta"
    ]
    auxiliary_gamma = float(auxiliary_selection["gamma_fit_first_half"])
    base_prediction = frame["current_best_prediction"].to_numpy(dtype=np.float64)
    base_prediction += auxiliary_gamma * (
        frame["prediction_refit_predicted_aux_only"].to_numpy(dtype=np.float64)
        - frame["prediction_refit_raw_only"].to_numpy(dtype=np.float64)
    )

    target = frame["target"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    time_id = frame["time_id"].to_numpy(dtype=np.int64)
    signal = frame["residual_prediction"].to_numpy(dtype=np.float64)
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

    gamma = fit_gamma(
        target[fit_mask],
        base_prediction[fit_mask],
        signal[fit_mask],
        weight[fit_mask],
        args.gamma_bound,
    )
    candidate_prediction = base_prediction + gamma * signal
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
        **protocol_checks,
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
    validation_predictions["temporal_residual_signal"] = signal
    validation_predictions["temporal_gamma"] = gamma
    validation_predictions["prediction"] = candidate_prediction
    prediction_path = args.results_dir / "validation_predictions.csv"
    validation_predictions.to_csv(prediction_path, index=False)

    report = {
        "strategy": "current_candidate_plus_strict_causal_temporal_ridge_residual",
        "leakage_safe": True,
        "official_test_used": False,
        "formula": "prediction = current_candidate + temporal_gamma * temporal_ridge_residual",
        "temporal_gamma": gamma,
        "protocol": {
            "coefficient_fit": [
                args.coefficient_fit_start,
                args.coefficient_fit_end - 1,
            ],
            "calibration": [args.calibration_start, args.calibration_end - 1],
            "holdout": [args.holdout_start, args.holdout_end - 1],
            "minimum_purge_gap": args.minimum_purge_gap,
        },
        "temporal_feature_engineering": feature_info,
        "temporal_model": model_info,
        "scores": scores,
        "holdout_blocks": block_rows,
        "holdout_positive_blocks": positive_blocks,
        "holdout_block_count": args.holdout_blocks,
        "promotion_checks": promotion_checks,
        "all_local_promotion_checks_passed": all_checks_passed,
        "status": (
            "local_temporal_candidate_passed_small_gain_pending_online_validation"
            if all_checks_passed
            else "temporal_candidate_rejected"
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
