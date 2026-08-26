from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FOLDS = {
    "fold0": Path("results/auxiliary_stacking_fold0_20260824"),
    "fold1": Path("results/auxiliary_stacking_fold1_refit_20260824"),
    "fold2": Path("results/auxiliary_stacking_fold2_refit_20260824"),
    "latest": Path("results/auxiliary_stacking_20260824"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit leakage-safe refit predictions from auxiliary stacking experiments."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/auxiliary_stacking_audit_20260824"),
    )
    parser.add_argument(
        "--fold",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override default fold inputs. May be repeated.",
    )
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--gamma-bound", type=float, default=2.0)
    parser.add_argument(
        "--reference-audit",
        type=Path,
        help="Optional incumbent audit whose latest delta metrics must be exceeded.",
    )
    return parser.parse_args()


def parse_folds(values: list[str]) -> dict[str, Path]:
    if not values:
        return DEFAULT_FOLDS.copy()
    folds: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", maxsplit=1)
        folds[name] = Path(raw_path)
    return folds


def weighted_zero_mean_r2(
    target: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    denominator = float(np.sum(weight * target * target))
    if denominator <= 1e-18:
        return 0.0
    return 1.0 - float(np.sum(weight * (target - prediction) ** 2)) / denominator


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


def segment_scores(
    target: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    split_time: int,
) -> dict[str, float]:
    masks = {
        "full": np.ones(len(target), dtype=bool),
        "fit_first_half": time_id < split_time,
        "holdout_second_half": time_id >= split_time,
    }
    result: dict[str, float] = {}
    for name, mask in masks.items():
        base_score = weighted_zero_mean_r2(target[mask], base[mask], weight[mask])
        candidate_score = weighted_zero_mean_r2(
            target[mask], candidate[mask], weight[mask]
        )
        result[f"base_{name}"] = base_score
        result[f"candidate_{name}"] = candidate_score
        result[f"gain_{name}"] = candidate_score - base_score
    return result


def evaluate_signal(
    target: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    base: np.ndarray,
    signal: np.ndarray,
    split_time: int,
    gamma_bound: float,
) -> tuple[dict[str, float], np.ndarray]:
    first = time_id < split_time
    second = ~first
    gamma = fit_gamma(
        target[first], base[first], signal[first], weight[first], gamma_bound
    )
    holdout_optimal_gamma = fit_gamma(
        target[second], base[second], signal[second], weight[second], gamma_bound
    )
    candidate = base + gamma * signal
    metrics = {
        "gamma_fit_first_half": gamma,
        "gamma_optimal_second_half_diagnostic": holdout_optimal_gamma,
        **segment_scores(target, weight, time_id, base, candidate, split_time),
    }
    return metrics, candidate


def score_slices(
    fold: str,
    signal_name: str,
    frame: pd.DataFrame,
    base: np.ndarray,
    candidate: np.ndarray,
    blocks: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    target = frame["target"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    time_id = frame["time_id"].to_numpy(dtype=np.int64)
    asset_id = frame["asset_id"].to_numpy(dtype=np.int64)

    def row_for_mask(group_type: str, group: str | int, mask: np.ndarray) -> dict:
        base_score = weighted_zero_mean_r2(target[mask], base[mask], weight[mask])
        candidate_score = weighted_zero_mean_r2(
            target[mask], candidate[mask], weight[mask]
        )
        return {
            "fold": fold,
            "signal": signal_name,
            "group_type": group_type,
            "group": group,
            "rows": int(mask.sum()),
            "base_score": base_score,
            "candidate_score": candidate_score,
            "gain": candidate_score - base_score,
        }

    block_rows: list[dict] = []
    unique_times = np.unique(time_id)
    for block, times in enumerate(np.array_split(unique_times, blocks)):
        mask = (time_id >= times[0]) & (time_id <= times[-1])
        row = row_for_mask("time_block", block, mask)
        row["time_min"] = int(times[0])
        row["time_max"] = int(times[-1])
        block_rows.append(row)

    asset_rows = [
        row_for_mask("asset", int(asset), asset_id == asset)
        for asset in np.unique(asset_id)
    ]

    magnitude_bucket = np.asarray(
        pd.qcut(np.abs(target), q=5, labels=False, duplicates="drop")
    )
    magnitude_rows = [
        row_for_mask("abs_target_quintile", int(bucket), magnitude_bucket == bucket)
        for bucket in np.unique(magnitude_bucket)
    ]

    weight_bucket = np.asarray(
        pd.qcut(weight, q=5, labels=False, duplicates="drop")
    )
    weight_rows = [
        row_for_mask("weight_quintile", int(bucket), weight_bucket == bucket)
        for bucket in np.unique(weight_bucket)
    ]
    return block_rows, asset_rows, magnitude_rows, weight_rows


def load_fold(path: Path) -> tuple[dict, pd.DataFrame, str]:
    summary_path = path / "summary.json"
    prediction_path = path / "outer_predictions.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = summary["selected_aux_variant_on_internal_calibration"]
    selected_column = f"prediction_refit_{selected}"
    usecols = [
        "row_id",
        "time_id",
        "asset_id",
        "target",
        "weight",
        "prediction_refit_raw_only",
        selected_column,
    ]
    header = pd.read_csv(prediction_path, nrows=0)
    if "current_best_prediction" in header.columns:
        usecols.append("current_best_prediction")
    frame = pd.read_csv(prediction_path, usecols=usecols)
    if not np.isfinite(frame[usecols].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"non-finite or missing values in {prediction_path}")
    return summary, frame, selected


def main() -> None:
    args = parse_args()
    folds = parse_folds(args.fold)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict] = []
    block_rows: list[dict] = []
    asset_rows: list[dict] = []
    magnitude_rows: list[dict] = []
    weight_rows: list[dict] = []
    variant_rows: list[dict] = []
    latest_external: dict[str, dict] = {}

    for fold, path in folds.items():
        print(f"auditing {fold}: {path}", flush=True)
        summary, frame, selected = load_fold(path)
        for variant, metrics in summary["target_variants"].items():
            refit_metrics = summary["target_refit_variants"][variant]
            variant_rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "selected_on_internal_calibration": variant == selected,
                    "feature_count": int(metrics["feature_count"]),
                    "calibration_selection_score": float(
                        metrics["calibration_selection"]["selection_score"]
                    ),
                    "calibration_full_score": float(
                        metrics["calibration_selection"]["full_score"]
                    ),
                    "refit_outer_full_score": float(
                        refit_metrics["outer_scores"]["full"]
                    ),
                    "refit_outer_holdout_score": float(
                        refit_metrics["outer_scores"]["second_half"]
                    ),
                }
            )
        target = frame["target"].to_numpy(dtype=np.float64)
        weight = frame["weight"].to_numpy(dtype=np.float64)
        time_id = frame["time_id"].to_numpy(dtype=np.int64)
        base = frame["prediction_refit_raw_only"].to_numpy(dtype=np.float64)
        selected_prediction = frame[f"prediction_refit_{selected}"].to_numpy(
            dtype=np.float64
        )
        split_time = int(summary["target_refit_variants"]["raw_only"]["outer_scores"]["split_time"])

        direct_metrics = segment_scores(
            target, weight, time_id, base, selected_prediction, split_time
        )
        direct_metrics.update(
            {
                "fold": fold,
                "selected_variant": selected,
                "signal": "direct_selected_refit",
                "gamma_fit_first_half": 1.0,
                "gamma_optimal_second_half_diagnostic": np.nan,
            }
        )
        fold_rows.append(direct_metrics)

        signals = {
            "selected_refit_absolute": selected_prediction,
            "selected_refit_delta": selected_prediction - base,
        }
        for signal_name, signal in signals.items():
            metrics, candidate = evaluate_signal(
                target,
                weight,
                time_id,
                base,
                signal,
                split_time,
                args.gamma_bound,
            )
            metrics.update(
                {
                    "fold": fold,
                    "selected_variant": selected,
                    "signal": signal_name,
                }
            )
            fold_rows.append(metrics)
            slices = score_slices(
                fold, signal_name, frame, base, candidate, args.blocks
            )
            block_rows.extend(slices[0])
            asset_rows.extend(slices[1])
            magnitude_rows.extend(slices[2])
            weight_rows.extend(slices[3])

        if "current_best_prediction" in frame.columns:
            current_best = frame["current_best_prediction"].to_numpy(dtype=np.float64)
            for signal_name, signal in signals.items():
                external_name = f"current_best_plus_{signal_name}"
                metrics, candidate = evaluate_signal(
                    target,
                    weight,
                    time_id,
                    current_best,
                    signal,
                    split_time,
                    args.gamma_bound,
                )
                latest_external[external_name] = {
                    "selected_variant": selected,
                    **metrics,
                }
                slices = score_slices(
                    fold, external_name, frame, current_best, candidate, args.blocks
                )
                block_rows.extend(slices[0])
                asset_rows.extend(slices[1])
                magnitude_rows.extend(slices[2])
                weight_rows.extend(slices[3])

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(args.results_dir / "fold_summary.csv", index=False)
    pd.DataFrame(block_rows).to_csv(args.results_dir / "block_audit.csv", index=False)
    pd.DataFrame(asset_rows).to_csv(args.results_dir / "asset_audit.csv", index=False)
    pd.DataFrame(magnitude_rows).to_csv(
        args.results_dir / "target_magnitude_audit.csv", index=False
    )
    pd.DataFrame(weight_rows).to_csv(
        args.results_dir / "weight_bucket_audit.csv", index=False
    )
    variant_frame = pd.DataFrame(variant_rows)
    variant_frame.to_csv(args.results_dir / "variant_audit.csv", index=False)

    historical = fold_frame[fold_frame["fold"] != "latest"]
    historical_summary: dict[str, dict] = {}
    for signal, group in historical.groupby("signal", sort=False):
        historical_summary[signal] = {
            "folds": int(len(group)),
            "positive_full_gains": int((group["gain_full"] > 0).sum()),
            "positive_holdout_gains": int((group["gain_holdout_second_half"] > 0).sum()),
            "mean_full_gain": float(group["gain_full"].mean()),
            "minimum_full_gain": float(group["gain_full"].min()),
            "mean_holdout_gain": float(group["gain_holdout_second_half"].mean()),
            "minimum_holdout_gain": float(group["gain_holdout_second_half"].min()),
        }

    external_abs = latest_external.get("current_best_plus_selected_refit_absolute", {})
    external_delta_name = "current_best_plus_selected_refit_delta"
    external_delta_blocks = [
        row
        for row in block_rows
        if row["fold"] == "latest" and row["signal"] == external_delta_name
    ]
    external_delta_assets = [
        row
        for row in asset_rows
        if row["fold"] == "latest" and row["signal"] == external_delta_name
    ]
    external_delta_magnitude = [
        row
        for row in magnitude_rows
        if row["fold"] == "latest" and row["signal"] == external_delta_name
    ]
    positive_blocks = sum(row["gain"] > 0 for row in external_delta_blocks)
    positive_assets = sum(row["gain"] > 0 for row in external_delta_assets)
    positive_magnitude_buckets = sum(
        row["gain"] > 0 for row in external_delta_magnitude
    )
    variant_selection_counts = {
        str(variant): int(group["selected_on_internal_calibration"].sum())
        for variant, group in variant_frame.groupby("variant", sort=True)
    }
    promotion_checks = {
        "historical_direct_positive_all_folds": historical_summary.get(
            "direct_selected_refit", {}
        ).get("positive_full_gains")
        == 3,
        "latest_external_first_half_gain_positive": external_abs.get(
            "gain_fit_first_half", 0.0
        )
        > 0,
        "latest_external_frozen_holdout_gain_positive": external_abs.get(
            "gain_holdout_second_half", 0.0
        )
        > 0,
        "latest_external_gamma_direction_stable": (
            external_abs.get("gamma_fit_first_half", 0.0)
            * external_abs.get("gamma_optimal_second_half_diagnostic", 0.0)
            > 0
        ),
        "latest_external_positive_time_block_majority": positive_blocks
        > len(external_delta_blocks) / 2,
        "latest_external_positive_asset_majority": positive_assets
        > len(external_delta_assets) / 2,
        "latest_external_all_target_magnitude_buckets_positive": (
            positive_magnitude_buckets == len(external_delta_magnitude)
        ),
    }
    incumbent_comparison: dict[str, float | int | str] = {}
    if args.reference_audit is not None:
        reference_report = json.loads(args.reference_audit.read_text(encoding="utf-8"))
        reference_delta = reference_report["latest_external_current_best"][
            "current_best_plus_selected_refit_delta"
        ]
        reference_slices = reference_report["latest_external_slice_counts"]
        current_delta = latest_external[external_delta_name]
        incumbent_comparison = {
            "reference_audit": str(args.reference_audit),
            "reference_full_gain": float(reference_delta["gain_full"]),
            "candidate_full_gain": float(current_delta["gain_full"]),
            "reference_holdout_gain": float(
                reference_delta["gain_holdout_second_half"]
            ),
            "candidate_holdout_gain": float(
                current_delta["gain_holdout_second_half"]
            ),
            "reference_positive_time_blocks": int(
                reference_slices["positive_time_blocks"]
            ),
            "candidate_positive_time_blocks": positive_blocks,
        }
        promotion_checks.update(
            {
                "latest_external_full_gain_exceeds_incumbent": (
                    current_delta["gain_full"] > reference_delta["gain_full"]
                ),
                "latest_external_holdout_gain_exceeds_incumbent": (
                    current_delta["gain_holdout_second_half"]
                    > reference_delta["gain_holdout_second_half"]
                ),
                "latest_external_positive_time_blocks_exceed_incumbent": (
                    positive_blocks > reference_slices["positive_time_blocks"]
                ),
            }
        )
    report = {
        "official_test_used": False,
        "fold_inputs": {name: str(path) for name, path in folds.items()},
        "historical_summary": historical_summary,
        "variant_selection_counts": variant_selection_counts,
        "latest_external_current_best": latest_external,
        "latest_external_slice_counts": {
            "positive_time_blocks": positive_blocks,
            "total_time_blocks": len(external_delta_blocks),
            "positive_assets": positive_assets,
            "total_assets": len(external_delta_assets),
            "positive_target_magnitude_buckets": positive_magnitude_buckets,
            "total_target_magnitude_buckets": len(external_delta_magnitude),
        },
        "incumbent_comparison": incumbent_comparison,
        "promotion_checks": promotion_checks,
        "all_promotion_checks_passed": all(promotion_checks.values()),
        "interpretation": (
            "Auxiliary predictions are generated from feature and asset_id only. "
            "Gamma is fit on each outer first half and frozen before scoring the second half."
        ),
    }
    (args.results_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
