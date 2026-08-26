from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_auxiliary_stacking import (
    AuxiliaryMLP,
    normalize_aux_inputs,
    parquet_time_bounds,
    predict_auxiliary_mlp,
    read_time_range,
)


DEFAULT_FOLDS = {
    "fold0": Path("results/auxiliary_stacking_fold0_20260824"),
    "fold1": Path("results/auxiliary_stacking_fold1_refit_20260824"),
    "fold2": Path("results/auxiliary_stacking_fold2_refit_20260824"),
    "latest": Path("results/auxiliary_stacking_20260824"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune a leakage-safe gate from feature-predicted weight/liquidity state."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/predicted_weight_gate_20260824"),
    )
    parser.add_argument("--predict-batch-size", type=int, default=16_384)
    parser.add_argument("--gamma-bound", type=float, default=2.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def weighted_zero_mean_r2(
    target: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> float:
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


def score_gain(
    target: np.ndarray,
    weight: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float, float]:
    base_score = weighted_zero_mean_r2(target, base, weight)
    candidate_score = weighted_zero_mean_r2(target, candidate, weight)
    return base_score, candidate_score, candidate_score - base_score


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def make_gate(
    state_z: np.ndarray,
    fit_mask: np.ndarray,
    gate_kind: str,
    parameter: float,
    temperature: float,
    floor: float,
) -> np.ndarray:
    if gate_kind == "constant":
        gate = np.ones(len(state_z), dtype=np.float64)
    elif gate_kind == "linear":
        gate = np.clip(1.0 + parameter * state_z, 0.05, 2.0)
    elif gate_kind == "sigmoid":
        gate = floor + (1.0 - floor) * sigmoid(
            (state_z - parameter) / temperature
        )
    else:
        raise ValueError(gate_kind)
    fit_mean = float(np.mean(gate[fit_mask]))
    if fit_mean <= 1e-12:
        return np.ones(len(state_z), dtype=np.float64)
    return gate / fit_mean


def gate_candidates() -> list[dict[str, float | str]]:
    candidates: list[dict[str, float | str]] = [
        {
            "gate_kind": "constant",
            "parameter": 0.0,
            "temperature": 1.0,
            "floor": 1.0,
        }
    ]
    for slope in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
        candidates.append(
            {
                "gate_kind": "linear",
                "parameter": slope,
                "temperature": 1.0,
                "floor": 0.0,
            }
        )
    for center in (-0.5, 0.0, 0.5):
        for temperature in (0.5, 1.0):
            for floor in (0.0, 0.25, 0.5):
                candidates.append(
                    {
                        "gate_kind": "sigmoid",
                        "parameter": center,
                        "temperature": temperature,
                        "floor": floor,
                    }
                )
    return candidates


def load_auxiliary_states(
    fold: str,
    fold_dir: Path,
    paths: list[Path],
    bounds: dict[Path, tuple[int, int]],
    predict_batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
    start, inclusive_end = summary["protocol"]["outer_validation"]
    checkpoint = torch.load(
        fold_dir / "auxiliary_mlp.pt",
        map_location="cpu",
        weights_only=False,
    )
    feature_names = list(checkpoint["feature_names"])
    auxiliary_names = list(checkpoint["auxiliary_names"])
    auxiliary_index = {name: index for index, name in enumerate(auxiliary_names)}
    columns = ["row_id", "time_id", "asset_id", *feature_names]
    print(f"{fold}: reading outer features [{start}, {inclusive_end + 1})", flush=True)
    frame = read_time_range(
        paths, bounds, columns, int(start), int(inclusive_end) + 1
    )
    normalizers = np.load(fold_dir / "auxiliary_normalizers.npz")
    auxiliary_x = normalize_aux_inputs(
        frame[feature_names].to_numpy(dtype=np.float32, copy=True),
        frame["asset_id"].to_numpy(dtype=np.int64, copy=False),
        normalizers["input_mean"],
        normalizers["input_scale"],
        int(checkpoint["input_dim"]) - len(feature_names),
    )
    with (fold_dir / "auxiliary_ridge.pkl").open("rb") as handle:
        auxiliary_ridge = pickle.load(handle)
    ridge_prediction = np.asarray(
        auxiliary_ridge.predict(auxiliary_x), dtype=np.float32
    )
    auxiliary_mlp = AuxiliaryMLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    auxiliary_mlp.load_state_dict(checkpoint["state_dict"])
    auxiliary_mlp.to(device).eval()
    mlp_prediction = predict_auxiliary_mlp(
        auxiliary_mlp, auxiliary_x, predict_batch_size, device
    )

    weight_index = auxiliary_index["weight"]
    liquidity_indices = [
        auxiliary_index[f"responder_{index:02d}"] for index in range(31, 38)
    ]
    states = pd.DataFrame(
        {
            "row_id": frame["row_id"].to_numpy(dtype=np.int64),
            "ridge_weight": ridge_prediction[:, weight_index],
            "mlp_weight": mlp_prediction[:, weight_index],
            "mean_weight": 0.5
            * (
                ridge_prediction[:, weight_index]
                + mlp_prediction[:, weight_index]
            ),
            "ridge_liquidity": -np.mean(
                ridge_prediction[:, liquidity_indices], axis=1
            ),
            "mlp_liquidity": -np.mean(
                mlp_prediction[:, liquidity_indices], axis=1
            ),
            "mean_liquidity": -0.5
            * (
                np.mean(ridge_prediction[:, liquidity_indices], axis=1)
                + np.mean(mlp_prediction[:, liquidity_indices], axis=1)
            ),
        }
    )
    del frame, auxiliary_x, ridge_prediction, mlp_prediction, auxiliary_mlp
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return states


def evaluate_slices(
    fold: str,
    name: str,
    frame: pd.DataFrame,
    base: np.ndarray,
    candidate: np.ndarray,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    target = frame["target"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    time_id = frame["time_id"].to_numpy(dtype=np.int64)
    asset_id = frame["asset_id"].to_numpy(dtype=np.int64)

    def make_row(group_type: str, group: int, mask: np.ndarray) -> dict:
        base_score, candidate_score, gain = score_gain(
            target[mask], weight[mask], base[mask], candidate[mask]
        )
        return {
            "fold": fold,
            "candidate": name,
            "group_type": group_type,
            "group": group,
            "rows": int(mask.sum()),
            "base_score": base_score,
            "candidate_score": candidate_score,
            "gain": gain,
        }

    block_rows: list[dict] = []
    for block, times in enumerate(np.array_split(np.unique(time_id), 8)):
        mask = (time_id >= times[0]) & (time_id <= times[-1])
        row = make_row("time_block", block, mask)
        row["time_min"] = int(times[0])
        row["time_max"] = int(times[-1])
        block_rows.append(row)
    asset_rows = [
        make_row("asset", int(asset), asset_id == asset)
        for asset in np.unique(asset_id)
    ]
    target_bucket = np.asarray(
        pd.qcut(np.abs(target), q=5, labels=False, duplicates="drop")
    )
    target_rows = [
        make_row("abs_target_quintile", int(bucket), target_bucket == bucket)
        for bucket in np.unique(target_bucket)
    ]
    weight_bucket = np.asarray(
        pd.qcut(weight, q=5, labels=False, duplicates="drop")
    )
    weight_rows = [
        make_row("weight_quintile", int(bucket), weight_bucket == bucket)
        for bucket in np.unique(weight_bucket)
    ]
    return block_rows, asset_rows, target_rows, weight_rows


def tune_fold(
    fold: str,
    fold_dir: Path,
    states: pd.DataFrame,
    gamma_bound: float,
) -> tuple[dict, pd.DataFrame, dict[str, list[dict]]]:
    summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
    selected_variant = summary["selected_aux_variant_on_internal_calibration"]
    prediction_columns = [
        "row_id",
        "time_id",
        "asset_id",
        "target",
        "weight",
        "prediction_refit_raw_only",
        f"prediction_refit_{selected_variant}",
    ]
    header = pd.read_csv(fold_dir / "outer_predictions.csv", nrows=0)
    if "current_best_prediction" in header.columns:
        prediction_columns.append("current_best_prediction")
    frame = pd.read_csv(
        fold_dir / "outer_predictions.csv", usecols=prediction_columns
    )
    frame = frame.merge(states, on="row_id", how="left", validate="one_to_one")
    if frame.isna().any().any():
        raise ValueError(f"missing state rows for {fold}")

    target = frame["target"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    time_id = frame["time_id"].to_numpy(dtype=np.int64)
    raw_control = frame["prediction_refit_raw_only"].to_numpy(dtype=np.float64)
    selected_prediction = frame[f"prediction_refit_{selected_variant}"].to_numpy(
        dtype=np.float64
    )
    signal = selected_prediction - raw_control
    if "current_best_prediction" in frame.columns:
        base = frame["current_best_prediction"].to_numpy(dtype=np.float64)
    else:
        base = raw_control

    unique_times = np.unique(time_id)
    half_time = int(unique_times[len(unique_times) // 2])
    first_mask = time_id < half_time
    holdout_mask = ~first_mask
    first_times = np.unique(time_id[first_mask])
    quarter_time = int(first_times[len(first_times) // 2])
    q1 = time_id < quarter_time
    q2 = first_mask & ~q1

    grid_rows: list[dict] = []
    state_columns = [
        "ridge_weight",
        "mlp_weight",
        "mean_weight",
        "ridge_liquidity",
        "mlp_liquidity",
        "mean_liquidity",
    ]
    for state_name in state_columns:
        state = frame[state_name].to_numpy(dtype=np.float64)
        state_mean = float(np.mean(state[first_mask]))
        state_scale = float(np.std(state[first_mask]))
        if not math.isfinite(state_scale) or state_scale < 1e-8:
            state_scale = 1.0
        state_z = np.clip((state - state_mean) / state_scale, -6.0, 6.0)
        for specification in gate_candidates():
            gate = make_gate(state_z, first_mask, **specification)
            gated_signal = signal * gate
            gamma_q1 = fit_gamma(
                target[q1], base[q1], gated_signal[q1], weight[q1], gamma_bound
            )
            _, _, gain_q2 = score_gain(
                target[q2],
                weight[q2],
                base[q2],
                base[q2] + gamma_q1 * gated_signal[q2],
            )
            gamma_q2 = fit_gamma(
                target[q2], base[q2], gated_signal[q2], weight[q2], gamma_bound
            )
            _, _, gain_q1 = score_gain(
                target[q1],
                weight[q1],
                base[q1],
                base[q1] + gamma_q2 * gated_signal[q1],
            )
            grid_rows.append(
                {
                    "fold": fold,
                    "state": state_name,
                    **specification,
                    "state_mean": state_mean,
                    "state_scale": state_scale,
                    "gamma_q1": gamma_q1,
                    "validation_gain_q2": gain_q2,
                    "gamma_q2": gamma_q2,
                    "validation_gain_q1": gain_q1,
                    "selection_min_gain": min(gain_q1, gain_q2),
                    "selection_mean_gain": 0.5 * (gain_q1 + gain_q2),
                }
            )
    grid = pd.DataFrame(grid_rows)
    grid = grid.sort_values(
        ["selection_min_gain", "selection_mean_gain"], ascending=False
    ).reset_index(drop=True)
    best = grid.iloc[0].to_dict()

    def build_candidate(row: dict) -> tuple[float, np.ndarray, np.ndarray]:
        state = frame[str(row["state"])].to_numpy(dtype=np.float64)
        state_z = np.clip(
            (state - float(row["state_mean"])) / float(row["state_scale"]),
            -6.0,
            6.0,
        )
        gate = make_gate(
            state_z,
            first_mask,
            str(row["gate_kind"]),
            float(row["parameter"]),
            float(row["temperature"]),
            float(row["floor"]),
        )
        gated_signal = signal * gate
        gamma = fit_gamma(
            target[first_mask],
            base[first_mask],
            gated_signal[first_mask],
            weight[first_mask],
            gamma_bound,
        )
        return gamma, gate, base + gamma * gated_signal

    gamma, selected_gate, candidate = build_candidate(best)
    constant_row = grid[grid["gate_kind"] == "constant"].iloc[0].to_dict()
    constant_gamma, constant_gate, constant_candidate = build_candidate(constant_row)
    del constant_gate

    def segment_result(mask: np.ndarray, prediction: np.ndarray) -> dict:
        base_score, candidate_score, gain = score_gain(
            target[mask], weight[mask], base[mask], prediction[mask]
        )
        return {
            "base": base_score,
            "candidate": candidate_score,
            "gain": gain,
        }

    result = {
        "fold": fold,
        "selected_auxiliary_variant": selected_variant,
        "split_time": half_time,
        "quarter_split_time": quarter_time,
        "selection": {
            key: (
                float(value)
                if isinstance(value, (np.floating, np.integer))
                else value
            )
            for key, value in best.items()
            if key != "fold"
        },
        "gamma": gamma,
        "constant_gamma": constant_gamma,
        "selected": {
            "full": segment_result(np.ones(len(frame), dtype=bool), candidate),
            "first_half": segment_result(first_mask, candidate),
            "holdout_second_half": segment_result(holdout_mask, candidate),
        },
        "constant": {
            "full": segment_result(
                np.ones(len(frame), dtype=bool), constant_candidate
            ),
            "first_half": segment_result(first_mask, constant_candidate),
            "holdout_second_half": segment_result(
                holdout_mask, constant_candidate
            ),
        },
    }
    result["gain_over_constant"] = {
        segment: result["selected"][segment]["candidate"]
        - result["constant"][segment]["candidate"]
        for segment in ("full", "first_half", "holdout_second_half")
    }
    slices: dict[str, list[dict]] = {
        "blocks": [],
        "assets": [],
        "target_buckets": [],
        "weight_buckets": [],
    }
    for name, prediction in (
        ("selected_gate", candidate),
        ("constant_gate", constant_candidate),
    ):
        rows = evaluate_slices(fold, name, frame, base, prediction)
        slices["blocks"].extend(rows[0])
        slices["assets"].extend(rows[1])
        slices["target_buckets"].extend(rows[2])
        slices["weight_buckets"].extend(rows[3])

    state_output = frame[
        ["row_id", "time_id", "asset_id", *state_columns]
    ].copy()
    state_output["selected_gate"] = selected_gate
    state_output["selected_prediction"] = candidate
    state_output["constant_prediction"] = constant_candidate
    return result, grid, {**slices, "state_output": state_output}


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    paths = sorted((args.data_root / "train").glob("train_partition_*.parquet"))
    if not paths:
        raise FileNotFoundError(args.data_root / "train")
    bounds = {path: parquet_time_bounds(path) for path in paths}

    fold_results: dict[str, dict] = {}
    grid_frames: list[pd.DataFrame] = []
    slice_rows: dict[str, list[dict]] = {
        "blocks": [],
        "assets": [],
        "target_buckets": [],
        "weight_buckets": [],
    }
    for fold, fold_dir in DEFAULT_FOLDS.items():
        states = load_auxiliary_states(
            fold,
            fold_dir,
            paths,
            bounds,
            args.predict_batch_size,
            device,
        )
        result, grid, outputs = tune_fold(
            fold, fold_dir, states, args.gamma_bound
        )
        fold_results[fold] = result
        grid_frames.append(grid)
        for key in slice_rows:
            slice_rows[key].extend(outputs[key])
        outputs["state_output"].to_csv(
            args.results_dir / f"{fold}_state_predictions.csv", index=False
        )
        print(
            f"{fold}: {result['selection']['state']} "
            f"{result['selection']['gate_kind']} "
            f"holdout_gain={result['selected']['holdout_second_half']['gain']:.9f} "
            f"over_constant={result['gain_over_constant']['holdout_second_half']:.9f}",
            flush=True,
        )

    pd.concat(grid_frames, ignore_index=True).to_csv(
        args.results_dir / "selection_grid.csv", index=False
    )
    for key, rows in slice_rows.items():
        pd.DataFrame(rows).to_csv(args.results_dir / f"{key}.csv", index=False)

    historical = [fold_results[name] for name in ("fold0", "fold1", "fold2")]
    latest = fold_results["latest"]
    latest_blocks = pd.DataFrame(slice_rows["blocks"])
    latest_blocks = latest_blocks[
        (latest_blocks["fold"] == "latest")
        & (latest_blocks["candidate"] == "selected_gate")
    ]
    latest_constant_blocks = pd.DataFrame(slice_rows["blocks"])
    latest_constant_blocks = latest_constant_blocks[
        (latest_constant_blocks["fold"] == "latest")
        & (latest_constant_blocks["candidate"] == "constant_gate")
    ]
    promotion_checks = {
        "historical_selected_holdout_positive_all": all(
            row["selected"]["holdout_second_half"]["gain"] > 0
            for row in historical
        ),
        "historical_gate_beats_constant_majority": sum(
            row["gain_over_constant"]["holdout_second_half"] > 0
            for row in historical
        )
        >= 2,
        "latest_selected_holdout_positive": latest["selected"][
            "holdout_second_half"
        ]["gain"]
        > 0,
        "latest_gate_beats_constant_holdout": latest["gain_over_constant"][
            "holdout_second_half"
        ]
        > 0,
        "latest_positive_blocks_not_worse": int((latest_blocks["gain"] > 0).sum())
        >= int((latest_constant_blocks["gain"] > 0).sum()),
    }
    report = {
        "official_test_used": False,
        "protocol": (
            "Gate family and state source are selected by symmetric cross-validation "
            "inside each outer first half. Gamma is then fit on the complete first half "
            "and frozen before the outer second half."
        ),
        "fold_results": fold_results,
        "latest_slice_counts": {
            "selected_positive_blocks": int((latest_blocks["gain"] > 0).sum()),
            "constant_positive_blocks": int(
                (latest_constant_blocks["gain"] > 0).sum()
            ),
            "total_blocks": int(len(latest_blocks)),
        },
        "promotion_checks": promotion_checks,
        "promote_gate_to_test_candidate": all(promotion_checks.values()),
    }
    (args.results_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
