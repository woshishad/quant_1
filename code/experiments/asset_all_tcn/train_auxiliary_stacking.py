from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class AuxiliaryMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe feature -> weight/responders -> target stacking experiment."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--stable-features-file",
        type=Path,
        default=Path("results/asset_all_stable_features_100k/stable_feature_ranking.csv"),
    )
    parser.add_argument(
        "--current-best-predictions",
        type=Path,
        default=Path("results/blend_latest_regime_xgb_residual/calibration_predictions.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/auxiliary_stacking_20260824"),
    )
    parser.add_argument("--aux-train-start", type=int, default=688_480)
    parser.add_argument("--aux-train-end", type=int, default=776_480, help="Exclusive end.")
    parser.add_argument("--aux-valid-start", type=int, default=777_480)
    parser.add_argument("--aux-valid-end", type=int, default=787_480, help="Exclusive end.")
    parser.add_argument("--target-train-start", type=int, default=788_480)
    parser.add_argument("--target-train-end", type=int, default=846_480, help="Exclusive end.")
    parser.add_argument("--cal-start", type=int, default=847_480)
    parser.add_argument("--cal-end", type=int, default=867_480, help="Exclusive end.")
    parser.add_argument("--outer-start", type=int, default=868_480)
    parser.add_argument("--outer-end", type=int, default=888_480, help="Exclusive end.")
    parser.add_argument("--purge-gap", type=int, default=1_000)
    parser.add_argument("--target-feature-count", type=int, default=48)
    parser.add_argument("--aux-ridge-alpha", type=float, default=1_000.0)
    parser.add_argument("--aux-hidden-dim", type=int, default=256)
    parser.add_argument("--aux-dropout", type=float, default=0.05)
    parser.add_argument("--aux-epochs", type=int, default=6)
    parser.add_argument("--aux-patience", type=int, default=2)
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3)
    parser.add_argument("--aux-batch-size", type=int, default=4_096)
    parser.add_argument("--predict-batch-size", type=int, default=16_384)
    parser.add_argument("--target-ridge-alpha", type=float, default=100.0)
    parser.add_argument("--aux-predictability-threshold", type=float, default=0.5)
    parser.add_argument("--aux-lowrank-threshold", type=float, default=0.2)
    parser.add_argument("--aux-target-relevance-count", type=int, default=12)
    parser.add_argument(
        "--aux-target-relevance-min-predictability", type=float, default=0.2
    )
    parser.add_argument("--lgbm-estimators", type=int, default=250)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.01)
    parser.add_argument("--lgbm-num-leaves", type=int, default=31)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=4_000)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=1_000.0)
    parser.add_argument("--lgbm-n-jobs", type=int, default=8)
    parser.add_argument(
        "--target-variant",
        action="append",
        default=[],
        help="Optional target variant whitelist. May be repeated; raw_only is required.",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_zero_mean_r2(y_true: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    denominator = float(np.sum(weight * y_true * y_true))
    if denominator <= 1e-18:
        return 0.0
    return 1.0 - float(np.sum(weight * (y_true - prediction) ** 2)) / denominator


def ordinary_r2(y_true: np.ndarray, prediction: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator <= 1e-18:
        return 0.0
    return 1.0 - float(np.sum((y_true - prediction) ** 2)) / denominator


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return 0.0
    left = left[valid].astype(np.float64, copy=False)
    right = right[valid].astype(np.float64, copy=False)
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = math.sqrt(float(np.sum(left * left) * np.sum(right * right)))
    if denominator <= 1e-18:
        return 0.0
    return float(np.sum(left * right) / denominator)


def safe_weighted_corr(
    left: np.ndarray, right: np.ndarray, weight: np.ndarray
) -> float:
    valid = (
        np.isfinite(left)
        & np.isfinite(right)
        & np.isfinite(weight)
        & (weight > 0)
    )
    if int(valid.sum()) < 2:
        return 0.0
    left = left[valid].astype(np.float64, copy=False)
    right = right[valid].astype(np.float64, copy=False)
    weight = weight[valid].astype(np.float64, copy=False)
    total_weight = float(np.sum(weight))
    if total_weight <= 1e-18:
        return 0.0
    left_centered = left - float(np.sum(weight * left) / total_weight)
    right_centered = right - float(np.sum(weight * right) / total_weight)
    denominator = math.sqrt(
        float(
            np.sum(weight * left_centered * left_centered)
            * np.sum(weight * right_centered * right_centered)
        )
    )
    if denominator <= 1e-18:
        return 0.0
    return float(np.sum(weight * left_centered * right_centered) / denominator)


def optimal_shrink(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    lower: float = 0.0,
    upper: float = 1.2,
) -> float:
    denominator = float(np.sum(weight * prediction * prediction))
    if denominator <= 1e-18:
        return 0.0
    value = float(np.sum(weight * y_true * prediction) / denominator)
    return float(np.clip(value, lower, upper))


def parquet_time_bounds(path: Path) -> tuple[int, int]:
    parquet_file = pq.ParquetFile(path)
    time_index = parquet_file.schema_arrow.names.index("time_id")
    minimum = min(
        parquet_file.metadata.row_group(i).column(time_index).statistics.min
        for i in range(parquet_file.metadata.num_row_groups)
    )
    maximum = max(
        parquet_file.metadata.row_group(i).column(time_index).statistics.max
        for i in range(parquet_file.metadata.num_row_groups)
    )
    return int(minimum), int(maximum)


def read_time_range(
    paths: list[Path],
    bounds: dict[Path, tuple[int, int]],
    columns: list[str],
    start: int,
    end: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        minimum, maximum = bounds[path]
        if maximum < start or minimum >= end:
            continue
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[("time_id", ">=", start), ("time_id", "<", end)],
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError(f"no rows found for time range [{start}, {end})")
    result = pd.concat(frames, ignore_index=True)
    if not result["time_id"].is_monotonic_increasing:
        result = result.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    return result


def validate_time_protocol(args: argparse.Namespace) -> None:
    ordered = [
        ("aux_train", args.aux_train_start, args.aux_train_end),
        ("aux_valid", args.aux_valid_start, args.aux_valid_end),
        ("target_train", args.target_train_start, args.target_train_end),
        ("calibration", args.cal_start, args.cal_end),
        ("outer", args.outer_start, args.outer_end),
    ]
    for name, start, end in ordered:
        if start >= end:
            raise ValueError(f"invalid {name} range [{start}, {end})")
    for (_, _, previous_end), (name, next_start, _) in zip(ordered, ordered[1:]):
        gap = next_start - previous_end
        if gap < args.purge_gap:
            raise ValueError(f"gap before {name} is {gap}, expected at least {args.purge_gap}")


def load_stable_features(path: Path, available: list[str], count: int) -> list[str]:
    frame = pd.read_csv(path)
    if "feature_name" not in frame.columns:
        raise ValueError(f"{path} does not contain feature_name")
    names = frame["feature_name"].astype(str).head(count).tolist()
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"stable feature file contains unknown columns: {missing[:10]}")
    return names


def fit_input_normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0).astype(np.float32)
    scale = np.nanstd(values, axis=0).astype(np.float32)
    mean[~np.isfinite(mean)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return mean, scale


def normalize_aux_inputs(
    values: np.ndarray,
    asset_id: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    asset_count: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values -= mean
    values /= scale
    np.nan_to_num(values, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
    np.clip(values, -10.0, 10.0, out=values)
    output = np.zeros((len(values), values.shape[1] + asset_count), dtype=np.float32)
    output[:, : values.shape[1]] = values
    output[np.arange(len(values)), values.shape[1] + asset_id.astype(np.int64)] = 1.0
    return output


def transform_auxiliary_targets(
    frame: pd.DataFrame,
    auxiliary_names: list[str],
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = frame[auxiliary_names].to_numpy(dtype=np.float32, copy=True)
    values[:, 0] = np.log1p(np.maximum(values[:, 0], 0.0))
    if mean is None or scale is None:
        mean = np.nanmean(values, axis=0).astype(np.float32)
        scale = np.nanstd(values, axis=0).astype(np.float32)
        mean[~np.isfinite(mean)] = 0.0
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    values -= mean
    values /= scale
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return values, mean, scale


def inverse_auxiliary_targets(
    normalized: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = normalized.astype(np.float64, copy=True)
    values *= scale.astype(np.float64)
    values += mean.astype(np.float64)
    values[:, 0] = np.expm1(np.clip(values[:, 0], -20.0, 20.0))
    values[:, 0] = np.maximum(values[:, 0], 0.0)
    return values


def train_auxiliary_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AuxiliaryMLP, list[dict[str, float]]]:
    model = AuxiliaryMLP(
        input_dim=train_x.shape[1],
        output_dim=train_y.shape[1],
        hidden_dim=args.aux_hidden_dim,
        dropout=args.aux_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.aux_learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.aux_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y)),
        batch_size=args.predict_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(args.aux_epochs):
        model.train()
        train_loss_sum = 0.0
        train_rows = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(batch_x)
                loss = nn.functional.smooth_l1_loss(prediction, batch_y, beta=1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach().cpu()) * len(batch_x)
            train_rows += len(batch_x)

        model.eval()
        valid_loss_sum = 0.0
        valid_rows = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    prediction = model(batch_x)
                    loss = nn.functional.smooth_l1_loss(prediction, batch_y, beta=1.0)
                valid_loss_sum += float(loss.detach().cpu()) * len(batch_x)
                valid_rows += len(batch_x)
        train_loss = train_loss_sum / max(train_rows, 1)
        valid_loss = valid_loss_sum / max(valid_rows, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss})
        print(
            f"aux epoch {epoch + 1}/{args.aux_epochs}: train={train_loss:.6f} valid={valid_loss:.6f}",
            flush=True,
        )
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.aux_patience:
                break
    if best_state is None:
        raise RuntimeError("auxiliary MLP did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, history


def predict_auxiliary_mlp(
    model: AuxiliaryMLP,
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(batch_x)
            predictions.append(prediction.float().cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def auxiliary_diagnostics(
    segment: str,
    model_name: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    auxiliary_names: list[str],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(auxiliary_names):
        valid = np.isfinite(truth[:, index]) & np.isfinite(prediction[:, index])
        rows.append(
            {
                "segment": segment,
                "model": model_name,
                "auxiliary_name": name,
                "rows": int(valid.sum()),
                "correlation": safe_corr(truth[valid, index].copy(), prediction[valid, index].copy()),
                "r2": ordinary_r2(truth[valid, index], prediction[valid, index]),
            }
        )
    return rows


def make_target_segment(
    segment_name: str,
    paths: list[Path],
    bounds: dict[Path, tuple[int, int]],
    start: int,
    end: int,
    all_feature_names: list[str],
    target_feature_names: list[str],
    auxiliary_names: list[str],
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    output_mean: np.ndarray,
    output_scale: np.ndarray,
    asset_count: int,
    auxiliary_ridge: Ridge,
    auxiliary_mlp: AuxiliaryMLP,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict[str, float | str]]]:
    columns = list(
        dict.fromkeys(
            ["row_id", "time_id", "asset_id", "target", "weight"]
            + all_feature_names
            + auxiliary_names
        )
    )
    print(f"reading {segment_name} [{start}, {end})", flush=True)
    frame = read_time_range(paths, bounds, columns, start, end)
    raw_features = frame[target_feature_names].to_numpy(dtype=np.float32, copy=True)
    auxiliary_truth = frame[auxiliary_names].to_numpy(dtype=np.float64, copy=True)
    aux_x = normalize_aux_inputs(
        frame[all_feature_names].to_numpy(dtype=np.float32, copy=True),
        frame["asset_id"].to_numpy(dtype=np.int64, copy=False),
        input_mean,
        input_scale,
        asset_count,
    )
    ridge_normalized = auxiliary_ridge.predict(aux_x).astype(np.float32)
    mlp_normalized = predict_auxiliary_mlp(
        auxiliary_mlp, aux_x, args.predict_batch_size, device
    )
    ridge_original = inverse_auxiliary_targets(ridge_normalized, output_mean, output_scale)
    mlp_original = inverse_auxiliary_targets(mlp_normalized, output_mean, output_scale)
    diagnostics = auxiliary_diagnostics(
        segment_name, "ridge", auxiliary_truth, ridge_original, auxiliary_names
    )
    diagnostics += auxiliary_diagnostics(
        segment_name, "mlp", auxiliary_truth, mlp_original, auxiliary_names
    )
    result = {
        "row_id": frame["row_id"].to_numpy(dtype=np.int64, copy=True),
        "time_id": frame["time_id"].to_numpy(dtype=np.int64, copy=True),
        "asset_id": frame["asset_id"].to_numpy(dtype=np.int64, copy=True),
        "target": frame["target"].to_numpy(dtype=np.float32, copy=True),
        "weight": frame["weight"].to_numpy(dtype=np.float32, copy=True),
        "raw": raw_features,
        "ridge_aux": ridge_normalized,
        "mlp_aux": mlp_normalized,
    }
    del frame, auxiliary_truth, aux_x, ridge_original, mlp_original
    gc.collect()
    return result, diagnostics


def normalize_component(
    train: np.ndarray,
    calibration: np.ndarray,
    outer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0).astype(np.float32)
    scale = np.nanstd(train, axis=0).astype(np.float32)
    mean[~np.isfinite(mean)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    outputs: list[np.ndarray] = []
    for values in (train, calibration, outer):
        normalized = values.astype(np.float32, copy=True)
        normalized -= mean
        normalized /= scale
        np.nan_to_num(normalized, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
        np.clip(normalized, -10.0, 10.0, out=normalized)
        outputs.append(normalized)
    return outputs[0], outputs[1], outputs[2], mean, scale


def select_predictable_indices(
    auxiliary_names: list[str],
    validation_correlations: dict[str, float],
    threshold: float,
) -> list[int]:
    indices = [
        index
        for index, name in enumerate(auxiliary_names)
        if validation_correlations.get(name, 0.0) >= threshold
    ]
    if not indices:
        raise ValueError(f"no auxiliary targets meet predictability threshold {threshold}")
    return indices


def build_time_forward_raw_residual(
    raw_values: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    unique_times = np.unique(time_id)
    split_time = int(unique_times[len(unique_times) // 2])
    fit_mask = time_id < split_time
    relevance_mask = ~fit_mask
    sample_weight = weight[fit_mask] / max(float(np.mean(weight[fit_mask])), 1e-12)
    raw_control = Ridge(alpha=ridge_alpha, solver="cholesky")
    raw_control.fit(
        raw_values[fit_mask], target[fit_mask], sample_weight=sample_weight
    )
    relevance_prediction = raw_control.predict(raw_values[relevance_mask])
    relevance_residual = target[relevance_mask] - relevance_prediction
    metadata = {
        "split_time": split_time,
        "fit_rows": int(fit_mask.sum()),
        "relevance_rows": int(relevance_mask.sum()),
        "raw_control_relevance_score": weighted_zero_mean_r2(
            target[relevance_mask], relevance_prediction, weight[relevance_mask]
        ),
    }
    return relevance_mask, relevance_residual, metadata


def select_target_relevant_indices(
    auxiliary_values: np.ndarray,
    auxiliary_names: list[str],
    validation_correlations: dict[str, float],
    relevance_mask: np.ndarray,
    relevance_residual: np.ndarray,
    weight: np.ndarray,
    count: int,
    minimum_predictability: float,
) -> tuple[list[int], list[dict]]:
    rows: list[dict] = []
    relevance_weight = weight[relevance_mask]
    for index, name in enumerate(auxiliary_names):
        predictability = float(validation_correlations.get(name, 0.0))
        residual_correlation = safe_weighted_corr(
            auxiliary_values[relevance_mask, index],
            relevance_residual,
            relevance_weight,
        )
        rows.append(
            {
                "auxiliary_name": name,
                "predictability_correlation": predictability,
                "raw_residual_correlation": residual_correlation,
                "relevance_score": max(predictability, 0.0)
                * abs(residual_correlation),
                "eligible": predictability >= minimum_predictability,
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    eligible_rows.sort(key=lambda row: row["relevance_score"], reverse=True)
    if not eligible_rows:
        raise ValueError(
            "no auxiliary targets meet the target-relevance predictability threshold "
            f"{minimum_predictability}"
        )
    selected_names = {
        row["auxiliary_name"] for row in eligible_rows[: min(count, len(eligible_rows))]
    }
    for row in rows:
        row["selected"] = row["auxiliary_name"] in selected_names
    indices = [
        index for index, name in enumerate(auxiliary_names) if name in selected_names
    ]
    return indices, rows


def build_lowrank_auxiliary_components(
    values: tuple[np.ndarray, np.ndarray, np.ndarray],
    auxiliary_names: list[str],
    validation_correlations: dict[str, float],
    threshold: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[dict]]:
    groups = {
        "weight": ["weight"],
        "responders_00_06": [f"responder_{index:02d}" for index in range(0, 7)],
        "responders_07_16": [f"responder_{index:02d}" for index in range(7, 17)],
        "responders_17_20": [f"responder_{index:02d}" for index in range(17, 21)],
        "responders_21_30": [f"responder_{index:02d}" for index in range(21, 31)],
        "responders_31_37": [f"responder_{index:02d}" for index in range(31, 38)],
        "responders_38_42": [f"responder_{index:02d}" for index in range(38, 43)],
        "responders_43_46": [f"responder_{index:02d}" for index in range(43, 47)],
    }
    auxiliary_index = {name: index for index, name in enumerate(auxiliary_names)}
    group_metadata: list[dict] = []
    group_definitions: list[tuple[np.ndarray, np.ndarray]] = []
    for group_name, names in groups.items():
        selected = [
            name
            for name in names
            if validation_correlations.get(name, 0.0) >= threshold
        ]
        if not selected:
            continue
        raw_weights = np.asarray(
            [validation_correlations[name] for name in selected], dtype=np.float64
        )
        normalized_weights = raw_weights / float(np.sum(raw_weights))
        indices = np.asarray([auxiliary_index[name] for name in selected], dtype=np.int64)
        group_definitions.append((indices, normalized_weights))
        group_metadata.append(
            {
                "group": group_name,
                "auxiliary_names": selected,
                "weights": normalized_weights.tolist(),
            }
        )
    if not group_definitions:
        raise ValueError(f"no low-rank groups meet predictability threshold {threshold}")

    outputs: list[np.ndarray] = []
    for segment_values in values:
        columns = [
            np.sum(
                segment_values[:, indices] * weights.astype(np.float32),
                axis=1,
                keepdims=True,
            )
            for indices, weights in group_definitions
        ]
        outputs.append(np.concatenate(columns, axis=1).astype(np.float32))
    return (outputs[0], outputs[1], outputs[2]), group_metadata


def score_halves(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
) -> dict[str, float]:
    unique_times = np.unique(time_id)
    split_time = int(unique_times[len(unique_times) // 2])
    first = time_id < split_time
    second = ~first
    return {
        "full": weighted_zero_mean_r2(y_true, prediction, weight),
        "first_half": weighted_zero_mean_r2(y_true[first], prediction[first], weight[first]),
        "second_half": weighted_zero_mean_r2(y_true[second], prediction[second], weight[second]),
        "split_time": split_time,
    }


def block_scores(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    blocks: int = 8,
) -> list[dict[str, float | int]]:
    unique_times = np.unique(time_id)
    rows: list[dict[str, float | int]] = []
    for block, times in enumerate(np.array_split(unique_times, blocks)):
        mask = np.isin(time_id, times)
        rows.append(
            {
                "block": block,
                "time_min": int(times[0]),
                "time_max": int(times[-1]),
                "rows": int(mask.sum()),
                "score": weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask]),
            }
        )
    return rows


def score_by_asset(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    asset_id: np.ndarray,
) -> dict[str, float]:
    return {
        str(int(asset)): weighted_zero_mean_r2(
            y_true[asset_id == asset], prediction[asset_id == asset], weight[asset_id == asset]
        )
        for asset in sorted(np.unique(asset_id))
    }


def fit_target_variant(
    name: str,
    train_x: np.ndarray,
    calibration_x: np.ndarray,
    outer_x: np.ndarray,
    train: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
    outer: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict, np.ndarray, object, object]:
    print(f"training target variant: {name} ({train_x.shape[1]} columns)", flush=True)
    y_train = train["target"]
    w_train = train["weight"]
    sample_weight = w_train / max(float(np.mean(w_train)), 1e-12)
    ridge = Ridge(alpha=args.target_ridge_alpha, solver="cholesky")
    ridge.fit(train_x, y_train, sample_weight=sample_weight)
    ridge_train = ridge.predict(train_x).astype(np.float32)
    ridge_cal = ridge.predict(calibration_x).astype(np.float32)
    ridge_outer = ridge.predict(outer_x).astype(np.float32)
    residual_train = y_train - ridge_train
    residual_model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=args.lgbm_estimators,
        learning_rate=args.lgbm_learning_rate,
        num_leaves=args.lgbm_num_leaves,
        min_child_samples=args.lgbm_min_child_samples,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=args.lgbm_reg_lambda,
        random_state=args.seed,
        n_jobs=args.lgbm_n_jobs,
        verbosity=-1,
    )
    residual_model.fit(train_x, residual_train, sample_weight=sample_weight)
    residual_cal = residual_model.predict(calibration_x).astype(np.float32)
    residual_outer = residual_model.predict(outer_x).astype(np.float32)
    candidate_rows: list[dict[str, float]] = []
    best: dict | None = None
    for residual_weight in np.linspace(0.0, 1.0, 5):
        raw_cal = ridge_cal + float(residual_weight) * residual_cal
        shrink = optimal_shrink(calibration["target"], raw_cal, calibration["weight"])
        prediction = shrink * raw_cal
        scores = score_halves(
            calibration["target"], prediction, calibration["weight"], calibration["time_id"]
        )
        selection_score = min(scores["first_half"], scores["second_half"])
        row = {
            "residual_weight": float(residual_weight),
            "shrink": float(shrink),
            "selection_score": float(selection_score),
            "full_score": float(scores["full"]),
            "first_half_score": float(scores["first_half"]),
            "second_half_score": float(scores["second_half"]),
        }
        candidate_rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    if best is None:
        raise RuntimeError(f"no target candidate produced for {name}")
    outer_prediction = float(best["shrink"]) * (
        ridge_outer + float(best["residual_weight"]) * residual_outer
    )
    outer_scores = score_halves(
        outer["target"], outer_prediction, outer["weight"], outer["time_id"]
    )
    metrics = {
        "variant": name,
        "feature_count": int(train_x.shape[1]),
        "calibration_selection": best,
        "calibration_candidates": candidate_rows,
        "outer_scores": outer_scores,
        "outer_score_by_asset": score_by_asset(
            outer["target"], outer_prediction, outer["weight"], outer["asset_id"]
        ),
        "outer_blocks": block_scores(
            outer["target"], outer_prediction, outer["weight"], outer["time_id"]
        ),
        "outer_prediction_std": float(np.std(outer_prediction)),
    }
    return metrics, outer_prediction.astype(np.float64), ridge, residual_model


def refit_target_variant(
    name: str,
    train_x: np.ndarray,
    calibration_x: np.ndarray,
    outer_x: np.ndarray,
    train: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
    outer: dict[str, np.ndarray],
    selection: dict[str, float],
    args: argparse.Namespace,
) -> tuple[dict, np.ndarray, Ridge, lgb.LGBMRegressor]:
    """Refit through calibration after all choices are frozen, then predict the later outer block."""
    print(f"refitting target variant through calibration: {name}", flush=True)
    combined_x = np.concatenate([train_x, calibration_x], axis=0)
    combined_y = np.concatenate([train["target"], calibration["target"]]).astype(np.float32)
    combined_weight = np.concatenate([train["weight"], calibration["weight"]]).astype(np.float32)
    sample_weight = combined_weight / max(float(np.mean(combined_weight)), 1e-12)
    ridge = Ridge(alpha=args.target_ridge_alpha, solver="cholesky")
    ridge.fit(combined_x, combined_y, sample_weight=sample_weight)
    ridge_train = ridge.predict(combined_x).astype(np.float32)
    ridge_outer = ridge.predict(outer_x).astype(np.float32)
    residual_model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=args.lgbm_estimators,
        learning_rate=args.lgbm_learning_rate,
        num_leaves=args.lgbm_num_leaves,
        min_child_samples=args.lgbm_min_child_samples,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=args.lgbm_reg_lambda,
        random_state=args.seed,
        n_jobs=args.lgbm_n_jobs,
        verbosity=-1,
    )
    residual_model.fit(combined_x, combined_y - ridge_train, sample_weight=sample_weight)
    raw_outer = ridge_outer + float(selection["residual_weight"]) * residual_model.predict(outer_x)
    prediction = float(selection["shrink"]) * raw_outer
    metrics = {
        "variant": name,
        "fit_rows": int(len(combined_y)),
        "frozen_selection": selection,
        "outer_scores": score_halves(
            outer["target"], prediction, outer["weight"], outer["time_id"]
        ),
        "outer_score_by_asset": score_by_asset(
            outer["target"], prediction, outer["weight"], outer["asset_id"]
        ),
        "outer_blocks": block_scores(
            outer["target"], prediction, outer["weight"], outer["time_id"]
        ),
        "outer_prediction_std": float(np.std(prediction)),
    }
    return metrics, prediction.astype(np.float64), ridge, residual_model


def fit_incremental_gamma(
    y_true: np.ndarray,
    base_prediction: np.ndarray,
    signal: np.ndarray,
    weight: np.ndarray,
    bound: float = 2.0,
) -> float:
    denominator = float(np.sum(weight * signal * signal))
    if denominator <= 1e-18:
        return 0.0
    gamma = float(np.sum(weight * signal * (y_true - base_prediction)) / denominator)
    return float(np.clip(gamma, -bound, bound))


def evaluate_incremental_signal(
    name: str,
    y_true: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    base_prediction: np.ndarray,
    signal: np.ndarray,
) -> tuple[dict, np.ndarray]:
    unique_times = np.unique(time_id)
    split_time = int(unique_times[len(unique_times) // 2])
    fit_mask = time_id < split_time
    holdout_mask = ~fit_mask
    gamma = fit_incremental_gamma(
        y_true[fit_mask], base_prediction[fit_mask], signal[fit_mask], weight[fit_mask]
    )
    prediction = base_prediction + gamma * signal
    base_scores = score_halves(y_true, base_prediction, weight, time_id)
    selected_scores = score_halves(y_true, prediction, weight, time_id)
    metrics = {
        "signal": name,
        "gamma_fit_first_half": gamma,
        "split_time": split_time,
        "base": base_scores,
        "selected": selected_scores,
        "improvement": {
            "full": selected_scores["full"] - base_scores["full"],
            "fit_first_half": selected_scores["first_half"] - base_scores["first_half"],
            "holdout_second_half": selected_scores["second_half"] - base_scores["second_half"],
        },
        "base_holdout_score": weighted_zero_mean_r2(
            y_true[holdout_mask], base_prediction[holdout_mask], weight[holdout_mask]
        ),
        "selected_holdout_score": weighted_zero_mean_r2(
            y_true[holdout_mask], prediction[holdout_mask], weight[holdout_mask]
        ),
    }
    return metrics, prediction


def main() -> None:
    args = parse_args()
    validate_time_protocol(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    paths = sorted((args.data_root / "train").glob("train_partition_*.parquet"))
    if not paths:
        raise FileNotFoundError(args.data_root / "train")
    bounds = {path: parquet_time_bounds(path) for path in paths}
    schema = pq.ParquetFile(paths[0]).schema_arrow.names
    all_feature_names = [name for name in schema if name.startswith("feature_")]
    responder_names = [name for name in schema if name.startswith("responder_")]
    auxiliary_names = ["weight", *responder_names]
    target_feature_names = load_stable_features(
        args.stable_features_file, schema, args.target_feature_count
    )
    asset_count = int(
        pq.read_table(paths[0], columns=["asset_id"])["asset_id"].to_pandas().max() + 1
    )

    auxiliary_columns = list(
        dict.fromkeys(["time_id", "asset_id"] + all_feature_names + auxiliary_names)
    )
    print("loading auxiliary train", flush=True)
    aux_train = read_time_range(
        paths, bounds, auxiliary_columns, args.aux_train_start, args.aux_train_end
    )
    print("loading auxiliary validation", flush=True)
    aux_valid = read_time_range(
        paths, bounds, auxiliary_columns, args.aux_valid_start, args.aux_valid_end
    )
    aux_train_values = aux_train[all_feature_names].to_numpy(dtype=np.float32, copy=True)
    input_mean, input_scale = fit_input_normalizer(aux_train_values)
    aux_train_x = normalize_aux_inputs(
        aux_train_values,
        aux_train["asset_id"].to_numpy(dtype=np.int64, copy=False),
        input_mean,
        input_scale,
        asset_count,
    )
    del aux_train_values
    aux_valid_x = normalize_aux_inputs(
        aux_valid[all_feature_names].to_numpy(dtype=np.float32, copy=True),
        aux_valid["asset_id"].to_numpy(dtype=np.int64, copy=False),
        input_mean,
        input_scale,
        asset_count,
    )
    aux_train_y, output_mean, output_scale = transform_auxiliary_targets(
        aux_train, auxiliary_names
    )
    aux_valid_y, _, _ = transform_auxiliary_targets(
        aux_valid, auxiliary_names, output_mean, output_scale
    )
    del aux_train, aux_valid
    gc.collect()

    print("fitting multi-output auxiliary Ridge", flush=True)
    auxiliary_ridge = Ridge(
        alpha=args.aux_ridge_alpha,
        solver="cholesky",
        fit_intercept=True,
    )
    auxiliary_ridge.fit(aux_train_x, aux_train_y)
    print("fitting shared auxiliary MLP", flush=True)
    auxiliary_mlp, auxiliary_history = train_auxiliary_mlp(
        aux_train_x, aux_train_y, aux_valid_x, aux_valid_y, args, device
    )
    aux_valid_ridge = auxiliary_ridge.predict(aux_valid_x).astype(np.float32)
    aux_valid_mlp = predict_auxiliary_mlp(
        auxiliary_mlp, aux_valid_x, args.predict_batch_size, device
    )
    validation_truth = inverse_auxiliary_targets(aux_valid_y, output_mean, output_scale)
    auxiliary_diagnostic_rows = auxiliary_diagnostics(
        "aux_valid",
        "ridge",
        validation_truth,
        inverse_auxiliary_targets(aux_valid_ridge, output_mean, output_scale),
        auxiliary_names,
    )
    auxiliary_diagnostic_rows += auxiliary_diagnostics(
        "aux_valid",
        "mlp",
        validation_truth,
        inverse_auxiliary_targets(aux_valid_mlp, output_mean, output_scale),
        auxiliary_names,
    )
    del aux_train_x, aux_train_y, aux_valid_x, aux_valid_y
    del aux_valid_ridge, aux_valid_mlp, validation_truth
    gc.collect()

    torch.save(
        {
            "state_dict": auxiliary_mlp.state_dict(),
            "input_dim": len(all_feature_names) + asset_count,
            "output_dim": len(auxiliary_names),
            "hidden_dim": args.aux_hidden_dim,
            "dropout": args.aux_dropout,
            "feature_names": all_feature_names,
            "auxiliary_names": auxiliary_names,
        },
        args.results_dir / "auxiliary_mlp.pt",
    )
    with (args.results_dir / "auxiliary_ridge.pkl").open("wb") as file:
        pickle.dump(auxiliary_ridge, file, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(
        args.results_dir / "auxiliary_normalizers.npz",
        input_mean=input_mean,
        input_scale=input_scale,
        output_mean=output_mean,
        output_scale=output_scale,
    )

    target_train, rows = make_target_segment(
        "target_train",
        paths,
        bounds,
        args.target_train_start,
        args.target_train_end,
        all_feature_names,
        target_feature_names,
        auxiliary_names,
        input_mean,
        input_scale,
        output_mean,
        output_scale,
        asset_count,
        auxiliary_ridge,
        auxiliary_mlp,
        args,
        device,
    )
    auxiliary_diagnostic_rows += rows
    calibration, rows = make_target_segment(
        "calibration",
        paths,
        bounds,
        args.cal_start,
        args.cal_end,
        all_feature_names,
        target_feature_names,
        auxiliary_names,
        input_mean,
        input_scale,
        output_mean,
        output_scale,
        asset_count,
        auxiliary_ridge,
        auxiliary_mlp,
        args,
        device,
    )
    auxiliary_diagnostic_rows += rows
    outer, rows = make_target_segment(
        "outer",
        paths,
        bounds,
        args.outer_start,
        args.outer_end,
        all_feature_names,
        target_feature_names,
        auxiliary_names,
        input_mean,
        input_scale,
        output_mean,
        output_scale,
        asset_count,
        auxiliary_ridge,
        auxiliary_mlp,
        args,
        device,
    )
    auxiliary_diagnostic_rows += rows
    auxiliary_diagnostics_frame = pd.DataFrame(auxiliary_diagnostic_rows)
    auxiliary_diagnostics_frame.to_csv(
        args.results_dir / "auxiliary_prediction_metrics.csv", index=False
    )

    component_values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    component_normalizers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for component in ("raw", "ridge_aux", "mlp_aux"):
        train_values, calibration_values, outer_values, mean, scale = normalize_component(
            target_train[component], calibration[component], outer[component]
        )
        component_values[component] = (train_values, calibration_values, outer_values)
        component_normalizers[component] = (mean, scale)
        del target_train[component], calibration[component], outer[component]
    gc.collect()
    np.savez_compressed(
        args.results_dir / "target_component_normalizers.npz",
        raw_mean=component_normalizers["raw"][0],
        raw_scale=component_normalizers["raw"][1],
        ridge_aux_mean=component_normalizers["ridge_aux"][0],
        ridge_aux_scale=component_normalizers["ridge_aux"][1],
        mlp_aux_mean=component_normalizers["mlp_aux"][0],
        mlp_aux_scale=component_normalizers["mlp_aux"][1],
    )

    auxiliary_index = {name: index for index, name in enumerate(auxiliary_names)}
    liquidity_names = ["weight", *[f"responder_{index:02d}" for index in range(31, 43)]]
    direction_names = [
        "responder_03",
        "responder_28",
        "responder_02",
        "responder_29",
        "responder_18",
        "responder_11",
        "responder_19",
        "responder_04",
        "responder_17",
        "responder_30",
    ]
    groups = {
        "ridge_weight": [auxiliary_index["weight"]],
        "mlp_weight": [auxiliary_index["weight"]],
        "ridge_liquidity": [auxiliary_index[name] for name in liquidity_names],
        "mlp_liquidity": [auxiliary_index[name] for name in liquidity_names],
        "ridge_direction": [auxiliary_index[name] for name in direction_names],
        "mlp_direction": [auxiliary_index[name] for name in direction_names],
    }
    for group_name, indices in groups.items():
        source = "ridge_aux" if group_name.startswith("ridge_") else "mlp_aux"
        component_values[group_name] = tuple(
            values[:, indices] for values in component_values[source]
        )

    relevance_mask, relevance_residual, relevance_protocol = (
        build_time_forward_raw_residual(
            component_values["raw"][0],
            target_train["target"],
            target_train["weight"],
            target_train["time_id"],
            args.target_ridge_alpha,
        )
    )
    predictability_metadata: dict[str, dict] = {}
    for model_name in ("ridge", "mlp"):
        model_rows = auxiliary_diagnostics_frame[
            (auxiliary_diagnostics_frame["segment"] == "aux_valid")
            & (auxiliary_diagnostics_frame["model"] == model_name)
        ]
        correlations = dict(
            zip(
                model_rows["auxiliary_name"].astype(str),
                model_rows["correlation"].astype(float),
            )
        )
        predictable_indices = select_predictable_indices(
            auxiliary_names, correlations, args.aux_predictability_threshold
        )
        source_name = f"{model_name}_aux"
        predictable_name = f"{model_name}_predictable"
        component_values[predictable_name] = tuple(
            values[:, predictable_indices] for values in component_values[source_name]
        )
        lowrank_values, lowrank_groups = build_lowrank_auxiliary_components(
            component_values[source_name],
            auxiliary_names,
            correlations,
            args.aux_lowrank_threshold,
        )
        lowrank_name = f"{model_name}_lowrank"
        component_values[lowrank_name] = lowrank_values
        target_relevant_indices, target_relevance_rows = (
            select_target_relevant_indices(
                component_values[source_name][0],
                auxiliary_names,
                correlations,
                relevance_mask,
                relevance_residual,
                target_train["weight"],
                args.aux_target_relevance_count,
                args.aux_target_relevance_min_predictability,
            )
        )
        target_relevant_name = f"{model_name}_target_relevant"
        component_values[target_relevant_name] = tuple(
            values[:, target_relevant_indices]
            for values in component_values[source_name]
        )
        predictability_metadata[model_name] = {
            "validation_correlations": correlations,
            "predictable_threshold": args.aux_predictability_threshold,
            "predictable_auxiliary_names": [
                auxiliary_names[index] for index in predictable_indices
            ],
            "lowrank_threshold": args.aux_lowrank_threshold,
            "lowrank_groups": lowrank_groups,
            "target_relevance_protocol": relevance_protocol,
            "target_relevance_count": args.aux_target_relevance_count,
            "target_relevance_min_predictability": (
                args.aux_target_relevance_min_predictability
            ),
            "target_relevance_rows": target_relevance_rows,
            "target_relevant_auxiliary_names": [
                auxiliary_names[index] for index in target_relevant_indices
            ],
        }
    variant_components = {
        "raw_only": ["raw"],
        "predicted_aux_only": ["ridge_aux", "mlp_aux"],
        "predicted_predictable_only": ["ridge_predictable", "mlp_predictable"],
        "predicted_lowrank_only": ["ridge_lowrank", "mlp_lowrank"],
        "predicted_target_relevant_only": [
            "ridge_target_relevant",
            "mlp_target_relevant",
        ],
        "raw_plus_ridge_weight": ["raw", "ridge_weight"],
        "raw_plus_mlp_weight": ["raw", "mlp_weight"],
        "raw_plus_ridge_liquidity": ["raw", "ridge_liquidity"],
        "raw_plus_mlp_liquidity": ["raw", "mlp_liquidity"],
        "raw_plus_ridge_direction": ["raw", "ridge_direction"],
        "raw_plus_mlp_direction": ["raw", "mlp_direction"],
        "raw_plus_ridge_aux": ["raw", "ridge_aux"],
        "raw_plus_mlp_aux": ["raw", "mlp_aux"],
        "raw_plus_all_aux": ["raw", "ridge_aux", "mlp_aux"],
        "raw_plus_predictable_aux": ["raw", "ridge_predictable", "mlp_predictable"],
        "raw_plus_lowrank_aux": ["raw", "ridge_lowrank", "mlp_lowrank"],
        "raw_plus_target_relevant_aux": [
            "raw",
            "ridge_target_relevant",
            "mlp_target_relevant",
        ],
    }
    if args.target_variant:
        requested_variants = set(args.target_variant)
        unknown_variants = sorted(requested_variants - set(variant_components))
        if unknown_variants:
            raise ValueError(f"unknown target variants: {unknown_variants}")
        if "raw_only" not in requested_variants:
            raise ValueError("--target-variant whitelist must include raw_only")
        variant_components = {
            name: components
            for name, components in variant_components.items()
            if name in requested_variants
        }
    variant_metrics: dict[str, dict] = {}
    variant_predictions: dict[str, np.ndarray] = {}
    refit_metrics: dict[str, dict] = {}
    refit_predictions: dict[str, np.ndarray] = {}
    chosen_models: dict[str, tuple[object, object]] = {}
    refit_models: dict[str, tuple[object, object]] = {}
    for variant, components in variant_components.items():
        train_x = np.concatenate([component_values[name][0] for name in components], axis=1)
        calibration_x = np.concatenate([component_values[name][1] for name in components], axis=1)
        outer_x = np.concatenate([component_values[name][2] for name in components], axis=1)
        metrics, prediction, ridge, residual_model = fit_target_variant(
            variant,
            train_x,
            calibration_x,
            outer_x,
            target_train,
            calibration,
            outer,
            args,
        )
        variant_metrics[variant] = metrics
        variant_predictions[variant] = prediction
        chosen_models[variant] = (ridge, residual_model)
        refit_info, refit_prediction, refit_ridge, refit_residual_model = refit_target_variant(
            variant,
            train_x,
            calibration_x,
            outer_x,
            target_train,
            calibration,
            outer,
            metrics["calibration_selection"],
            args,
        )
        refit_metrics[variant] = refit_info
        refit_predictions[variant] = refit_prediction
        refit_models[variant] = (refit_ridge, refit_residual_model)
        del train_x, calibration_x, outer_x
        gc.collect()

    auxiliary_variants = [name for name in variant_components if name != "raw_only"]
    selected_aux_variant = max(
        auxiliary_variants,
        key=lambda name: variant_metrics[name]["calibration_selection"]["selection_score"],
    )
    raw_prediction = variant_predictions["raw_only"]
    selected_aux_prediction = variant_predictions[selected_aux_variant]
    auxiliary_delta = selected_aux_prediction - raw_prediction

    outer_index = pd.DataFrame(
        {
            "row_id": outer["row_id"],
            "time_id": outer["time_id"],
            "asset_id": outer["asset_id"],
        }
    )
    external_predictions = outer_index.copy()
    external_predictions["target"] = outer["target"]
    external_predictions["weight"] = outer["weight"]
    external_predictions["raw_control_prediction"] = raw_prediction
    external_predictions["selected_aux_prediction"] = selected_aux_prediction
    external_predictions["auxiliary_delta"] = auxiliary_delta
    for variant, prediction in variant_predictions.items():
        external_predictions[f"prediction_{variant}"] = prediction
    for variant, prediction in refit_predictions.items():
        external_predictions[f"prediction_refit_{variant}"] = prediction
    external_incremental_tests: dict[str, dict] = {}
    if args.current_best_predictions.exists():
        current_best = pd.read_csv(
            args.current_best_predictions,
            usecols=["row_id", "prediction"],
        )
        aligned = outer_index.merge(
            current_best, on="row_id", how="left", validate="one_to_one"
        )
        if not aligned["prediction"].isna().any():
            current_prediction = aligned["prediction"].to_numpy(dtype=np.float64)
            delta_metrics, delta_blend_prediction = evaluate_incremental_signal(
                "auxiliary_delta_over_raw",
                outer["target"],
                outer["weight"],
                outer["time_id"],
                current_prediction,
                auxiliary_delta,
            )
            absolute_metrics, absolute_blend_prediction = evaluate_incremental_signal(
                "selected_two_stage_prediction",
                outer["target"],
                outer["weight"],
                outer["time_id"],
                current_prediction,
                selected_aux_prediction,
            )
            external_predictions["current_best_prediction"] = current_prediction
            external_predictions["delta_blend_prediction"] = delta_blend_prediction
            external_predictions["absolute_blend_prediction"] = absolute_blend_prediction
            external_incremental_tests = {
                "auxiliary_delta": delta_metrics,
                "absolute_two_stage_prediction": absolute_metrics,
            }
    external_predictions.to_csv(args.results_dir / "outer_predictions.csv", index=False)

    for variant, (ridge, residual_model) in chosen_models.items():
        with (args.results_dir / f"target_{variant}_ridge.pkl").open("wb") as file:
            pickle.dump(ridge, file, protocol=pickle.HIGHEST_PROTOCOL)
        residual_model.booster_.save_model(
            args.results_dir / f"target_{variant}_residual_lgbm.txt"
        )
    for variant, (ridge, residual_model) in refit_models.items():
        with (args.results_dir / f"target_refit_{variant}_ridge.pkl").open("wb") as file:
            pickle.dump(ridge, file, protocol=pickle.HIGHEST_PROTOCOL)
        residual_model.booster_.save_model(
            args.results_dir / f"target_refit_{variant}_residual_lgbm.txt"
        )

    protocol = {
        "aux_train": [args.aux_train_start, args.aux_train_end - 1],
        "aux_valid": [args.aux_valid_start, args.aux_valid_end - 1],
        "target_train": [args.target_train_start, args.target_train_end - 1],
        "calibration": [args.cal_start, args.cal_end - 1],
        "outer_validation": [args.outer_start, args.outer_end - 1],
        "minimum_purge_gap": args.purge_gap,
        "target_stage_inputs": "raw feature subset plus predictions made by auxiliary models trained strictly earlier",
        "forbidden_inputs": "true row-level weight/responders are never target model features",
    }
    summary = {
        "leakage_safe": True,
        "official_test_used": False,
        "protocol": protocol,
        "rows": {
            "target_train": int(len(target_train["target"])),
            "calibration": int(len(calibration["target"])),
            "outer": int(len(outer["target"])),
        },
        "features": {
            "auxiliary_input_feature_count": len(all_feature_names),
            "target_raw_feature_count": len(target_feature_names),
            "predicted_auxiliary_count_per_model": len(auxiliary_names),
            "auxiliary_names": auxiliary_names,
            "target_feature_names": target_feature_names,
            "auxiliary_groups": {
                "liquidity": liquidity_names,
                "direction": direction_names,
            },
            "predictability_selection": predictability_metadata,
        },
        "auxiliary_mlp_history": auxiliary_history,
        "target_variants": variant_metrics,
        "target_refit_variants": refit_metrics,
        "selected_aux_variant_on_internal_calibration": selected_aux_variant,
        "selected_aux_calibration_selection_score": variant_metrics[selected_aux_variant][
            "calibration_selection"
        ]["selection_score"],
        "raw_calibration_selection_score": variant_metrics["raw_only"][
            "calibration_selection"
        ]["selection_score"],
        "external_incremental_tests": external_incremental_tests,
        "promotion_rule": "promote only if the preselected auxiliary variant improves internal calibration stability and a first-half-fitted incremental coefficient improves the untouched outer second half",
        "outputs": {
            "summary": str(args.results_dir / "summary.json"),
            "auxiliary_metrics": str(args.results_dir / "auxiliary_prediction_metrics.csv"),
            "outer_predictions": str(args.results_dir / "outer_predictions.csv"),
            "auxiliary_mlp": str(args.results_dir / "auxiliary_mlp.pt"),
            "auxiliary_ridge": str(args.results_dir / "auxiliary_ridge.pkl"),
        },
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.results_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
