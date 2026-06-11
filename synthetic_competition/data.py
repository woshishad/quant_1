from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SyntheticConfig
from .io import ensure_dir, write_json, write_table


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_latent_state(config: SyntheticConfig, total_times: int, n_assets: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    time_grid = np.linspace(0.0, 2.0 * np.pi, total_times, endpoint=False)
    time_trend = (
        0.7 * np.sin(1.2 * time_grid)
        + 0.35 * np.cos(0.5 * time_grid)
        + np.cumsum(rng.normal(0.0, 0.04, size=total_times))
    )
    regime = np.sign(np.sin(0.45 * time_grid) + 0.15 * np.cos(1.7 * time_grid))
    regime[regime == 0] = 1.0
    cycle = np.sin(2.1 * time_grid) + 0.3 * np.cos(3.5 * time_grid)
    asset_alpha = rng.normal(0.0, 1.0, size=n_assets)
    asset_style = rng.normal(0.0, 1.0, size=n_assets)
    asset_liquidity = rng.normal(0.0, 1.0, size=n_assets)
    return {
        "time_grid": time_grid,
        "time_trend": time_trend,
        "regime": regime,
        "cycle": cycle,
        "asset_alpha": asset_alpha,
        "asset_style": asset_style,
        "asset_liquidity": asset_liquidity,
    }


def _build_feature_frame(config: SyntheticConfig, total_times: int, n_assets: int) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    latent = _make_latent_state(config, total_times, n_assets)
    rows: list[dict[str, Any]] = []
    row_id = 0
    for time_id in range(total_times):
        time_trend = latent["time_trend"][time_id]
        regime = latent["regime"][time_id]
        cycle = latent["cycle"][time_id]
        for asset_id in range(n_assets):
            asset_alpha = latent["asset_alpha"][asset_id]
            asset_style = latent["asset_style"][asset_id]
            asset_liquidity = latent["asset_liquidity"][asset_id]
            noise = rng.normal(0.0, 1.0, size=config.n_features)
            feature_values = np.zeros(config.n_features, dtype=float)

            feature_values[0] = 1.3 * time_trend + 0.45 * asset_alpha + 0.12 * regime + 0.15 * noise[0]
            feature_values[1] = 1.05 * regime + 0.25 * cycle - 0.2 * asset_style + 0.18 * noise[1]
            feature_values[2] = 1.15 * cycle + 0.4 * asset_alpha + 0.2 * noise[2]
            feature_values[3] = 1.1 * time_trend - 0.35 * asset_liquidity + 0.15 * noise[3]
            feature_values[4] = 0.7 * feature_values[0] + 0.35 * feature_values[2] + 0.2 * noise[4]
            feature_values[5] = -0.6 * feature_values[1] + 0.45 * asset_alpha + 0.15 * noise[5]
            feature_values[6] = 0.55 * asset_style + 0.15 * cycle + 0.2 * noise[6]
            feature_values[7] = 0.9 * asset_liquidity + 0.45 * regime + 0.2 * noise[7]
            feature_values[8] = 0.85 * feature_values[2] - 0.2 * time_trend + 0.15 * noise[8]
            feature_values[9] = 0.75 * feature_values[3] + 0.15 * asset_alpha + 0.15 * noise[9]
            for feature_id in range(10, config.n_features):
                latent_mix = (
                    0.25 * time_trend
                    + 0.15 * regime
                    + 0.2 * asset_alpha
                    - 0.1 * asset_style
                    + 0.12 * cycle
                    + 0.1 * asset_liquidity
                )
                feature_values[feature_id] = latent_mix + 0.7 * noise[feature_id]

            weight_signal = (
                1.7 * feature_values[0]
                - 1.25 * feature_values[1]
                + 1.05 * feature_values[4]
                + 0.65 * feature_values[7] * feature_values[1]
                + 0.25 * time_trend
                - 0.2 * asset_style
                + 0.18 * noise[10 % config.n_features]
            )
            weight = 0.25 + 1.6 * sigmoid(weight_signal / 2.0)

            target_signal = (
                0.95 * feature_values[2]
                - 0.8 * feature_values[3]
                + 0.65 * feature_values[4]
                + 0.5 * feature_values[8] * feature_values[1]
                - 0.22 * regime
                + 0.18 * asset_alpha
                + 0.15 * noise[11 % config.n_features]
            )
            target = 0.42 * target_signal + 0.35 * rng.normal(0.0, 1.0)

            responder_0 = 0.7 * feature_values[2] - 0.15 * feature_values[1] + 0.2 * rng.normal(0.0, 1.0)
            responder_1 = 0.6 * feature_values[0] + 0.25 * feature_values[7] + 0.2 * rng.normal(0.0, 1.0)
            responder_2 = -0.45 * feature_values[3] + 0.3 * feature_values[8] + 0.2 * rng.normal(0.0, 1.0)

            row = {
                "row_id": row_id,
                "time_id": time_id,
                "asset_id": asset_id,
                "weight": float(weight),
                "target": float(target),
                "responder_0": float(responder_0),
                "responder_1": float(responder_1),
                "responder_2": float(responder_2),
            }
            for feature_id in range(config.n_features):
                row[f"feature_{feature_id}"] = float(feature_values[feature_id])
            rows.append(row)
            row_id += 1

    frame = pd.DataFrame(rows)
    ordered_columns = (
        ["row_id", "time_id", "asset_id"]
        + [f"feature_{i}" for i in range(config.n_features)]
        + [f"responder_{i}" for i in range(config.n_responders)]
        + ["target", "weight"]
    )
    return frame[ordered_columns]


def _build_test_frame(config: SyntheticConfig, total_times: int, n_assets: int) -> pd.DataFrame:
    frame = _build_feature_frame(config, total_times, n_assets)
    feature_columns = ["row_id", "time_id", "asset_id"] + [f"feature_{i}" for i in range(config.n_features)]
    frame = frame[feature_columns].copy()
    frame["row_id"] = np.arange(len(frame), dtype=int)
    return frame


def _partition_frame(frame: pd.DataFrame, time_col: str, partition_size: int) -> list[pd.DataFrame]:
    partitions: list[pd.DataFrame] = []
    for start in range(frame[time_col].min(), frame[time_col].max() + 1, partition_size):
        stop = start + partition_size
        partition = frame[(frame[time_col] >= start) & (frame[time_col] < stop)].copy()
        if not partition.empty:
            partitions.append(partition)
    return partitions


def generate_dataset(config: SyntheticConfig, output_dir: str | Path) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    train_frame = _build_feature_frame(config, config.n_train_times, config.n_assets)
    train_frame["target"] = train_frame["target"] - train_frame["target"].mean()
    train_frame["weight"] = train_frame["weight"] / max(train_frame["weight"].mean(), 1e-12)
    test_frame = _build_test_frame(config, config.n_test_times, config.n_assets)
    sample_submission = pd.DataFrame(
        {
            "row_id": test_frame["row_id"].to_numpy(),
            "target": np.zeros(len(test_frame), dtype=float),
        }
    )

    train_path = write_table(train_frame, output_dir / "train")
    test_path = write_table(test_frame, output_dir / "test")
    sample_path = write_table(sample_submission, output_dir / "sample_submission")

    train_partitions_dir = ensure_dir(output_dir / "train_partitions")
    test_partitions_dir = ensure_dir(output_dir / "test_partitions")
    train_partitions = []
    for index, partition in enumerate(_partition_frame(train_frame, "time_id", config.train_partition_size)):
        path = write_table(partition, train_partitions_dir / f"train_partition_{index:03d}")
        train_partitions.append(str(path.name))
    test_partitions = []
    for index, partition in enumerate(_partition_frame(test_frame, "time_id", config.test_partition_size)):
        path = write_table(partition, test_partitions_dir / f"test_partition_{index:03d}")
        test_partitions.append(str(path.name))

    manifest = {
        "config": asdict(config),
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "feature_columns": [f"feature_{i}" for i in range(config.n_features)],
        "responder_columns": [f"responder_{i}" for i in range(config.n_responders)],
        "weight_drivers": [f"feature_{i}" for i in config.weight_driver_features],
        "target_drivers": [f"feature_{i}" for i in config.target_driver_features],
        "shared_drivers": [f"feature_{i}" for i in config.shared_driver_features],
        "train_path": str(train_path.name),
        "test_path": str(test_path.name),
        "sample_submission_path": str(sample_path.name),
        "train_partitions": train_partitions,
        "test_partitions": test_partitions,
    }
    write_json(manifest, output_dir / "manifest.json")
    return manifest
