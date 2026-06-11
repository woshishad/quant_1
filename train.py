from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthetic_competition.analysis import correlation_table, feature_recovery_report, permutation_importance
from synthetic_competition.config import SyntheticConfig
from synthetic_competition.io import ensure_dir, read_json, read_table, write_json, write_table
from synthetic_competition.metrics import r2_score, weighted_r2
from synthetic_competition.models import FeatureInteractionBuilder, FusionRegressor, RidgeRegressor


def build_feature_frame(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    return frame[feature_names].to_numpy(dtype=float)


def time_split(frame: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_times = np.sort(frame["time_id"].unique())
    split_index = max(1, int(len(unique_times) * (1.0 - validation_fraction)))
    train_times = unique_times[:split_index]
    valid_times = unique_times[split_index:]
    train_frame = frame[frame["time_id"].isin(train_times)].copy()
    valid_frame = frame[frame["time_id"].isin(valid_times)].copy()
    return train_frame, valid_frame


def save_model_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    return write_json(bundle, path)


def load_or_generate_data(data_dir: Path, config: SyntheticConfig) -> Path:
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        return manifest_path
    from synthetic_competition.data import generate_dataset

    generate_dataset(config, data_dir)
    return manifest_path


def train_pipeline(data_dir: Path, artifacts_dir: Path, config: SyntheticConfig) -> dict[str, Any]:
    ensure_dir(artifacts_dir)
    load_or_generate_data(data_dir, config)

    train_frame = read_table(data_dir / "train")
    feature_names = [c for c in train_frame.columns if c.startswith("feature_")]
    target_name = "target"
    weight_name = "weight"

    train_fold, valid_fold = time_split(train_frame, config.validation_fraction)
    X_train = build_feature_frame(train_fold, feature_names)
    X_valid = build_feature_frame(valid_fold, feature_names)
    y_train = train_fold[target_name].to_numpy(dtype=float)
    y_valid = valid_fold[target_name].to_numpy(dtype=float)
    w_train = train_fold[weight_name].to_numpy(dtype=float)
    w_valid = valid_fold[weight_name].to_numpy(dtype=float)

    weight_model = RidgeRegressor(alpha=config.weight_ridge_alpha).fit(X_train, w_train)
    weight_valid_pred = weight_model.predict(X_valid)

    interaction_builder = FeatureInteractionBuilder(config.interaction_features, include_squares=True)
    base_target_model = RidgeRegressor(alpha=config.ridge_alpha)
    interaction_target_model = RidgeRegressor(alpha=config.interaction_ridge_alpha)
    target_model = FusionRegressor(
        base_model=base_target_model,
        interaction_model=interaction_target_model,
        interaction_builder=interaction_builder,
        base_weight=0.65,
    ).fit(X_train, y_train, sample_weight=w_train)
    target_valid_pred = target_model.predict(X_valid)

    base_train_pred = target_model.base_model.predict(X_valid)
    interaction_valid_features, interaction_names = interaction_builder.transform(X_valid)
    interaction_valid_pred = target_model.interaction_model.predict(interaction_valid_features)

    target_metrics = {
        "weighted_r2": weighted_r2(y_valid, target_valid_pred, sample_weight=w_valid),
        "r2": r2_score(y_valid, target_valid_pred),
        "baseline_zero_weighted_r2": weighted_r2(y_valid, np.zeros_like(y_valid), sample_weight=w_valid),
        "baseline_zero_r2": r2_score(y_valid, np.zeros_like(y_valid)),
    }
    weight_metrics = {
        "r2": r2_score(w_valid, weight_valid_pred),
    }

    feature_corr_target = correlation_table(X_train, y_train, feature_names)
    feature_corr_weight = correlation_table(X_train, w_train, feature_names)
    target_permutation = permutation_importance(
        target_model.predict,
        X_valid,
        y_valid,
        feature_names,
        sample_weight=w_valid,
        metric=weighted_r2,
        random_state=config.seed,
    )
    weight_permutation = permutation_importance(
        weight_model.predict,
        X_valid,
        w_valid,
        feature_names,
        sample_weight=None,
        metric=r2_score,
        random_state=config.seed + 1,
    )

    target_recovery = feature_recovery_report(
        feature_names=feature_names,
        base_coef=target_model.base_model.coef_,
        interaction_names=interaction_names,
        interaction_coef=target_model.interaction_model.coef_,
        true_drivers=[f"feature_{i}" for i in config.target_driver_features],
        top_k_size=8,
    )
    weight_recovery = feature_recovery_report(
        feature_names=feature_names,
        base_coef=weight_model.coef_,
        interaction_names=[],
        interaction_coef=np.array([]),
        true_drivers=[f"feature_{i}" for i in config.weight_driver_features],
        top_k_size=8,
    )

    bundle = {
        "config": asdict(config),
        "feature_names": feature_names,
        "target_model": {
            "base_model": {
                "alpha": target_model.base_model.alpha,
                "mean": target_model.base_model.standardizer.mean_.tolist(),
                "scale": target_model.base_model.standardizer.scale_.tolist(),
                "coef": target_model.base_model.coef_.tolist(),
                "intercept": target_model.base_model.intercept_,
            },
            "interaction_model": {
                "alpha": target_model.interaction_model.alpha,
                "mean": target_model.interaction_model.standardizer.mean_.tolist(),
                "scale": target_model.interaction_model.standardizer.scale_.tolist(),
                "coef": target_model.interaction_model.coef_.tolist(),
                "intercept": target_model.interaction_model.intercept_,
            },
            "interaction_features": list(config.interaction_features),
            "base_weight": target_model.base_weight,
        },
        "weight_model": {
            "alpha": weight_model.alpha,
            "mean": weight_model.standardizer.mean_.tolist(),
            "scale": weight_model.standardizer.scale_.tolist(),
            "coef": weight_model.coef_.tolist(),
            "intercept": weight_model.intercept_,
        },
        "metrics": {
            "target": target_metrics,
            "weight": weight_metrics,
        },
        "recovery": {
            "target": target_recovery,
            "weight": weight_recovery,
        },
    }

    save_model_bundle(bundle, artifacts_dir / "model_bundle.json")
    write_json(bundle["metrics"], artifacts_dir / "metrics.json")
    write_json(bundle["recovery"], artifacts_dir / "feature_recovery.json")
    write_table(feature_corr_target, artifacts_dir / "target_feature_correlation")
    write_table(feature_corr_weight, artifacts_dir / "weight_feature_correlation")
    write_table(target_permutation, artifacts_dir / "target_permutation_importance")
    write_table(weight_permutation, artifacts_dir / "weight_permutation_importance")

    summary = {
        "train_rows": len(train_frame),
        "valid_rows": len(valid_fold),
        "feature_count": len(feature_names),
        "metrics": bundle["metrics"],
        "recovery": bundle["recovery"],
    }
    write_json(summary, artifacts_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synthetic competition models.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SyntheticConfig(seed=args.seed)
    summary = train_pipeline(args.data_dir, args.artifacts_dir, config)
    print(pd.Series(summary["metrics"]["target"]))
    print(pd.Series(summary["metrics"]["weight"]))
    print(pd.Series(summary["recovery"]["target"]))


if __name__ == "__main__":
    main()
