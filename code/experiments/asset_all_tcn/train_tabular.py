from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all-asset Ridge/LightGBM tabular baselines.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/asset_all_time50000"))
    parser.add_argument(
        "--selected-features-file",
        type=Path,
        default=Path("results/asset01_ga_lgbm_tcn/selected_features.csv"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset_all_tabular_baseline"))
    parser.add_argument("--train-end-time", type=int, default=39_999)
    parser.add_argument("--valid-start-time", type=int, default=40_000)
    parser.add_argument("--valid-end-time", type=int, default=49_999)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--use-asset-dummies", action="store_true")
    parser.add_argument("--lgbm-estimators", type=int, default=300)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.01)
    parser.add_argument("--lgbm-num-leaves", type=int, default=7)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=5000)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=200.0)
    return parser.parse_args()


def load_selected_feature_names(path: Path, available_columns: list[str]) -> list[str]:
    selected_frame = pd.read_csv(path)
    if "feature_name" in selected_frame.columns:
        names = selected_frame["feature_name"].astype(str).tolist()
    elif "feature_index" in selected_frame.columns:
        indices = selected_frame["feature_index"].astype(int).tolist()
        feature_columns = [col for col in available_columns if col.startswith("feature_")]
        names = [feature_columns[index] for index in indices]
    else:
        raise ValueError(f"{path} must contain feature_name or feature_index column")
    missing = sorted(set(names) - set(available_columns))
    if missing:
        raise ValueError(f"selected features are not present in dataset: {missing[:10]}")
    return names


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    denominator = float(np.sum(weight * y_true * y_true))
    if denominator <= 0:
        return 0.0
    return 1.0 - float(np.sum(weight * (y_true - y_pred) ** 2)) / denominator


def optimal_shrink(y_true: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.sum(weight * prediction * prediction))
    if denominator <= 1e-18:
        return 0.0
    shrink = float(np.sum(weight * y_true * prediction) / denominator)
    return min(1.2, max(0.0, shrink))


def score_prediction(y_true: np.ndarray, prediction: np.ndarray, weight: np.ndarray, asset_id: np.ndarray) -> dict:
    shrink = optimal_shrink(y_true, prediction, weight)
    shrunk_prediction = shrink * prediction
    by_asset = {}
    for asset in sorted(np.unique(asset_id)):
        mask = asset_id == asset
        by_asset[str(int(asset))] = weighted_zero_mean_r2(y_true[mask], shrunk_prediction[mask], weight[mask])
    return {
        "raw_score": float(weighted_zero_mean_r2(y_true, prediction, weight)),
        "shrink": float(shrink),
        "shrink_score": float(weighted_zero_mean_r2(y_true, shrunk_prediction, weight)),
        "shrink_score_by_asset": by_asset,
    }


def standardize(train_x: np.ndarray, valid_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 表格模型同样只用训练段拟合标准化统计量，验证段不参与均值/方差估计。
    mean = np.nanmean(train_x, axis=0).astype(np.float32)
    scale = np.nanstd(train_x, axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    train_x = np.nan_to_num((train_x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    valid_x = np.nan_to_num((valid_x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return train_x, valid_x, mean, scale


def add_asset_dummies(x: np.ndarray, asset_id: np.ndarray, asset_ids: list[int]) -> tuple[np.ndarray, list[str]]:
    # asset_id 是公开可用的身份信息；给 tabular 模型 one-hot 后，可以学习每个标的自己的截距/偏置。
    dummy = np.zeros((len(asset_id), len(asset_ids)), dtype=np.float32)
    index = {asset: i for i, asset in enumerate(asset_ids)}
    for row, asset in enumerate(asset_id):
        dummy[row, index[int(asset)]] = 1.0
    names = [f"asset_dummy_{asset}" for asset in asset_ids]
    return np.concatenate([x, dummy], axis=1), names


def find_best_two_model_blend(y_true: np.ndarray, ridge_pred: np.ndarray, lgbm_pred: np.ndarray, weight: np.ndarray) -> dict:
    best = {"score": -np.inf}
    for ridge_weight in np.linspace(0.0, 1.0, 201):
        lgbm_weight = 1.0 - ridge_weight
        base = ridge_weight * ridge_pred + lgbm_weight * lgbm_pred
        shrink = optimal_shrink(y_true, base, weight)
        prediction = shrink * base
        score = weighted_zero_mean_r2(y_true, prediction, weight)
        if score > best["score"]:
            best = {
                "score": float(score),
                "weights": {
                    "ridge": float(ridge_weight),
                    "lgbm": float(lgbm_weight),
                    "shrink": float(shrink),
                },
                "prediction": prediction,
            }
    return best


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_dir / "train.parquet"
    schema_columns = pq.ParquetFile(data_path).schema_arrow.names
    selected_features = load_selected_feature_names(args.selected_features_file, schema_columns)
    columns = ["row_id", "time_id", "asset_id", "weight", "target"] + selected_features
    frame = pd.read_parquet(data_path, columns=columns).sort_values(["time_id", "asset_id"], kind="mergesort")

    time_values = frame["time_id"].to_numpy(dtype=np.int64)
    train_mask = time_values <= args.train_end_time
    valid_mask = (time_values >= args.valid_start_time) & (time_values <= args.valid_end_time)
    asset_ids = [int(asset) for asset in sorted(frame["asset_id"].unique().tolist())]

    train_x = frame.loc[train_mask, selected_features].to_numpy(dtype=np.float32)
    valid_x = frame.loc[valid_mask, selected_features].to_numpy(dtype=np.float32)
    train_x, valid_x, mean, scale = standardize(train_x, valid_x)
    feature_names = selected_features.copy()
    if args.use_asset_dummies:
        train_asset = frame.loc[train_mask, "asset_id"].to_numpy(dtype=np.int64)
        valid_asset = frame.loc[valid_mask, "asset_id"].to_numpy(dtype=np.int64)
        train_x, dummy_names = add_asset_dummies(train_x, train_asset, asset_ids)
        valid_x, _ = add_asset_dummies(valid_x, valid_asset, asset_ids)
        feature_names += dummy_names

    y_train = frame.loc[train_mask, "target"].to_numpy(dtype=np.float32)
    y_valid = frame.loc[valid_mask, "target"].to_numpy(dtype=np.float32)
    w_train = frame.loc[train_mask, "weight"].to_numpy(dtype=np.float32)
    w_valid = frame.loc[valid_mask, "weight"].to_numpy(dtype=np.float32)
    asset_valid = frame.loc[valid_mask, "asset_id"].to_numpy(dtype=np.int64)
    sample_weight = w_train / max(float(np.mean(w_train)), 1e-12)

    ridge = Ridge(alpha=args.ridge_alpha, solver="lsqr", max_iter=500)
    ridge.fit(train_x, y_train, sample_weight=sample_weight)
    ridge_pred = ridge.predict(valid_x)

    lgbm = LGBMRegressor(
        objective="regression",
        n_estimators=args.lgbm_estimators,
        learning_rate=args.lgbm_learning_rate,
        num_leaves=args.lgbm_num_leaves,
        min_child_samples=args.lgbm_min_child_samples,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=args.lgbm_reg_lambda,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    train_frame = pd.DataFrame(train_x, columns=feature_names)
    valid_frame = pd.DataFrame(valid_x, columns=feature_names)
    lgbm.fit(train_frame, y_train, sample_weight=sample_weight)
    lgbm_pred = lgbm.predict(valid_frame)

    ridge_metrics = score_prediction(y_valid, ridge_pred, w_valid, asset_valid)
    lgbm_metrics = score_prediction(y_valid, lgbm_pred, w_valid, asset_valid)
    blend = find_best_two_model_blend(y_valid, ridge_pred, lgbm_pred, w_valid)
    blend_by_asset = {}
    for asset in asset_ids:
        mask = asset_valid == asset
        blend_by_asset[str(asset)] = weighted_zero_mean_r2(y_valid[mask], blend["prediction"][mask], w_valid[mask])

    predictions = frame.loc[valid_mask, ["row_id", "time_id", "asset_id", "target", "weight"]].copy()
    predictions["ridge_prediction"] = ridge_pred
    predictions["lgbm_prediction"] = lgbm_pred
    predictions["prediction"] = blend["prediction"]
    predictions["error"] = predictions["prediction"] - predictions["target"]
    predictions.to_csv(args.results_dir / "validation_predictions.csv", index=False)

    metrics = {
        "data_dir": str(args.data_dir),
        "selected_features_file": str(args.selected_features_file),
        "selected_feature_count": int(len(selected_features)),
        "use_asset_dummies": bool(args.use_asset_dummies),
        "train_end_time": int(args.train_end_time),
        "valid_start_time": int(args.valid_start_time),
        "valid_end_time": int(args.valid_end_time),
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "ridge": ridge_metrics,
        "lgbm": lgbm_metrics,
        "blend": {
            "score": float(blend["score"]),
            "weights": blend["weights"],
            "score_by_asset": blend_by_asset,
        },
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"feature_name": selected_features}).to_csv(args.results_dir / "selected_features.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
