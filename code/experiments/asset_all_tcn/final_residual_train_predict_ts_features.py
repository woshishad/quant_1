from __future__ import annotations

import argparse
import gc
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from final_train_predict import (
    BASE_COLUMNS_TEST,
    BASE_COLUMNS_TRAIN,
    load_feature_ranking,
    parquet_paths,
    read_partitioned_frame,
    reorder_like_sample,
    schema_columns,
    time_range,
)
from walk_forward_tabular import (
    apply_shrink,
    calibrate_shrink_info,
    score_candidate_on_calibration,
    standardize,
    summarize_shrink_info,
    weighted_zero_mean_r2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ridge + LightGBM residual，并额外加入 asset 内时序 lag/rolling 与 time_id 横截面特征。"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset_all_residual_lgbm_ts_features"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/asset_all_residual_lgbm_ts_features"))
    parser.add_argument(
        "--fixed-features-file",
        type=Path,
        default=Path("results/asset_all_stable_features_100k/selected_features_stable_top128.csv"),
    )
    parser.add_argument("--sample-submission", type=Path, default=None)
    parser.add_argument("--train-lookback-time-points", type=int, default=75_000)
    parser.add_argument("--cal-time-points", type=int, default=20_000)
    parser.add_argument("--max-train-time-id", type=int, default=None)
    parser.add_argument("--test-start-time", type=int, default=None)
    parser.add_argument("--test-end-time", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument(
        "--skip-test-prediction",
        action="store_true",
        help="只跑 calibration 验证，不读取 official test，也不生成 submission；用于快速判断新特征是否有用。",
    )

    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument(
        "--engineered-top-k",
        type=int,
        default=16,
        help="只对排名靠前的 N 个 feature 做 lag/rolling/cross-sectional 扩展，控制内存和过拟合。",
    )
    parser.add_argument("--lag-steps", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--rolling-windows", type=int, nargs="+", default=[20, 60])
    parser.add_argument("--rolling-min-period-frac", type=float, default=0.25)
    parser.add_argument(
        "--market-history-top-k",
        type=int,
        default=8,
        help="对排名靠前的 N 个工程特征构造 15 标的横截面历史，0 表示关闭。",
    )
    parser.add_argument("--disable-lag", action="store_true")
    parser.add_argument("--disable-delta", action="store_true")
    parser.add_argument("--disable-rolling", action="store_true")
    parser.add_argument("--disable-cross-section", action="store_true")
    parser.add_argument("--disable-market-history", action="store_true")

    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--lgbm-num-leaves", type=int, default=31)
    parser.add_argument("--lgbm-estimators", type=int, default=200)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.005)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=4000)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=1000.0)
    parser.add_argument("--lgbm-subsample", type=float, default=0.7)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.7)
    parser.add_argument("--lgbm-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--lgbm-n-jobs", type=int, default=-1)
    parser.add_argument(
        "--ridge-raw-only",
        action="store_true",
        help="Ridge 只使用原始 top-k 特征；LightGBM residual 使用原始+工程特征，避免线性底座被噪声扩展特征破坏。",
    )
    parser.add_argument(
        "--residual-feature-set",
        choices=["all", "raw", "engineered", "historical"],
        default="all",
        help="残差模型使用哪些特征；historical 只保留 lag/delta/rolling。",
    )
    parser.add_argument(
        "--residual-model",
        choices=["lgbm", "ridge", "per_asset_ridge"],
        default="lgbm",
    )
    parser.add_argument("--residual-ridge-alpha", type=float, default=10_000.0)

    parser.add_argument("--residual-weight-min", type=float, default=0.0)
    parser.add_argument("--residual-weight-max", type=float, default=1.0)
    parser.add_argument("--residual-weight-step", type=float, default=0.25)
    parser.add_argument("--shrink-mode", choices=["global", "per_asset"], default="per_asset")
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[1.0, 1.2, 1.4])
    parser.add_argument("--candidate-score-mode", choices=["full", "mean_halves", "min_halves"], default="min_halves")
    parser.add_argument("--no-save-models", action="store_true")
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


def max_feature_history(args: argparse.Namespace) -> int:
    """读取原始数据时往前多读一些，只用于计算早期训练行的 lag/rolling 特征。"""
    lag_max = 0 if args.disable_lag and args.disable_delta else max(args.lag_steps or [0])
    rolling_max = 0 if args.disable_rolling else max(args.rolling_windows or [0])
    return int(max(lag_max, rolling_max))


def rolling_min_periods(window: int, frac: float) -> int:
    return max(2, int(round(float(window) * float(frac))))


def add_time_series_and_cross_section_features(
    frame: pd.DataFrame,
    raw_feature_names: list[str],
    engineered_feature_names: list[str],
    market_history_feature_names: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[str]]:
    """
    构造新特征。

    约束：
    - asset 内 lag/rolling 只沿时间向过去看，不使用 target。
    - lag 必须满足精确的 time_id 间隔；缺失时点不会被误当成连续历史。
    - 横截面特征只使用同一个 time_id 已观测到的 feature 分布，不使用未来 target。
    - 市场历史先聚合同一 time_id 的全部标的，再严格向过去 shift。
    - 返回时重新按 time_id/asset_id 排序，保证后续 mask 和输出稳定。
    """
    working = frame.sort_values(["asset_id", "time_id"], kind="mergesort").reset_index(drop=True)
    new_columns: dict[str, pd.Series | np.ndarray] = {}
    created_features: list[str] = []
    asset_group = working.groupby("asset_id", sort=False)
    current_time = working["time_id"].astype(np.int64)

    for feature_name in engineered_feature_names:
        current = working[feature_name].astype(np.float32)

        lag_cache: dict[int, pd.Series] = {}
        if not args.disable_lag or not args.disable_delta:
            for lag in args.lag_steps:
                lag_value = asset_group[feature_name].shift(int(lag)).astype(np.float32)
                lag_time = asset_group["time_id"].shift(int(lag))
                lag_value = lag_value.where(
                    (current_time - lag_time) == int(lag)
                )
                lag_cache[int(lag)] = lag_value
                if not args.disable_lag:
                    name = f"{feature_name}_lag{int(lag)}"
                    new_columns[name] = lag_value
                    created_features.append(name)
                if not args.disable_delta:
                    name = f"{feature_name}_delta{int(lag)}"
                    new_columns[name] = (current - lag_value).astype(np.float32)
                    created_features.append(name)

        if not args.disable_rolling:
            # rolling 使用 shift(1) 后的过去值，避免把当前行直接混入滚动统计。
            past = asset_group[feature_name].shift(1).astype(np.float32)
            past_group = past.groupby(working["asset_id"], sort=False)
            for window in args.rolling_windows:
                window = int(window)
                min_periods = rolling_min_periods(window, args.rolling_min_period_frac)
                rolling = past_group.rolling(window=window, min_periods=min_periods)
                mean_value = rolling.mean().reset_index(level=0, drop=True).astype(np.float32)
                std_value = rolling.std().reset_index(level=0, drop=True).astype(np.float32)

                mean_name = f"{feature_name}_rollmean{window}"
                std_name = f"{feature_name}_rollstd{window}"
                dev_name = f"{feature_name}_rolldev{window}"
                new_columns[mean_name] = mean_value
                new_columns[std_name] = std_value
                new_columns[dev_name] = (current - mean_value).astype(np.float32)
                created_features.extend([mean_name, std_name, dev_name])

        if not args.disable_cross_section:
            time_group = working.groupby("time_id", sort=False)[feature_name]
            mean_value = time_group.transform("mean").astype(np.float32)
            std_value = time_group.transform("std").replace(0.0, np.nan).astype(np.float32)
            rank_value = time_group.rank(method="average", pct=True).astype(np.float32) - np.float32(0.5)

            z_name = f"{feature_name}_cs_z"
            demean_name = f"{feature_name}_cs_demean"
            rank_name = f"{feature_name}_cs_rank"
            new_columns[z_name] = ((current - mean_value) / std_value).astype(np.float32)
            new_columns[demean_name] = (current - mean_value).astype(np.float32)
            new_columns[rank_name] = rank_value.astype(np.float32)
            created_features.extend([z_name, demean_name, rank_name])

    if market_history_feature_names and not args.disable_market_history:
        market_frame = (
            working.groupby("time_id", sort=True)[market_history_feature_names]
            .agg(["mean", "std"])
            .sort_index()
        )
        market_time = market_frame.index.to_series().astype(np.int64)
        for feature_name in market_history_feature_names:
            market_mean = market_frame[(feature_name, "mean")].astype(np.float32)
            market_std = market_frame[(feature_name, "std")].astype(np.float32)

            mean_name = f"{feature_name}_market_mean"
            std_name = f"{feature_name}_market_std"
            new_columns[mean_name] = current_time.map(market_mean).astype(np.float32)
            new_columns[std_name] = current_time.map(market_std).astype(np.float32)
            created_features.extend([mean_name, std_name])

            if not args.disable_lag or not args.disable_delta:
                for lag in args.lag_steps:
                    lag = int(lag)
                    lagged_mean = market_mean.shift(lag)
                    lagged_time = market_time.shift(lag)
                    lagged_mean = lagged_mean.where(
                        (market_time - lagged_time) == lag
                    )
                    if not args.disable_lag:
                        name = f"{feature_name}_market_mean_lag{lag}"
                        new_columns[name] = current_time.map(lagged_mean).astype(
                            np.float32
                        )
                        created_features.append(name)
                    if not args.disable_delta:
                        name = f"{feature_name}_market_mean_delta{lag}"
                        market_delta = market_mean - lagged_mean
                        new_columns[name] = current_time.map(market_delta).astype(
                            np.float32
                        )
                        created_features.append(name)

            if not args.disable_rolling:
                past_market_mean = market_mean.shift(1)
                for window in args.rolling_windows:
                    window = int(window)
                    min_periods = rolling_min_periods(
                        window, args.rolling_min_period_frac
                    )
                    rolling = past_market_mean.rolling(
                        window=window, min_periods=min_periods
                    )
                    rolling_mean = rolling.mean().astype(np.float32)
                    rolling_std = rolling.std().astype(np.float32)
                    rolling_dev = (market_mean - rolling_mean).astype(np.float32)
                    values = {
                        f"{feature_name}_market_rollmean{window}": rolling_mean,
                        f"{feature_name}_market_rollstd{window}": rolling_std,
                        f"{feature_name}_market_rolldev{window}": rolling_dev,
                    }
                    for name, series in values.items():
                        new_columns[name] = current_time.map(series).astype(np.float32)
                        created_features.append(name)

    if new_columns:
        engineered_frame = pd.DataFrame(new_columns, index=working.index)
        working = pd.concat([working, engineered_frame], axis=1)

    all_model_features = raw_feature_names + created_features
    working = working.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    return working, all_model_features


def fit_ridge(train_x: np.ndarray, y_train: np.ndarray, w_train: np.ndarray, alpha: float) -> Ridge:
    """Ridge 学低方差线性底座，减少残差模型需要解释的目标噪声。"""
    sample_weight = w_train / max(float(np.mean(w_train)), 1e-12)
    model = Ridge(alpha=float(alpha), solver="lsqr", max_iter=500)
    model.fit(train_x, y_train, sample_weight=sample_weight)
    return model


def fit_lgbm_residual_models(
    train_x: np.ndarray,
    residual_train: np.ndarray,
    w_train: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
) -> list[LGBMRegressor]:
    """LightGBM 只学习 Ridge 残差；多 seed 平均可以降低随机方差。"""
    sample_weight = w_train / max(float(np.mean(w_train)), 1e-12)
    train_frame = pd.DataFrame(train_x, columns=feature_names)
    models = []
    for seed in args.lgbm_seeds:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=int(args.lgbm_estimators),
            learning_rate=float(args.lgbm_learning_rate),
            num_leaves=int(args.lgbm_num_leaves),
            min_child_samples=int(args.lgbm_min_child_samples),
            subsample=float(args.lgbm_subsample),
            subsample_freq=1,
            colsample_bytree=float(args.lgbm_colsample_bytree),
            reg_lambda=float(args.lgbm_reg_lambda),
            random_state=int(seed),
            n_jobs=int(args.lgbm_n_jobs),
            verbose=-1,
        )
        model.fit(train_frame, residual_train, sample_weight=sample_weight)
        models.append(model)
    return models


def predict_lgbm_average(models: list[LGBMRegressor], predict_x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    predict_frame = pd.DataFrame(predict_x, columns=feature_names)
    predictions = [model.predict(predict_frame) for model in models]
    return np.mean(np.vstack(predictions), axis=0)


def historical_feature_indices(
    feature_names: list[str], ridge_feature_count: int
) -> list[int]:
    markers = ("_lag", "_delta", "_roll")
    return [
        index
        for index, name in enumerate(feature_names)
        if index >= ridge_feature_count and any(marker in name for marker in markers)
    ]


def fit_predict_residual_protocol(
    train_x_raw: np.ndarray,
    predict_x_raw: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    feature_names: list[str],
    ridge_feature_count: int,
    args: argparse.Namespace,
    train_asset_id: np.ndarray | None = None,
    predict_asset_id: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Ridge, list[object]]:
    """训练当前值 Ridge，再用所选模型预测剩余残差。"""
    train_x, predict_x, _, _ = standardize(train_x_raw, predict_x_raw)
    if args.ridge_raw_only:
        # Ridge 保持低方差，只看原始特征；残差模型再去吃高维时序/横截面扩展。
        ridge_train_x = train_x[:, :ridge_feature_count]
        ridge_predict_x = predict_x[:, :ridge_feature_count]
    else:
        ridge_train_x = train_x
        ridge_predict_x = predict_x
    ridge_model = fit_ridge(ridge_train_x, y_train, w_train, args.ridge_alpha)
    ridge_train = ridge_model.predict(ridge_train_x)
    ridge_predict = ridge_model.predict(ridge_predict_x)
    residual_train = y_train - ridge_train
    if args.residual_feature_set == "raw":
        residual_train_x = train_x[:, :ridge_feature_count]
        residual_predict_x = predict_x[:, :ridge_feature_count]
        residual_feature_names = feature_names[:ridge_feature_count]
    elif args.residual_feature_set == "engineered":
        if len(feature_names) <= ridge_feature_count:
            raise ValueError("--residual-feature-set engineered 需要至少一个工程特征")
        residual_train_x = train_x[:, ridge_feature_count:]
        residual_predict_x = predict_x[:, ridge_feature_count:]
        residual_feature_names = feature_names[ridge_feature_count:]
    elif args.residual_feature_set == "historical":
        indices = historical_feature_indices(feature_names, ridge_feature_count)
        if not indices:
            raise ValueError(
                "--residual-feature-set historical 需要 lag/delta/rolling 特征"
            )
        residual_train_x = train_x[:, indices]
        residual_predict_x = predict_x[:, indices]
        residual_feature_names = [feature_names[index] for index in indices]
    else:
        residual_train_x = train_x
        residual_predict_x = predict_x
        residual_feature_names = feature_names
    if args.residual_model == "per_asset_ridge":
        if train_asset_id is None or predict_asset_id is None:
            raise ValueError("per_asset_ridge 需要 train/predict asset_id")
        residual_predict = np.zeros(len(residual_predict_x), dtype=np.float64)
        residual_models = []
        for asset in sorted(np.unique(train_asset_id)):
            train_mask = train_asset_id == asset
            predict_mask = predict_asset_id == asset
            if not np.any(predict_mask):
                continue
            model = fit_ridge(
                residual_train_x[train_mask],
                residual_train[train_mask],
                w_train[train_mask],
                args.residual_ridge_alpha,
            )
            residual_predict[predict_mask] = model.predict(
                residual_predict_x[predict_mask]
            )
            residual_models.append({"asset_id": int(asset), "model": model})
    elif args.residual_model == "ridge":
        residual_model = fit_ridge(
            residual_train_x,
            residual_train,
            w_train,
            args.residual_ridge_alpha,
        )
        residual_predict = residual_model.predict(residual_predict_x)
        residual_models: list[object] = [residual_model]
    else:
        residual_models = fit_lgbm_residual_models(
            residual_train_x,
            residual_train,
            w_train,
            residual_feature_names,
            args,
        )
        residual_predict = predict_lgbm_average(
            residual_models, residual_predict_x, residual_feature_names
        )
    return ridge_predict, residual_predict, ridge_model, residual_models


def search_residual_weight_and_shrink(
    y_cal: np.ndarray,
    w_cal: np.ndarray,
    asset_cal: np.ndarray,
    time_cal: np.ndarray,
    ridge_cal: np.ndarray,
    residual_cal: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """在 calibration 上选择 residual 权重和 shrink。"""
    best = {"score": -np.inf}
    residual_weights = np.arange(
        args.residual_weight_min,
        args.residual_weight_max + 1e-12,
        args.residual_weight_step,
    )
    for residual_weight in residual_weights:
        raw_prediction = ridge_cal + float(residual_weight) * residual_cal
        for cap in args.shrink_cap_candidates:
            shrink_info = calibrate_shrink_info(
                y_cal,
                raw_prediction,
                w_cal,
                asset_cal,
                args.shrink_mode,
                float(cap),
            )
            prediction = apply_shrink(raw_prediction, asset_cal, shrink_info)
            score_info = score_candidate_on_calibration(
                y_cal,
                prediction,
                w_cal,
                time_cal,
                args.candidate_score_mode,
            )
            if score_info["selection_score"] > best["score"]:
                best = {
                    "score": float(score_info["selection_score"]),
                    "score_info": score_info,
                    "residual_weight": float(residual_weight),
                    "shrink_info": shrink_info,
                    "shrink_summary": summarize_shrink_info(shrink_info),
                    "raw_prediction": raw_prediction,
                    "prediction": prediction,
                }
    return best


def save_zip(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=csv_path.name)


def save_pickle(path: Path, payload: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def prediction_stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "null_count": int(np.sum(~np.isfinite(values))),
        "finite_count": int(np.sum(np.isfinite(values))),
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_submission is None:
        args.sample_submission = args.raw_data_dir / "sample_submission.csv"

    train_paths = parquet_paths(args.raw_data_dir, "train")
    test_paths = parquet_paths(args.raw_data_dir, "test")
    train_min_time, train_max_time_available = time_range(train_paths)
    test_min_time, test_max_time_available = time_range(test_paths)
    train_end_time = (
        min(train_max_time_available, int(args.max_train_time_id))
        if args.max_train_time_id is not None
        else train_max_time_available
    )
    train_start_time = max(train_min_time, train_end_time - int(args.train_lookback_time_points) + 1)
    feature_history_start_time = max(train_min_time, train_start_time - max_feature_history(args))
    fit_train_end_time = train_end_time - int(args.cal_time_points)
    cal_start_time = fit_train_end_time + 1
    if fit_train_end_time < train_start_time:
        raise ValueError("cal-time-points 对当前 lookback 来说过大")

    available_columns = schema_columns(train_paths)
    ranking = load_feature_ranking(args.fixed_features_file, available_columns)
    raw_features = ranking.head(int(args.top_k))["feature_name"].astype(str).tolist()
    engineered_base_features = ranking.head(int(args.engineered_top_k))["feature_name"].astype(str).tolist()
    market_history_features = engineered_base_features[
        : max(0, int(args.market_history_top_k))
    ]

    print(
        f"TS-feature residual split: fit={train_start_time}..{fit_train_end_time}, "
        f"cal={cal_start_time}..{train_end_time}, feature_history_start={feature_history_start_time}"
    )
    print(
        f"raw_features={len(raw_features)}, engineered_base_features={len(engineered_base_features)}, "
        f"market_history_features={len(market_history_features)}, "
        f"lags={args.lag_steps}, rolling={args.rolling_windows}"
    )

    train_raw = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + raw_features,
        min_time=feature_history_start_time,
        max_time=train_end_time,
    )
    train_frame, model_features = add_time_series_and_cross_section_features(
        train_raw,
        raw_features,
        engineered_base_features,
        market_history_features,
        args,
    )
    del train_raw
    gc.collect()

    train_time = train_frame["time_id"].to_numpy(dtype=np.int64)
    fit_mask = (train_time >= train_start_time) & (train_time <= fit_train_end_time)
    cal_mask = (train_time >= cal_start_time) & (train_time <= train_end_time)
    final_train_mask = (train_time >= train_start_time) & (train_time <= train_end_time)

    fit_x_raw = train_frame.loc[fit_mask, model_features].to_numpy(dtype=np.float32)
    cal_x_raw = train_frame.loc[cal_mask, model_features].to_numpy(dtype=np.float32)
    y_fit = train_frame.loc[fit_mask, "target"].to_numpy(dtype=np.float32)
    w_fit = train_frame.loc[fit_mask, "weight"].to_numpy(dtype=np.float32)
    y_cal = train_frame.loc[cal_mask, "target"].to_numpy(dtype=np.float32)
    w_cal = train_frame.loc[cal_mask, "weight"].to_numpy(dtype=np.float32)
    asset_fit = train_frame.loc[fit_mask, "asset_id"].to_numpy(dtype=np.int64)
    asset_cal = train_frame.loc[cal_mask, "asset_id"].to_numpy(dtype=np.int64)
    time_cal = train_frame.loc[cal_mask, "time_id"].to_numpy(dtype=np.int64)

    ridge_cal, residual_cal, _, _ = fit_predict_residual_protocol(
        fit_x_raw,
        cal_x_raw,
        y_fit,
        w_fit,
        model_features,
        len(raw_features),
        args,
        asset_fit,
        asset_cal,
    )
    best = search_residual_weight_and_shrink(
        y_cal,
        w_cal,
        asset_cal,
        time_cal,
        ridge_cal,
        residual_cal,
        args,
    )

    cal_predictions = train_frame.loc[cal_mask, BASE_COLUMNS_TRAIN].copy()
    cal_predictions["ridge_prediction"] = ridge_cal
    cal_predictions["residual_prediction"] = residual_cal
    cal_predictions["raw_prediction"] = best["raw_prediction"]
    cal_predictions["prediction"] = best["prediction"]
    cal_predictions["error"] = cal_predictions["prediction"] - cal_predictions["target"]
    cal_predictions.to_csv(args.results_dir / "calibration_predictions.csv", index=False)

    output_files = {
        "calibration_predictions": str(args.results_dir / "calibration_predictions.csv"),
    }
    test_stats = None
    submission_path = None
    zip_path = None
    trained_payload = None

    if not args.skip_test_prediction:
        test_min = args.test_start_time if args.test_start_time is not None else test_min_time
        test_max = args.test_end_time if args.test_end_time is not None else test_max_time_available
        test_raw = read_partitioned_frame(
            test_paths,
            BASE_COLUMNS_TEST + raw_features,
            min_time=test_min,
            max_time=test_max,
            max_rows=args.max_test_rows,
        )
        test_for_prediction = test_raw.copy()
        test_for_prediction["weight"] = 0.0
        test_for_prediction["target"] = 0.0
        combined_raw = pd.concat(
            [
                train_frame.loc[final_train_mask, BASE_COLUMNS_TRAIN + raw_features],
                test_for_prediction[BASE_COLUMNS_TRAIN + raw_features],
            ],
            ignore_index=True,
        )
        del test_raw, test_for_prediction
        gc.collect()

        combined_frame, _ = add_time_series_and_cross_section_features(
            combined_raw,
            raw_features,
            engineered_base_features,
            market_history_features,
            args,
        )
        del combined_raw
        gc.collect()

        combined_time = combined_frame["time_id"].to_numpy(dtype=np.int64)
        combined_train_mask = (combined_time >= train_start_time) & (combined_time <= train_end_time)
        combined_test_mask = combined_time > train_end_time
        final_x_raw = combined_frame.loc[combined_train_mask, model_features].to_numpy(dtype=np.float32)
        test_x_raw = combined_frame.loc[combined_test_mask, model_features].to_numpy(dtype=np.float32)
        y_final = combined_frame.loc[combined_train_mask, "target"].to_numpy(dtype=np.float32)
        w_final = combined_frame.loc[combined_train_mask, "weight"].to_numpy(dtype=np.float32)
        asset_final = combined_frame.loc[combined_train_mask, "asset_id"].to_numpy(dtype=np.int64)
        asset_test = combined_frame.loc[combined_test_mask, "asset_id"].to_numpy(dtype=np.int64)

        ridge_test, residual_test, ridge_model, lgbm_models = fit_predict_residual_protocol(
            final_x_raw,
            test_x_raw,
            y_final,
            w_final,
            model_features,
            len(raw_features),
            args,
            asset_final,
            asset_test,
        )
        raw_test = ridge_test + float(best["residual_weight"]) * residual_test
        prediction_test = apply_shrink(raw_test, asset_test, best["shrink_info"])

        test_predictions = combined_frame.loc[combined_test_mask, BASE_COLUMNS_TEST].copy()
        test_predictions["ridge_prediction"] = ridge_test
        test_predictions["residual_prediction"] = residual_test
        test_predictions["raw_prediction"] = raw_test
        test_predictions["prediction"] = prediction_test
        test_predictions.to_csv(args.results_dir / "final_test_predictions.csv", index=False)
        submission = test_predictions[["row_id", "prediction"]].rename(columns={"prediction": "target"})
        submission = reorder_like_sample(submission, args.sample_submission)
        submission_path = args.results_dir / "submission.csv"
        zip_path = args.results_dir / "submission.zip"
        submission.to_csv(submission_path, index=False)
        save_zip(submission_path, zip_path)
        test_stats = prediction_stats(prediction_test)
        output_files.update(
            {
                "final_test_predictions": str(args.results_dir / "final_test_predictions.csv"),
                "submission": str(submission_path),
                "submission_zip": str(zip_path),
            }
        )
        trained_payload = {
            "ridge_model": ridge_model,
            "lgbm_models": lgbm_models,
            "model_features": model_features,
            "raw_features": raw_features,
            "engineered_base_features": engineered_base_features,
            "market_history_features": market_history_features,
            "calibration": best,
        }

    metrics = {
        "leakage_safe": True,
        "official_test_used_for_training": False,
        "environment": "quant-competition-wsl",
        "raw_data_dir": str(args.raw_data_dir),
        "future_function_guard": {
            "fit_train": f"{train_start_time} <= time_id <= {fit_train_end_time}",
            "calibration": f"{cal_start_time} <= time_id <= {train_end_time}",
            "official_test_prediction_only": None
            if args.skip_test_prediction
            else f"{args.test_start_time or test_min_time} <= time_id <= {args.test_end_time or test_max_time_available}",
            "feature_history_only": f"{feature_history_start_time} <= time_id < {train_start_time}",
        },
        "train_window": {
            "raw_train_min_time": int(train_min_time),
            "raw_train_max_time": int(train_max_time_available),
            "used_train_start_time": int(train_start_time),
            "used_train_end_time": int(train_end_time),
            "feature_history_start_time": int(feature_history_start_time),
            "train_lookback_time_points": int(args.train_lookback_time_points),
        },
        "rows": {
            "fit_train": int(fit_mask.sum()),
            "calibration": int(cal_mask.sum()),
            "final_train": int(final_train_mask.sum()),
            "test_predicted": None if args.skip_test_prediction else int(test_stats["finite_count"]),
        },
        "feature_engineering": {
            "raw_feature_count": int(len(raw_features)),
            "engineered_base_feature_count": int(len(engineered_base_features)),
            "market_history_feature_count": int(len(market_history_features)),
            "model_feature_count": int(len(model_features)),
            "lag_steps": [int(value) for value in args.lag_steps],
            "rolling_windows": [int(value) for value in args.rolling_windows],
            "rolling_min_period_frac": float(args.rolling_min_period_frac),
            "disable_lag": bool(args.disable_lag),
            "disable_delta": bool(args.disable_delta),
            "disable_rolling": bool(args.disable_rolling),
            "disable_cross_section": bool(args.disable_cross_section),
            "disable_market_history": bool(args.disable_market_history),
        },
        "model": {
            "base_model": "Ridge",
            "residual_model": args.residual_model,
            "residual_ridge_alpha": float(args.residual_ridge_alpha),
            "ridge_alpha": float(args.ridge_alpha),
            "lgbm_num_leaves": int(args.lgbm_num_leaves),
            "lgbm_estimators": int(args.lgbm_estimators),
            "lgbm_learning_rate": float(args.lgbm_learning_rate),
            "lgbm_min_child_samples": int(args.lgbm_min_child_samples),
            "lgbm_reg_lambda": float(args.lgbm_reg_lambda),
            "lgbm_seeds": [int(seed) for seed in args.lgbm_seeds],
            "ridge_raw_only": bool(args.ridge_raw_only),
            "residual_feature_set": args.residual_feature_set,
        },
        "calibration": {
            "candidate_score_mode": args.candidate_score_mode,
            "selection_score": float(best["score"]),
            "full_score": float(best["score_info"]["full_score"]),
            "first_half_score": float(best["score_info"]["first_half_score"]),
            "second_half_score": float(best["score_info"]["second_half_score"]),
            "residual_weight": float(best["residual_weight"]),
            "shrink_info": best["shrink_info"],
            "shrink_summary": best["shrink_summary"],
            "ridge_only_raw_score": float(weighted_zero_mean_r2(y_cal, ridge_cal, w_cal)),
            "residual_raw_score": float(weighted_zero_mean_r2(y_cal - ridge_cal, residual_cal, w_cal)),
        },
        "test_prediction_stats": test_stats,
        "output_files": output_files,
    }
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    (args.model_dir / "metadata.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    pd.DataFrame({"feature_name": model_features}).to_csv(args.results_dir / "model_features.csv", index=False)
    if trained_payload is not None and not args.no_save_models:
        save_pickle(args.model_dir / "residual_ts_feature_models.pkl", trained_payload)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
