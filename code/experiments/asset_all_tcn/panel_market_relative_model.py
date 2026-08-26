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
        description="面板模型：先预测 time_id 的市场共同 target，再预测每个 asset 的相对偏离。"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset_all_panel_market_relative"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/asset_all_panel_market_relative"))
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
    parser.add_argument("--skip-test-prediction", action="store_true")

    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument(
        "--cs-top-k",
        type=int,
        default=16,
        help="relative head 额外使用前 N 个特征的横截面 demean/z/rank。",
    )
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--market-ridge-alpha", type=float, default=100.0)
    parser.add_argument("--relative-ridge-alpha", type=float, default=100.0)

    parser.add_argument("--market-lgbm-num-leaves", type=int, default=15)
    parser.add_argument("--market-lgbm-estimators", type=int, default=300)
    parser.add_argument("--market-lgbm-learning-rate", type=float, default=0.01)
    parser.add_argument("--market-lgbm-min-child-samples", type=int, default=200)
    parser.add_argument("--market-lgbm-reg-lambda", type=float, default=300.0)

    parser.add_argument("--relative-lgbm-num-leaves", type=int, default=31)
    parser.add_argument("--relative-lgbm-estimators", type=int, default=200)
    parser.add_argument("--relative-lgbm-learning-rate", type=float, default=0.005)
    parser.add_argument("--relative-lgbm-min-child-samples", type=int, default=4000)
    parser.add_argument("--relative-lgbm-reg-lambda", type=float, default=1000.0)
    parser.add_argument("--lgbm-subsample", type=float, default=0.7)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.7)
    parser.add_argument("--lgbm-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--lgbm-n-jobs", type=int, default=-1)

    parser.add_argument("--market-weight-min", type=float, default=-1.0)
    parser.add_argument("--market-weight-max", type=float, default=3.0)
    parser.add_argument("--market-weight-step", type=float, default=0.25)
    parser.add_argument("--relative-weight-min", type=float, default=0.0)
    parser.add_argument("--relative-weight-max", type=float, default=2.0)
    parser.add_argument("--relative-weight-step", type=float, default=0.25)
    parser.add_argument("--shrink-mode", choices=["global", "per_asset"], default="per_asset")
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[0.5, 0.8, 1.0, 1.2])
    parser.add_argument("--candidate-score-mode", choices=["full", "mean_halves", "min_halves"], default="full")
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


def save_zip(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=csv_path.name)


def save_pickle(path: Path, payload: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def weighted_market_target(frame: pd.DataFrame) -> pd.DataFrame:
    """每个 time_id 的加权 target 均值，作为 market head 的训练标签。"""
    values = frame[["time_id", "target", "weight"]].copy()
    values["weighted_target"] = values["target"] * values["weight"]
    grouped = values.groupby("time_id", sort=True)
    out = grouped.agg(
        weighted_target_sum=("weighted_target", "sum"),
        weight_sum=("weight", "sum"),
        target_mean=("target", "mean"),
        row_count=("target", "size"),
    ).reset_index()
    out["market_target"] = out["weighted_target_sum"] / out["weight_sum"].replace(0.0, np.nan)
    out["market_target"] = out["market_target"].fillna(out["target_mean"]).astype(np.float32)
    return out[["time_id", "market_target", "target_mean", "weight_sum", "row_count"]]


def make_time_features(frame: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """把一个 time_id 的 15 个 asset 压成一行市场状态特征。"""
    grouped = frame.groupby("time_id", sort=True)[feature_names]
    mean_frame = grouped.mean().add_suffix("_tmean")
    std_frame = grouped.std().fillna(0.0).add_suffix("_tstd")
    min_frame = grouped.min().add_suffix("_tmin")
    max_frame = grouped.max().add_suffix("_tmax")
    out = pd.concat([mean_frame, std_frame, min_frame, max_frame], axis=1).reset_index()
    feature_columns = [column for column in out.columns if column != "time_id"]
    return out, feature_columns


def make_relative_features(
    frame: pd.DataFrame,
    raw_features: list[str],
    cs_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """逐行 relative head 特征：原始特征 + asset_id + 当前 time_id 横截面相对位置。"""
    working = frame.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True).copy()
    new_columns: dict[str, np.ndarray | pd.Series] = {
        "asset_id_float": working["asset_id"].astype(np.float32),
    }
    created = ["asset_id_float"]
    time_group = working.groupby("time_id", sort=False)
    for feature in cs_features:
        current = working[feature].astype(np.float32)
        mean_value = time_group[feature].transform("mean").astype(np.float32)
        std_value = time_group[feature].transform("std").replace(0.0, np.nan).astype(np.float32)
        rank_value = time_group[feature].rank(method="average", pct=True).astype(np.float32) - np.float32(0.5)
        demean_name = f"{feature}_panel_demean"
        z_name = f"{feature}_panel_z"
        rank_name = f"{feature}_panel_rank"
        new_columns[demean_name] = (current - mean_value).astype(np.float32)
        new_columns[z_name] = ((current - mean_value) / std_value).astype(np.float32)
        new_columns[rank_name] = rank_value.astype(np.float32)
        created.extend([demean_name, z_name, rank_name])
    if new_columns:
        working = pd.concat([working, pd.DataFrame(new_columns, index=working.index)], axis=1)
    feature_columns = raw_features + created
    return working, feature_columns


def fit_ridge(train_x: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=float(alpha), solver="lsqr", max_iter=500)
    normalized_weight = sample_weight / max(float(np.mean(sample_weight)), 1e-12)
    model.fit(train_x, y_train, sample_weight=normalized_weight)
    return model


def fit_lgbm_average(
    train_x: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    predict_x: np.ndarray,
    feature_names: list[str],
    *,
    num_leaves: int,
    estimators: int,
    learning_rate: float,
    min_child_samples: int,
    reg_lambda: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[LGBMRegressor]]:
    normalized_weight = sample_weight / max(float(np.mean(sample_weight)), 1e-12)
    train_frame = pd.DataFrame(train_x, columns=feature_names)
    predict_frame = pd.DataFrame(predict_x, columns=feature_names)
    models: list[LGBMRegressor] = []
    predictions = []
    for seed in args.lgbm_seeds:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=int(estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            min_child_samples=int(min_child_samples),
            subsample=float(args.lgbm_subsample),
            subsample_freq=1,
            colsample_bytree=float(args.lgbm_colsample_bytree),
            reg_lambda=float(reg_lambda),
            random_state=int(seed),
            n_jobs=int(args.lgbm_n_jobs),
            verbose=-1,
        )
        model.fit(train_frame, y_train, sample_weight=normalized_weight)
        models.append(model)
        predictions.append(model.predict(predict_frame))
    return np.mean(np.vstack(predictions), axis=0), models


def fit_predict_market(
    fit_time: pd.DataFrame,
    predict_time: pd.DataFrame,
    time_feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, Ridge, list[LGBMRegressor]]:
    """market head：用 time-level 聚合特征预测当期横截面 target 均值。"""
    fit_x_raw = fit_time[time_feature_columns].to_numpy(dtype=np.float32)
    predict_x_raw = predict_time[time_feature_columns].to_numpy(dtype=np.float32)
    fit_x, predict_x, _, _ = standardize(fit_x_raw, predict_x_raw)
    y_fit = fit_time["market_target"].to_numpy(dtype=np.float32)
    w_fit = fit_time["weight_sum"].to_numpy(dtype=np.float32)

    ridge = fit_ridge(fit_x, y_fit, w_fit, args.market_ridge_alpha)
    ridge_predict = ridge.predict(predict_x)
    lgbm_predict, lgbm_models = fit_lgbm_average(
        fit_x,
        y_fit,
        w_fit,
        predict_x,
        time_feature_columns,
        num_leaves=args.market_lgbm_num_leaves,
        estimators=args.market_lgbm_estimators,
        learning_rate=args.market_lgbm_learning_rate,
        min_child_samples=args.market_lgbm_min_child_samples,
        reg_lambda=args.market_lgbm_reg_lambda,
        args=args,
    )
    return ridge_predict, lgbm_predict, ridge, lgbm_models


def fit_predict_relative(
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    row_feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, Ridge, list[LGBMRegressor]]:
    """relative head：预测 target - market_target 的 asset 相对偏离。"""
    fit_x_raw = fit_frame[row_feature_columns].to_numpy(dtype=np.float32)
    predict_x_raw = predict_frame[row_feature_columns].to_numpy(dtype=np.float32)
    fit_x, predict_x, _, _ = standardize(fit_x_raw, predict_x_raw)
    y_fit = fit_frame["relative_target"].to_numpy(dtype=np.float32)
    w_fit = fit_frame["weight"].to_numpy(dtype=np.float32)

    ridge = fit_ridge(fit_x, y_fit, w_fit, args.relative_ridge_alpha)
    ridge_fit = ridge.predict(fit_x)
    ridge_predict = ridge.predict(predict_x)
    residual_fit = y_fit - ridge_fit
    lgbm_residual_predict, lgbm_models = fit_lgbm_average(
        fit_x,
        residual_fit,
        w_fit,
        predict_x,
        row_feature_columns,
        num_leaves=args.relative_lgbm_num_leaves,
        estimators=args.relative_lgbm_estimators,
        learning_rate=args.relative_lgbm_learning_rate,
        min_child_samples=args.relative_lgbm_min_child_samples,
        reg_lambda=args.relative_lgbm_reg_lambda,
        args=args,
    )
    return ridge_predict, lgbm_residual_predict, ridge, lgbm_models


def map_time_prediction(time_id: pd.Series, time_prediction_frame: pd.DataFrame, column: str) -> np.ndarray:
    mapping = time_prediction_frame.set_index("time_id")[column]
    return time_id.map(mapping).to_numpy(dtype=np.float64)


def search_panel_blend(
    y_cal: np.ndarray,
    w_cal: np.ndarray,
    asset_cal: np.ndarray,
    time_cal: np.ndarray,
    market_cal: np.ndarray,
    relative_cal: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """在 calibration 上搜索 market/relative 权重和 shrink。"""
    best = {"score": -np.inf}
    market_weights = np.arange(args.market_weight_min, args.market_weight_max + 1e-12, args.market_weight_step)
    relative_weights = np.arange(args.relative_weight_min, args.relative_weight_max + 1e-12, args.relative_weight_step)
    for market_weight in market_weights:
        for relative_weight in relative_weights:
            raw_prediction = float(market_weight) * market_cal + float(relative_weight) * relative_cal
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
                        "market_weight": float(market_weight),
                        "relative_weight": float(relative_weight),
                        "shrink_info": shrink_info,
                        "shrink_summary": summarize_shrink_info(shrink_info),
                        "raw_prediction": raw_prediction,
                        "prediction": prediction,
                    }
    return best


def score_time_blocks(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    block_count: int,
) -> dict[str, float]:
    values = []
    for chunk in np.array_split(np.unique(time_id), block_count):
        mask = (time_id >= int(chunk[0])) & (time_id <= int(chunk[-1]))
        values.append(float(weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])))
    scores = np.asarray(values, dtype=np.float64)
    return {
        f"block{block_count}_mean": float(np.mean(scores)),
        f"block{block_count}_min": float(np.min(scores)),
        f"block{block_count}_negative_count": int(np.sum(scores < 0.0)),
    }


def prediction_stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "null_count": int(np.sum(~np.isfinite(values))),
        "finite_count": int(np.sum(np.isfinite(values))),
    }


def build_training_frames(
    raw_frame: pd.DataFrame,
    raw_features: list[str],
    cs_features: list[str],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, list[str]]:
    market_targets = weighted_market_target(raw_frame)
    time_features, time_feature_columns = make_time_features(raw_frame, raw_features)
    time_frame = time_features.merge(market_targets, on="time_id", how="left")

    row_frame, row_feature_columns = make_relative_features(raw_frame, raw_features, cs_features)
    row_frame = row_frame.merge(market_targets[["time_id", "market_target"]], on="time_id", how="left")
    row_frame["relative_target"] = (row_frame["target"] - row_frame["market_target"]).astype(np.float32)
    return row_frame, row_feature_columns, time_frame, time_feature_columns


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
    fit_train_end_time = train_end_time - int(args.cal_time_points)
    cal_start_time = fit_train_end_time + 1
    if fit_train_end_time < train_start_time:
        raise ValueError("cal-time-points 对当前 lookback 来说过大")

    available_columns = schema_columns(train_paths)
    ranking = load_feature_ranking(args.fixed_features_file, available_columns)
    raw_features = ranking.head(int(args.top_k))["feature_name"].astype(str).tolist()
    cs_features = ranking.head(int(args.cs_top_k))["feature_name"].astype(str).tolist()
    print(
        f"Panel split: fit={train_start_time}..{fit_train_end_time}, "
        f"cal={cal_start_time}..{train_end_time}; raw_features={len(raw_features)}, cs_features={len(cs_features)}"
    )

    raw_train = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + raw_features,
        min_time=train_start_time,
        max_time=train_end_time,
    )
    row_frame, row_feature_columns, time_frame, time_feature_columns = build_training_frames(
        raw_train,
        raw_features,
        cs_features,
    )
    del raw_train
    gc.collect()

    row_time = row_frame["time_id"].to_numpy(dtype=np.int64)
    fit_row_mask = (row_time >= train_start_time) & (row_time <= fit_train_end_time)
    cal_row_mask = (row_time >= cal_start_time) & (row_time <= train_end_time)
    final_row_mask = (row_time >= train_start_time) & (row_time <= train_end_time)

    time_values = time_frame["time_id"].to_numpy(dtype=np.int64)
    fit_time_mask = (time_values >= train_start_time) & (time_values <= fit_train_end_time)
    cal_time_mask = (time_values >= cal_start_time) & (time_values <= train_end_time)

    market_ridge_cal, market_lgbm_cal, _, _ = fit_predict_market(
        time_frame.loc[fit_time_mask].copy(),
        time_frame.loc[cal_time_mask].copy(),
        time_feature_columns,
        args,
    )
    cal_time_prediction = time_frame.loc[cal_time_mask, ["time_id", "market_target"]].copy()
    cal_time_prediction["market_ridge_prediction"] = market_ridge_cal
    cal_time_prediction["market_lgbm_prediction"] = market_lgbm_cal
    cal_time_prediction["market_prediction"] = 0.5 * market_ridge_cal + 0.5 * market_lgbm_cal

    relative_ridge_cal, relative_residual_cal, _, _ = fit_predict_relative(
        row_frame.loc[fit_row_mask].copy(),
        row_frame.loc[cal_row_mask].copy(),
        row_feature_columns,
        args,
    )
    relative_cal = relative_ridge_cal + relative_residual_cal
    market_cal = map_time_prediction(row_frame.loc[cal_row_mask, "time_id"], cal_time_prediction, "market_prediction")
    y_cal = row_frame.loc[cal_row_mask, "target"].to_numpy(dtype=np.float64)
    w_cal = row_frame.loc[cal_row_mask, "weight"].to_numpy(dtype=np.float64)
    asset_cal = row_frame.loc[cal_row_mask, "asset_id"].to_numpy(dtype=np.int64)
    time_cal = row_frame.loc[cal_row_mask, "time_id"].to_numpy(dtype=np.int64)

    best = search_panel_blend(y_cal, w_cal, asset_cal, time_cal, market_cal, relative_cal, args)
    cal_predictions = row_frame.loc[cal_row_mask, BASE_COLUMNS_TRAIN + ["market_target", "relative_target"]].copy()
    cal_predictions["market_prediction"] = market_cal
    cal_predictions["relative_ridge_prediction"] = relative_ridge_cal
    cal_predictions["relative_residual_prediction"] = relative_residual_cal
    cal_predictions["relative_prediction"] = relative_cal
    cal_predictions["raw_prediction"] = best["raw_prediction"]
    cal_predictions["prediction"] = best["prediction"]
    cal_predictions["error"] = cal_predictions["prediction"] - cal_predictions["target"]
    cal_predictions.to_csv(args.results_dir / "calibration_predictions.csv", index=False)
    cal_time_prediction.to_csv(args.results_dir / "market_calibration_predictions.csv", index=False)

    output_files = {
        "calibration_predictions": str(args.results_dir / "calibration_predictions.csv"),
        "market_calibration_predictions": str(args.results_dir / "market_calibration_predictions.csv"),
    }
    test_stats = None
    trained_payload = None

    if not args.skip_test_prediction:
        test_min = args.test_start_time if args.test_start_time is not None else test_min_time
        test_max = args.test_end_time if args.test_end_time is not None else test_max_time_available
        raw_test = read_partitioned_frame(
            test_paths,
            BASE_COLUMNS_TEST + raw_features,
            min_time=test_min,
            max_time=test_max,
            max_rows=args.max_test_rows,
        )
        raw_test_for_features = raw_test.copy()
        raw_test_for_features["target"] = 0.0
        raw_test_for_features["weight"] = 0.0

        final_train_rows = row_frame.loc[final_row_mask, BASE_COLUMNS_TRAIN + raw_features].copy()
        raw_combined = pd.concat(
            [final_train_rows, raw_test_for_features[BASE_COLUMNS_TRAIN + raw_features]],
            ignore_index=True,
        )
        row_combined, row_feature_columns_final, time_combined, time_feature_columns_final = build_training_frames(
            raw_combined,
            raw_features,
            cs_features,
        )
        del raw_combined, raw_test_for_features, final_train_rows
        gc.collect()

        combined_time = row_combined["time_id"].to_numpy(dtype=np.int64)
        combined_train_mask = (combined_time >= train_start_time) & (combined_time <= train_end_time)
        combined_test_mask = combined_time > train_end_time
        time_combined_values = time_combined["time_id"].to_numpy(dtype=np.int64)
        time_train_mask = (time_combined_values >= train_start_time) & (time_combined_values <= train_end_time)
        time_test_mask = time_combined_values > train_end_time

        market_ridge_test, market_lgbm_test, market_ridge_model, market_lgbm_models = fit_predict_market(
            time_combined.loc[time_train_mask].copy(),
            time_combined.loc[time_test_mask].copy(),
            time_feature_columns_final,
            args,
        )
        test_time_prediction = time_combined.loc[time_test_mask, ["time_id"]].copy()
        test_time_prediction["market_prediction"] = 0.5 * market_ridge_test + 0.5 * market_lgbm_test

        relative_ridge_test, relative_residual_test, relative_ridge_model, relative_lgbm_models = fit_predict_relative(
            row_combined.loc[combined_train_mask].copy(),
            row_combined.loc[combined_test_mask].copy(),
            row_feature_columns_final,
            args,
        )
        relative_test = relative_ridge_test + relative_residual_test
        market_test = map_time_prediction(row_combined.loc[combined_test_mask, "time_id"], test_time_prediction, "market_prediction")
        raw_prediction_test = (
            float(best["market_weight"]) * market_test + float(best["relative_weight"]) * relative_test
        )
        asset_test = row_combined.loc[combined_test_mask, "asset_id"].to_numpy(dtype=np.int64)
        prediction_test = apply_shrink(raw_prediction_test, asset_test, best["shrink_info"])

        test_predictions = row_combined.loc[combined_test_mask, BASE_COLUMNS_TEST].copy()
        test_predictions["market_prediction"] = market_test
        test_predictions["relative_ridge_prediction"] = relative_ridge_test
        test_predictions["relative_residual_prediction"] = relative_residual_test
        test_predictions["relative_prediction"] = relative_test
        test_predictions["raw_prediction"] = raw_prediction_test
        test_predictions["prediction"] = prediction_test
        test_predictions.to_csv(args.results_dir / "final_test_predictions.csv", index=False)
        submission = test_predictions[["row_id", "prediction"]].rename(columns={"prediction": "target"})
        submission = reorder_like_sample(submission, args.sample_submission)
        submission_path = args.results_dir / "submission.csv"
        zip_path = args.results_dir / "submission.zip"
        submission.to_csv(submission_path, index=False)
        save_zip(submission_path, zip_path)
        output_files.update(
            {
                "final_test_predictions": str(args.results_dir / "final_test_predictions.csv"),
                "submission": str(submission_path),
                "submission_zip": str(zip_path),
            }
        )
        test_stats = prediction_stats(prediction_test)
        trained_payload = {
            "market_ridge_model": market_ridge_model,
            "market_lgbm_models": market_lgbm_models,
            "relative_ridge_model": relative_ridge_model,
            "relative_lgbm_models": relative_lgbm_models,
            "raw_features": raw_features,
            "cs_features": cs_features,
            "time_feature_columns": time_feature_columns_final,
            "row_feature_columns": row_feature_columns_final,
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
        },
        "train_window": {
            "raw_train_min_time": int(train_min_time),
            "raw_train_max_time": int(train_max_time_available),
            "used_train_start_time": int(train_start_time),
            "used_train_end_time": int(train_end_time),
            "train_lookback_time_points": int(args.train_lookback_time_points),
        },
        "rows": {
            "fit_train": int(fit_row_mask.sum()),
            "calibration": int(cal_row_mask.sum()),
            "final_train": int(final_row_mask.sum()),
            "calibration_time_points": int(cal_time_mask.sum()),
            "test_predicted": None if args.skip_test_prediction else int(test_stats["finite_count"]),
        },
        "feature_config": {
            "raw_feature_count": int(len(raw_features)),
            "cs_feature_count": int(len(cs_features)),
            "row_feature_count": int(len(row_feature_columns)),
            "time_feature_count": int(len(time_feature_columns)),
        },
        "model": {
            "market_head": "Ridge + LightGBM 50/50",
            "relative_head": "Ridge + LightGBM residual",
            "market_ridge_alpha": float(args.market_ridge_alpha),
            "relative_ridge_alpha": float(args.relative_ridge_alpha),
            "market_lgbm_num_leaves": int(args.market_lgbm_num_leaves),
            "relative_lgbm_num_leaves": int(args.relative_lgbm_num_leaves),
            "lgbm_seeds": [int(seed) for seed in args.lgbm_seeds],
        },
        "calibration": {
            "candidate_score_mode": args.candidate_score_mode,
            "selection_score": float(best["score"]),
            "full_score": float(best["score_info"]["full_score"]),
            "first_half_score": float(best["score_info"]["first_half_score"]),
            "second_half_score": float(best["score_info"]["second_half_score"]),
            "market_weight": float(best["market_weight"]),
            "relative_weight": float(best["relative_weight"]),
            "shrink_info": best["shrink_info"],
            "shrink_summary": best["shrink_summary"],
            "market_only_score": float(weighted_zero_mean_r2(y_cal, market_cal, w_cal)),
            "relative_only_score": float(weighted_zero_mean_r2(y_cal, relative_cal, w_cal)),
            **score_time_blocks(y_cal, best["prediction"], w_cal, time_cal, 4),
            **score_time_blocks(y_cal, best["prediction"], w_cal, time_cal, 8),
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
    pd.DataFrame({"feature_name": raw_features}).to_csv(args.results_dir / "raw_features.csv", index=False)
    pd.DataFrame({"feature_name": row_feature_columns}).to_csv(args.results_dir / "row_features.csv", index=False)
    pd.DataFrame({"feature_name": time_feature_columns}).to_csv(args.results_dir / "time_features.csv", index=False)
    if trained_payload is not None and not args.no_save_models:
        save_pickle(args.model_dir / "panel_models.pkl", trained_payload)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
