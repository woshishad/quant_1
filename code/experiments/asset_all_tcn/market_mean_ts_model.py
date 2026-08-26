from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from final_train_predict import (
    BASE_COLUMNS_TRAIN,
    load_feature_ranking,
    parquet_paths,
    read_partitioned_frame,
    schema_columns,
    time_range,
)
from walk_forward_tabular import (
    apply_shrink,
    calibrate_shrink_info,
    score_candidate_on_calibration,
    summarize_shrink_info,
    weighted_zero_mean_r2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "只预测每个 time_id 的横截面加权 target 均值。"
            "这个脚本用于验证市场共同项是否能给现有模型带来增益。"
        )
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset_all_market_mean_ts_probe"))
    parser.add_argument(
        "--fixed-features-file",
        type=Path,
        default=Path("results/asset_all_stable_features_100k/selected_features_stable_top128.csv"),
    )
    parser.add_argument("--train-lookback-time-points", type=int, default=75_000)
    parser.add_argument("--cal-time-points", type=int, default=20_000)
    parser.add_argument("--max-train-time-id", type=int, default=None)

    # 特征规模控制。time-level 模型样本只有几万行，特征可以比逐行模型稍微多一点。
    parser.add_argument("--top-k-agg", type=int, default=64)
    parser.add_argument("--pivot-top-k", type=int, default=24)
    parser.add_argument("--history-top-k", type=int, default=32)
    parser.add_argument("--lag-steps", type=int, nargs="+", default=[1, 2, 5, 20])
    parser.add_argument("--rolling-windows", type=int, nargs="+", default=[20, 60, 240])

    # Ridge 负责给一个强正则的线性基准；LightGBM 负责吃非线性横截面结构。
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[10.0, 100.0, 1000.0])
    parser.add_argument("--lgbm-num-leaves-candidates", type=int, nargs="+", default=[7, 15])
    parser.add_argument("--lgbm-min-child-samples-candidates", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--lgbm-reg-lambda-candidates", type=float, nargs="+", default=[100.0, 300.0])
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.01)
    parser.add_argument("--lgbm-subsample", type=float, default=0.8)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--lgbm-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--lgbm-n-jobs", type=int, default=-1)

    # 融合搜索只在 calibration 上做；不会把官方 test 信息用于训练或调参。
    parser.add_argument("--candidate-top-n", type=int, default=8)
    parser.add_argument("--ensemble-max-size", type=int, default=5)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--shrink-mode", choices=["global", "per_asset"], default="per_asset")
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[0.8, 1.0, 1.2, 1.4])
    parser.add_argument("--candidate-score-mode", choices=["full", "mean_halves", "min_halves"], default="full")
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


def normalized_weight(weight: np.ndarray) -> np.ndarray:
    return weight / max(float(np.mean(weight)), 1e-12)


def weighted_market_target(frame: pd.DataFrame) -> pd.DataFrame:
    """把每个 time_id 内 15 个标的的 target 压成一个加权市场均值。"""
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


def add_aggregate_features(frame: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """用同一时点所有标的的均值、波动、极值和极差描述市场横截面状态。"""
    grouped = frame.groupby("time_id", sort=True)[feature_names]
    mean_frame = grouped.mean().add_suffix("_xmean")
    std_frame = grouped.std().fillna(0.0).add_suffix("_xstd")
    min_frame = grouped.min().add_suffix("_xmin")
    max_frame = grouped.max().add_suffix("_xmax")
    range_frame = (max_frame.to_numpy(dtype=np.float32) - min_frame.to_numpy(dtype=np.float32))
    range_columns = [f"{name}_xrange" for name in feature_names]
    range_frame = pd.DataFrame(range_frame, index=max_frame.index, columns=range_columns)
    out = pd.concat([mean_frame, std_frame, min_frame, max_frame, range_frame], axis=1).reset_index()
    return out, [column for column in out.columns if column != "time_id"]


def add_asset_pivot_features(frame: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """保留少量 asset-specific 当前值，避免市场模型只看到被平均后的信息。"""
    pieces = []
    feature_columns: list[str] = []
    for feature in feature_names:
        pivot = frame.pivot_table(index="time_id", columns="asset_id", values=feature, aggfunc="mean")
        pivot = pivot.sort_index(axis=1)
        pivot.columns = [f"{feature}_asset_{int(asset)}" for asset in pivot.columns]
        feature_columns.extend([str(column) for column in pivot.columns])
        pieces.append(pivot)
    if not pieces:
        return pd.DataFrame({"time_id": sorted(frame["time_id"].unique())}), []
    out = pd.concat(pieces, axis=1).reset_index()
    return out, feature_columns


def add_history_features(
    time_frame: pd.DataFrame,
    base_mean_columns: list[str],
    lag_steps: list[int],
    rolling_windows: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    """
    构造只依赖当前和过去 feature 的时序特征。
    这里不使用历史 target，因为官方 test 是批量未知 target，历史 target 会让验证不再可复现。
    """
    out = time_frame.sort_values("time_id", kind="mergesort").reset_index(drop=True).copy()
    created: dict[str, pd.Series] = {}
    for column in base_mean_columns:
        current = out[column].astype(np.float32)
        for lag in lag_steps:
            lagged = current.shift(lag)
            created[f"{column}_lag_{lag}"] = lagged
            created[f"{column}_delta_{lag}"] = current - lagged
        for window in rolling_windows:
            minimum = max(3, min(window, window // 4))
            rolling_mean = current.shift(1).rolling(window=window, min_periods=minimum).mean()
            rolling_std = current.shift(1).rolling(window=window, min_periods=minimum).std()
            created[f"{column}_roll_mean_{window}"] = rolling_mean
            created[f"{column}_roll_std_{window}"] = rolling_std
            created[f"{column}_roll_dev_{window}"] = current - rolling_mean
    if created:
        out = pd.concat([out, pd.DataFrame(created, index=out.index)], axis=1)
    return out, list(created.keys())


def build_time_feature_frame(
    frame: pd.DataFrame,
    agg_features: list[str],
    pivot_features: list[str],
    history_features: list[str],
    lag_steps: list[int],
    rolling_windows: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    """把逐行面板数据转换成每个 time_id 一行的市场状态特征。"""
    aggregate_frame, aggregate_columns = add_aggregate_features(frame, agg_features)
    pivot_frame, pivot_columns = add_asset_pivot_features(frame, pivot_features)
    time_frame = aggregate_frame.merge(pivot_frame, on="time_id", how="left")
    history_base_columns = [f"{feature}_xmean" for feature in history_features if f"{feature}_xmean" in time_frame]
    time_frame, history_columns = add_history_features(time_frame, history_base_columns, lag_steps, rolling_windows)
    feature_columns = aggregate_columns + pivot_columns + history_columns
    return time_frame, feature_columns


def prepare_matrix(
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用 fit 段统计量填补缺失并标准化；避免 calibration 信息流入训练预处理。"""
    fit_x = fit_frame[feature_columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    predict_x = predict_frame[feature_columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    medians = np.nanmedian(fit_x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    fit_nan = ~np.isfinite(fit_x)
    predict_nan = ~np.isfinite(predict_x)
    if fit_nan.any():
        fit_x[fit_nan] = np.take(medians, np.where(fit_nan)[1])
    if predict_nan.any():
        predict_x[predict_nan] = np.take(medians, np.where(predict_nan)[1])

    means = fit_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = fit_x.std(axis=0, dtype=np.float64).astype(np.float32)
    stds = np.where(stds > 1e-8, stds, 1.0).astype(np.float32)
    fit_z = ((fit_x - means) / stds).astype(np.float32)
    predict_z = ((predict_x - means) / stds).astype(np.float32)
    return fit_x, predict_x, fit_z, predict_z


def fit_ridge_candidates(
    fit_z: np.ndarray,
    predict_z: np.ndarray,
    y_fit: np.ndarray,
    w_fit: np.ndarray,
    alphas: list[float],
) -> dict[str, np.ndarray]:
    predictions = {}
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), solver="lsqr", max_iter=800)
        model.fit(fit_z, y_fit, sample_weight=normalized_weight(w_fit))
        predictions[f"ridge_alpha_{alpha:g}"] = model.predict(predict_z).astype(np.float64)
    return predictions


def fit_lgbm_candidates(
    fit_x: np.ndarray,
    predict_x: np.ndarray,
    y_fit: np.ndarray,
    w_fit: np.ndarray,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    predictions = {}
    fit_frame = pd.DataFrame(fit_x, columns=feature_columns)
    predict_frame = pd.DataFrame(predict_x, columns=feature_columns)
    for leaves in args.lgbm_num_leaves_candidates:
        for child in args.lgbm_min_child_samples_candidates:
            for reg_lambda in args.lgbm_reg_lambda_candidates:
                seed_predictions = []
                for seed in args.lgbm_seeds:
                    model = LGBMRegressor(
                        objective="regression",
                        n_estimators=int(args.lgbm_estimators),
                        learning_rate=float(args.lgbm_learning_rate),
                        num_leaves=int(leaves),
                        min_child_samples=int(child),
                        subsample=float(args.lgbm_subsample),
                        subsample_freq=1,
                        colsample_bytree=float(args.lgbm_colsample_bytree),
                        reg_lambda=float(reg_lambda),
                        random_state=int(seed),
                        n_jobs=int(args.lgbm_n_jobs),
                        verbose=-1,
                    )
                    model.fit(fit_frame, y_fit, sample_weight=normalized_weight(w_fit))
                    seed_predictions.append(model.predict(predict_frame))
                name = f"lgbm_l{leaves}_child{child}_lambda{reg_lambda:g}"
                predictions[name] = np.mean(np.vstack(seed_predictions), axis=0).astype(np.float64)
    return predictions


def map_time_predictions_to_rows(
    row_time_id: np.ndarray,
    time_id: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    mapping = pd.Series(prediction, index=time_id)
    return pd.Series(row_time_id).map(mapping).to_numpy(dtype=np.float64)


def score_raw_prediction(
    y_true: np.ndarray,
    raw_prediction: np.ndarray,
    weight: np.ndarray,
    asset_id: np.ndarray,
    time_id: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    best = {"selection_score": -np.inf}
    for cap in args.shrink_cap_candidates:
        shrink_info = calibrate_shrink_info(
            y_true,
            raw_prediction,
            weight,
            asset_id,
            args.shrink_mode,
            float(cap),
        )
        prediction = apply_shrink(raw_prediction, asset_id, shrink_info)
        score_info = score_candidate_on_calibration(
            y_true,
            prediction,
            weight,
            time_id,
            args.candidate_score_mode,
        )
        if score_info["selection_score"] > best["selection_score"]:
            best = {
                **score_info,
                "shrink_info": shrink_info,
                "shrink_summary": summarize_shrink_info(shrink_info),
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
    unique_times = np.unique(time_id)
    for chunk in np.array_split(unique_times, block_count):
        if len(chunk) == 0:
            continue
        mask = (time_id >= int(chunk[0])) & (time_id <= int(chunk[-1]))
        values.append(float(weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])))
    scores = np.asarray(values, dtype=np.float64)
    return {
        f"block{block_count}_mean_score": float(np.mean(scores)),
        f"block{block_count}_min_score": float(np.min(scores)),
        f"block{block_count}_last_score": float(scores[-1]),
        f"block{block_count}_negative_count": int(np.sum(scores < 0.0)),
    }


def greedy_ensemble_search(
    candidate_row_predictions: dict[str, np.ndarray],
    candidate_scores: pd.DataFrame,
    y_true: np.ndarray,
    weight: np.ndarray,
    asset_id: np.ndarray,
    time_id: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """在少量强候选上做贪心线性融合，避免高维网格搜索爆炸。"""
    top_names = candidate_scores.sort_values("selection_score", ascending=False)["name"].head(args.candidate_top_n).tolist()
    current_name = top_names[0]
    current_raw = candidate_row_predictions[current_name].copy()
    current_weights = {current_name: 1.0}
    best_score = score_raw_prediction(y_true, current_raw, weight, asset_id, time_id, args)

    history = [
        {
            "step": 1,
            "added": current_name,
            "mix_current": 1.0,
            "selection_score": best_score["selection_score"],
            "full_score": best_score["full_score"],
            "first_half_score": best_score["first_half_score"],
            "second_half_score": best_score["second_half_score"],
            "weights": json.dumps(current_weights, ensure_ascii=False, sort_keys=True),
        }
    ]

    for step in range(2, int(args.ensemble_max_size) + 1):
        step_best = None
        for name in top_names:
            if name in current_weights:
                continue
            candidate_raw = candidate_row_predictions[name]
            for mix_current in np.arange(0.0, 1.0 + 1e-12, float(args.blend_step)):
                raw = float(mix_current) * current_raw + (1.0 - float(mix_current)) * candidate_raw
                score_info = score_raw_prediction(y_true, raw, weight, asset_id, time_id, args)
                if step_best is None or score_info["selection_score"] > step_best["score_info"]["selection_score"]:
                    step_best = {
                        "name": name,
                        "mix_current": float(mix_current),
                        "raw": raw,
                        "score_info": score_info,
                    }
        if step_best is None:
            break
        if step_best["score_info"]["selection_score"] <= best_score["selection_score"] + 1e-12:
            break

        # 更新融合权重：旧 ensemble 乘 mix，新候选乘 1 - mix。
        mix = float(step_best["mix_current"])
        current_weights = {name: weight_value * mix for name, weight_value in current_weights.items()}
        current_weights[step_best["name"]] = 1.0 - mix
        current_weights = {name: value for name, value in current_weights.items() if abs(value) > 1e-12}
        current_raw = step_best["raw"]
        best_score = step_best["score_info"]
        history.append(
            {
                "step": step,
                "added": step_best["name"],
                "mix_current": mix,
                "selection_score": best_score["selection_score"],
                "full_score": best_score["full_score"],
                "first_half_score": best_score["first_half_score"],
                "second_half_score": best_score["second_half_score"],
                "weights": json.dumps(current_weights, ensure_ascii=False, sort_keys=True),
            }
        )

    return {
        "raw_prediction": current_raw,
        "prediction": best_score["prediction"],
        "weights": current_weights,
        "score_info": best_score,
        "history": pd.DataFrame(history),
    }


def plot_market_prediction(time_frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(time_frame["time_id"], time_frame["market_target"], label="market target", linewidth=1.0, alpha=0.75)
    plt.plot(time_frame["time_id"], time_frame["market_prediction"], label="prediction", linewidth=1.0, alpha=0.75)
    plt.legend()
    plt.xlabel("time_id")
    plt.ylabel("weighted target mean")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_candidate_scores(candidate_scores: pd.DataFrame, output_path: Path) -> None:
    top = candidate_scores.sort_values("selection_score", ascending=False).head(20).iloc[::-1]
    plt.figure(figsize=(10, max(4, 0.32 * len(top))))
    plt.barh(top["name"], top["selection_score"], color="#2563eb")
    plt.xlabel("selection score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train_paths = parquet_paths(args.raw_data_dir, "train")
    train_min_time, train_max_time_available = time_range(train_paths)
    train_end_time = (
        min(train_max_time_available, int(args.max_train_time_id))
        if args.max_train_time_id is not None
        else train_max_time_available
    )
    train_start_time = max(train_min_time, train_end_time - int(args.train_lookback_time_points) + 1)
    fit_train_end_time = train_end_time - int(args.cal_time_points)
    cal_start_time = fit_train_end_time + 1
    if fit_train_end_time < train_start_time:
        raise ValueError("--cal-time-points 对当前 lookback 来说过大")

    available_columns = schema_columns(train_paths)
    ranking = load_feature_ranking(args.fixed_features_file, available_columns)
    agg_features = ranking.head(int(args.top_k_agg))["feature_name"].astype(str).tolist()
    pivot_features = ranking.head(int(args.pivot_top_k))["feature_name"].astype(str).tolist()
    history_features = ranking.head(int(args.history_top_k))["feature_name"].astype(str).tolist()
    read_features = sorted(set(agg_features + pivot_features + history_features))

    print(
        f"Market mean split: fit={train_start_time}..{fit_train_end_time}, "
        f"cal={cal_start_time}..{train_end_time}; features={len(read_features)}"
    )
    raw_train = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + read_features,
        min_time=train_start_time,
        max_time=train_end_time,
    )

    market_targets = weighted_market_target(raw_train)
    time_features, feature_columns = build_time_feature_frame(
        raw_train,
        agg_features,
        pivot_features,
        history_features,
        args.lag_steps,
        args.rolling_windows,
    )
    del raw_train
    gc.collect()

    time_frame = time_features.merge(market_targets, on="time_id", how="left")
    time_values = time_frame["time_id"].to_numpy(dtype=np.int64)
    fit_mask = (time_values >= train_start_time) & (time_values <= fit_train_end_time)
    cal_mask = (time_values >= cal_start_time) & (time_values <= train_end_time)

    fit_time = time_frame.loc[fit_mask].copy()
    cal_time = time_frame.loc[cal_mask].copy()
    fit_x, cal_x, fit_z, cal_z = prepare_matrix(fit_time, cal_time, feature_columns)
    y_fit = fit_time["market_target"].to_numpy(dtype=np.float32)
    w_fit = fit_time["weight_sum"].to_numpy(dtype=np.float32)

    candidate_time_predictions: dict[str, np.ndarray] = {}
    candidate_time_predictions.update(
        fit_ridge_candidates(fit_z, cal_z, y_fit, w_fit, [float(alpha) for alpha in args.ridge_alphas])
    )
    candidate_time_predictions.update(
        fit_lgbm_candidates(fit_x, cal_x, y_fit, w_fit, feature_columns, args)
    )

    row_frame = market_targets.merge(
        pd.DataFrame({"time_id": cal_time["time_id"].to_numpy(dtype=np.int64)}),
        on="time_id",
        how="inner",
    )
    del row_frame

    # 行级评分仍然使用原始 15 标的 target/weight，因为比赛指标是逐行 R2。
    raw_cal = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN,
        min_time=cal_start_time,
        max_time=train_end_time,
    )
    y_cal = raw_cal["target"].to_numpy(dtype=np.float64)
    w_cal = raw_cal["weight"].to_numpy(dtype=np.float64)
    asset_cal = raw_cal["asset_id"].to_numpy(dtype=np.int64)
    row_time_cal = raw_cal["time_id"].to_numpy(dtype=np.int64)
    cal_time_id = cal_time["time_id"].to_numpy(dtype=np.int64)

    candidate_row_predictions = {
        name: map_time_predictions_to_rows(row_time_cal, cal_time_id, prediction)
        for name, prediction in candidate_time_predictions.items()
    }

    rows = []
    for name, prediction in candidate_row_predictions.items():
        score_info = score_raw_prediction(y_cal, prediction, w_cal, asset_cal, row_time_cal, args)
        rows.append(
            {
                "name": name,
                "selection_score": score_info["selection_score"],
                "full_score": score_info["full_score"],
                "first_half_score": score_info["first_half_score"],
                "second_half_score": score_info["second_half_score"],
                **score_info["shrink_summary"],
            }
        )
    candidate_scores = pd.DataFrame(rows).sort_values("selection_score", ascending=False)

    ensemble = greedy_ensemble_search(
        candidate_row_predictions,
        candidate_scores,
        y_cal,
        w_cal,
        asset_cal,
        row_time_cal,
        args,
    )

    cal_predictions = raw_cal.copy()
    cal_predictions["market_target"] = map_time_predictions_to_rows(
        row_time_cal,
        cal_time_id,
        cal_time["market_target"].to_numpy(dtype=np.float64),
    )
    cal_predictions["raw_prediction"] = ensemble["raw_prediction"]
    cal_predictions["prediction"] = ensemble["prediction"]
    cal_predictions["error"] = cal_predictions["prediction"] - cal_predictions["target"]
    cal_predictions.to_csv(args.results_dir / "calibration_predictions.csv", index=False)

    cal_time_output = cal_time[["time_id", "market_target", "weight_sum", "row_count"]].copy()
    for name, prediction in candidate_time_predictions.items():
        cal_time_output[f"prediction_{name}"] = prediction
    raw_market_by_time = pd.Series(ensemble["raw_prediction"], index=row_time_cal).groupby(level=0).mean()
    pred_market_by_time = pd.Series(ensemble["prediction"], index=row_time_cal).groupby(level=0).mean()
    cal_time_output["raw_market_prediction"] = cal_time_output["time_id"].map(raw_market_by_time).to_numpy(dtype=np.float64)
    cal_time_output["market_prediction"] = cal_time_output["time_id"].map(pred_market_by_time).to_numpy(dtype=np.float64)
    cal_time_output.to_csv(args.results_dir / "market_calibration_predictions.csv", index=False)

    candidate_scores.to_csv(args.results_dir / "candidate_scores.csv", index=False)
    ensemble["history"].to_csv(args.results_dir / "ensemble_history.csv", index=False)
    plot_market_prediction(cal_time_output, args.results_dir / "market_target_vs_prediction.png")
    plot_candidate_scores(candidate_scores, args.results_dir / "candidate_scores_top20.png")

    score_info = ensemble["score_info"]
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_safe": True,
        "official_test_used": False,
        "raw_data_dir": str(args.raw_data_dir),
        "split": {
            "fit_train": f"{train_start_time} <= time_id <= {fit_train_end_time}",
            "calibration": f"{cal_start_time} <= time_id <= {train_end_time}",
            "train_lookback_time_points": int(args.train_lookback_time_points),
            "cal_time_points": int(args.cal_time_points),
        },
        "rows": {
            "fit_time_points": int(fit_mask.sum()),
            "cal_time_points": int(cal_mask.sum()),
            "cal_rows": int(len(cal_predictions)),
        },
        "feature_config": {
            "read_feature_count": int(len(read_features)),
            "time_feature_count": int(len(feature_columns)),
            "top_k_agg": int(args.top_k_agg),
            "pivot_top_k": int(args.pivot_top_k),
            "history_top_k": int(args.history_top_k),
            "lag_steps": [int(value) for value in args.lag_steps],
            "rolling_windows": [int(value) for value in args.rolling_windows],
        },
        "model": {
            "base_candidates": int(len(candidate_time_predictions)),
            "ensemble_weights": ensemble["weights"],
            "candidate_score_mode": args.candidate_score_mode,
            "shrink_mode": args.shrink_mode,
            "shrink_info": score_info["shrink_info"],
        },
        "calibration": {
            "selection_score": float(score_info["selection_score"]),
            "full_score": float(score_info["full_score"]),
            "first_half_score": float(score_info["first_half_score"]),
            "second_half_score": float(score_info["second_half_score"]),
            "market_target_time_r2_raw": float(
                weighted_zero_mean_r2(
                    cal_time_output["market_target"].to_numpy(dtype=np.float64),
                    cal_time_output["raw_market_prediction"].to_numpy(dtype=np.float64),
                    cal_time_output["weight_sum"].to_numpy(dtype=np.float64),
                )
            ),
            "prediction_std": float(np.std(cal_predictions["prediction"].to_numpy(dtype=np.float64))),
            "raw_prediction_std": float(np.std(cal_predictions["raw_prediction"].to_numpy(dtype=np.float64))),
            **score_info["shrink_summary"],
            **score_time_blocks(y_cal, ensemble["prediction"], w_cal, row_time_cal, 4),
            **score_time_blocks(y_cal, ensemble["prediction"], w_cal, row_time_cal, 8),
        },
        "output_files": {
            "calibration_predictions": str(args.results_dir / "calibration_predictions.csv"),
            "market_calibration_predictions": str(args.results_dir / "market_calibration_predictions.csv"),
            "candidate_scores": str(args.results_dir / "candidate_scores.csv"),
            "ensemble_history": str(args.results_dir / "ensemble_history.csv"),
            "market_target_vs_prediction": str(args.results_dir / "market_target_vs_prediction.png"),
            "candidate_scores_top20": str(args.results_dir / "candidate_scores_top20.png"),
        },
    }
    with (args.results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, default=json_default)

    print(json.dumps(metrics["calibration"], ensure_ascii=False, indent=2, default=json_default))
    print(f"Saved outputs to {args.results_dir}")


if __name__ == "__main__":
    main()
