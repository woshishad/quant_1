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
    parser = argparse.ArgumentParser(description="Leakage-safe walk-forward all-asset tabular experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/asset_all_time100000"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset_all_walk_forward_tabular_100k"))
    parser.add_argument("--fit-train-end-time", type=int, default=59_999)
    parser.add_argument("--cal-start-time", type=int, default=60_000)
    parser.add_argument("--cal-end-time", type=int, default=79_999)
    parser.add_argument("--test-start-time", type=int, default=80_000)
    parser.add_argument("--test-end-time", type=int, default=99_999)
    parser.add_argument("--top-k-candidates", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--lgbm-num-leaves-candidates", type=int, nargs="+", default=[15, 31])
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.005)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=8000)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=500.0)
    parser.add_argument("--lgbm-subsample", type=float, default=0.7)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.7)
    parser.add_argument(
        "--lgbm-seeds",
        type=int,
        nargs="+",
        default=[42],
        help="LightGBM 随机种子列表；多个 seed 会分别训练后平均预测，用来降低低信噪比任务里的随机方差。",
    )
    parser.add_argument("--blend-step", type=float, default=0.01)
    parser.add_argument("--max-cal-shrink", type=float, default=None)
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[1.2])
    parser.add_argument(
        "--candidate-score-mode",
        choices=["full", "mean_halves", "min_halves"],
        default="full",
        help="候选参数在 calibration 段上的打分方式；min_halves 会偏向前后半段都稳定的参数。",
    )
    parser.add_argument(
        "--shrink-mode",
        choices=["global", "per_asset"],
        default="global",
        help="global 表示所有标的共用一个 shrink；per_asset 表示每个标的在 calibration 段单独校准 shrink。",
    )
    parser.add_argument("--fixed-features-file", type=Path, default=None)
    return parser.parse_args()


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    denominator = float(np.sum(weight * y_true * y_true))
    if denominator <= 0:
        return 0.0
    return 1.0 - float(np.sum(weight * (y_true - y_pred) ** 2)) / denominator


def optimal_shrink(y_true: np.ndarray, prediction: np.ndarray, weight: np.ndarray, cap: float = 1.2) -> float:
    # 在固定预测方向的前提下，闭式求解让加权 MSE 最小的缩放系数。
    # 金融 target 信噪比低，直接输出常常幅度过大；shrink 可以把预测幅度压回更稳的区间。
    denominator = float(np.sum(weight * prediction * prediction))
    if denominator <= 1e-18:
        return 0.0
    shrink = float(np.sum(weight * y_true * prediction) / denominator)
    return min(float(cap), max(0.0, shrink))


def score_by_asset(y_true: np.ndarray, prediction: np.ndarray, weight: np.ndarray, asset_id: np.ndarray) -> dict[str, float]:
    scores = {}
    for asset in sorted(np.unique(asset_id)):
        mask = asset_id == asset
        scores[str(int(asset))] = weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])
    return scores


def calibrate_shrink_info(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    asset_id: np.ndarray,
    shrink_mode: str,
    shrink_cap: float,
) -> dict:
    # shrink 只能用 calibration 段 target 校准，不能用未来 holdout/test 段。
    # per_asset 模式对每个标的分别压缩预测幅度，用来减少弱标的被统一放大后拖累总分的风险。
    global_shrink = optimal_shrink(y_true, prediction, weight, shrink_cap)
    shrink_info = {
        "mode": shrink_mode,
        "cap": float(shrink_cap),
        "global": float(global_shrink),
        "by_asset": {},
    }
    if shrink_mode == "global":
        return shrink_info

    for asset in sorted(np.unique(asset_id)):
        mask = asset_id == asset
        asset_shrink = optimal_shrink(y_true[mask], prediction[mask], weight[mask], shrink_cap)
        shrink_info["by_asset"][str(int(asset))] = float(asset_shrink)
    return shrink_info


def shrink_values_for_assets(asset_id: np.ndarray, shrink_info: dict) -> np.ndarray:
    # 生成每一行样本对应的 shrink 值；测试段如果出现 calibration 未见过的标的，就回退到 global shrink。
    shrink_values = np.full(len(asset_id), float(shrink_info["global"]), dtype=np.float64)
    if shrink_info.get("mode") != "per_asset":
        return shrink_values
    for asset_name, shrink in shrink_info.get("by_asset", {}).items():
        shrink_values[asset_id == int(asset_name)] = float(shrink)
    return shrink_values


def apply_shrink(prediction: np.ndarray, asset_id: np.ndarray, shrink_info: dict) -> np.ndarray:
    return shrink_values_for_assets(asset_id, shrink_info) * prediction


def summarize_shrink_info(shrink_info: dict) -> dict[str, float]:
    # 这些摘要会写进 candidate_metrics.csv，方便横向看每组参数是否依赖过强放大。
    if shrink_info.get("mode") == "per_asset" and shrink_info.get("by_asset"):
        values = np.asarray(list(shrink_info["by_asset"].values()), dtype=np.float64)
    else:
        values = np.asarray([float(shrink_info["global"])], dtype=np.float64)
    return {
        "cal_shrink": float(shrink_info["global"]),
        "cal_shrink_min": float(np.min(values)),
        "cal_shrink_mean": float(np.mean(values)),
        "cal_shrink_max": float(np.max(values)),
    }


def score_candidate_on_calibration(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    score_mode: str,
) -> dict:
    # full 是老逻辑；mean/min halves 用 calibration 的前后半段做稳定性约束，减少某一小段行情过拟合。
    full_score = weighted_zero_mean_r2(y_true, prediction, weight)
    unique_times = np.unique(time_id)
    if len(unique_times) < 2:
        return {
            "selection_score": float(full_score),
            "full_score": float(full_score),
            "first_half_score": float(full_score),
            "second_half_score": float(full_score),
        }

    split_time = unique_times[len(unique_times) // 2]
    first_mask = time_id < split_time
    second_mask = ~first_mask
    first_score = weighted_zero_mean_r2(y_true[first_mask], prediction[first_mask], weight[first_mask])
    second_score = weighted_zero_mean_r2(y_true[second_mask], prediction[second_mask], weight[second_mask])
    if score_mode == "mean_halves":
        selection_score = 0.5 * (first_score + second_score)
    elif score_mode == "min_halves":
        selection_score = min(first_score, second_score)
    else:
        selection_score = full_score
    return {
        "selection_score": float(selection_score),
        "full_score": float(full_score),
        "first_half_score": float(first_score),
        "second_half_score": float(second_score),
    }


def load_fixed_feature_ranking(path: Path, feature_columns: list[str]) -> pd.DataFrame:
    selected = pd.read_csv(path)
    if "feature_name" not in selected.columns:
        raise ValueError(f"{path} must contain feature_name column")
    names = selected["feature_name"].astype(str).tolist()
    missing = sorted(set(names) - set(feature_columns))
    if missing:
        raise ValueError(f"{path} contains unknown features: {missing[:10]}")
    ranking = pd.DataFrame({"feature_name": names})
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking["cal_score"] = np.nan
    ranking["cal_raw_score"] = np.nan
    ranking["cal_shrink"] = np.nan
    return ranking


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 所有标准化统计量只在当前训练段拟合，不看校准/测试段。
    mean = np.nanmean(train_x, axis=0).astype(np.float32)
    scale = np.nanstd(train_x, axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    train_x = np.nan_to_num((train_x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    other_x = np.nan_to_num((other_x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return train_x, other_x, mean, scale


def screen_features(
    data_path: Path,
    feature_columns: list[str],
    base: pd.DataFrame,
    fit_train_mask: np.ndarray,
    cal_mask: np.ndarray,
) -> pd.DataFrame:
    # 因子筛选只用 fit_train 段拟合单因子方向，用 calibration 段打分。
    # calibration 段发生在最终 test 段之前，因此它是“过去信息”，不是未来函数。
    y_train = base.loc[fit_train_mask, "target"].to_numpy(dtype=np.float64)
    y_cal = base.loc[cal_mask, "target"].to_numpy(dtype=np.float64)
    w_train = base.loc[fit_train_mask, "weight"].to_numpy(dtype=np.float64)
    w_cal = base.loc[cal_mask, "weight"].to_numpy(dtype=np.float64)
    train_weight = w_train / max(float(np.mean(w_train)), 1e-12)
    rows = []

    for index, feature_name in enumerate(feature_columns):
        values = pd.read_parquet(data_path, columns=[feature_name])[feature_name]
        train_values = values.loc[fit_train_mask].to_numpy(dtype=np.float64)
        cal_values = values.loc[cal_mask].to_numpy(dtype=np.float64)
        mean = float(np.nanmean(train_values))
        scale = float(np.nanstd(train_values))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        train_x = np.nan_to_num((train_values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
        cal_x = np.nan_to_num((cal_values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
        denominator = float(np.sum(train_weight * train_x * train_x))
        coef = 0.0 if denominator <= 1e-18 else float(np.sum(train_weight * train_x * y_train) / denominator)
        raw_prediction = coef * cal_x
        shrink = optimal_shrink(y_cal, raw_prediction, w_cal)
        prediction = shrink * raw_prediction
        rows.append(
            {
                "feature_index": index,
                "feature_name": feature_name,
                "cal_score": weighted_zero_mean_r2(y_cal, prediction, w_cal),
                "cal_raw_score": weighted_zero_mean_r2(y_cal, raw_prediction, w_cal),
                "cal_shrink": shrink,
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(feature_columns):
            print(f"screened {index + 1}/{len(feature_columns)} features")

    ranking = pd.DataFrame(rows).sort_values("cal_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def fit_predict_pair(
    train_x: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    predict_x: np.ndarray,
    feature_names: list[str],
    ridge_alpha: float,
    lgbm_num_leaves: int,
    lgbm_estimators: int,
    lgbm_learning_rate: float,
    lgbm_min_child_samples: int,
    lgbm_reg_lambda: float,
    lgbm_seeds: list[int] | None = None,
    lgbm_subsample: float = 0.7,
    lgbm_colsample_bytree: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, Ridge, list[LGBMRegressor]]:
    sample_weight = w_train / max(float(np.mean(w_train)), 1e-12)
    ridge = Ridge(alpha=ridge_alpha, solver="lsqr", max_iter=500)
    ridge.fit(train_x, y_train, sample_weight=sample_weight)
    ridge_pred = ridge.predict(predict_x)

    # 金融数据噪声很高，单个 LightGBM seed 可能会吃到偶然切分/采样带来的方差。
    # 多 seed 平均不改变训练窗口，只是在同一历史数据上做 bagging，通常比单纯加深树更稳。
    if lgbm_seeds is None:
        lgbm_seeds = [42]
    train_frame = pd.DataFrame(train_x, columns=feature_names)
    predict_frame = pd.DataFrame(predict_x, columns=feature_names)
    lgbm_models = []
    lgbm_predictions = []
    for seed in lgbm_seeds:
        lgbm = LGBMRegressor(
            objective="regression",
            n_estimators=lgbm_estimators,
            learning_rate=lgbm_learning_rate,
            num_leaves=lgbm_num_leaves,
            min_child_samples=lgbm_min_child_samples,
            subsample=lgbm_subsample,
            subsample_freq=1,
            colsample_bytree=lgbm_colsample_bytree,
            reg_lambda=lgbm_reg_lambda,
            random_state=int(seed),
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(train_frame, y_train, sample_weight=sample_weight)
        lgbm_models.append(lgbm)
        lgbm_predictions.append(lgbm.predict(predict_frame))
    lgbm_pred = np.mean(np.vstack(lgbm_predictions), axis=0)
    return ridge_pred, lgbm_pred, ridge, lgbm_models


def find_best_blend(
    y_true: np.ndarray,
    weight: np.ndarray,
    asset_id: np.ndarray,
    time_id: np.ndarray,
    ridge_pred: np.ndarray,
    lgbm_pred: np.ndarray,
    step: float,
    shrink_mode: str,
    shrink_cap_candidates: list[float],
    score_mode: str,
) -> dict:
    # 融合权重和 shrink 只在 calibration 段拟合，后续原样应用到未来 test 段。
    best = {"score": -np.inf}
    for ridge_weight in np.arange(0.0, 1.0 + 1e-12, step):
        lgbm_weight = 1.0 - ridge_weight
        base = ridge_weight * ridge_pred + lgbm_weight * lgbm_pred
        for shrink_cap in shrink_cap_candidates:
            shrink_info = calibrate_shrink_info(y_true, base, weight, asset_id, shrink_mode, shrink_cap)
            prediction = apply_shrink(base, asset_id, shrink_info)
            score_info = score_candidate_on_calibration(y_true, prediction, weight, time_id, score_mode)
            score = score_info["selection_score"]
            if score > best["score"]:
                shrink_summary = summarize_shrink_info(shrink_info)
                best = {
                    "score": float(score),
                    "score_info": score_info,
                    "weights": {
                        "ridge": float(ridge_weight),
                        "lgbm": float(lgbm_weight),
                        "shrink": float(shrink_info["global"]),
                        "shrink_mode": shrink_mode,
                        "shrink_cap": float(shrink_cap),
                    },
                    "shrink_info": shrink_info,
                    "shrink_summary": shrink_summary,
                }
    return best


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_dir / "train.parquet"
    schema_columns = pq.ParquetFile(data_path).schema_arrow.names
    feature_columns = [column for column in schema_columns if column.startswith("feature_")]
    base_columns = ["row_id", "time_id", "asset_id", "weight", "target"]
    base = pd.read_parquet(data_path, columns=base_columns).sort_values(["time_id", "asset_id"], kind="mergesort")
    time_values = base["time_id"].to_numpy(dtype=np.int64)
    fit_train_mask = time_values <= args.fit_train_end_time
    cal_mask = (time_values >= args.cal_start_time) & (time_values <= args.cal_end_time)
    final_train_mask = time_values <= args.cal_end_time
    test_mask = (time_values >= args.test_start_time) & (time_values <= args.test_end_time)

    if args.fixed_features_file is None:
        ranking = screen_features(data_path, feature_columns, base, fit_train_mask, cal_mask)
        feature_source = "calibration_screen"
    else:
        ranking = load_fixed_feature_ranking(args.fixed_features_file, feature_columns)
        feature_source = "fixed_features_file"
        print(f"Using fixed feature ranking from {args.fixed_features_file}")
    ranking.to_csv(args.results_dir / "feature_ranking_calibration.csv", index=False)

    max_top_k = max(args.top_k_candidates)
    feature_names = ranking.head(max_top_k)["feature_name"].astype(str).tolist()
    frame = pd.read_parquet(data_path, columns=base_columns + feature_names).sort_values(
        ["time_id", "asset_id"], kind="mergesort"
    )

    y_fit = frame.loc[fit_train_mask, "target"].to_numpy(dtype=np.float32)
    w_fit = frame.loc[fit_train_mask, "weight"].to_numpy(dtype=np.float32)
    y_cal = frame.loc[cal_mask, "target"].to_numpy(dtype=np.float32)
    w_cal = frame.loc[cal_mask, "weight"].to_numpy(dtype=np.float32)
    asset_cal = frame.loc[cal_mask, "asset_id"].to_numpy(dtype=np.int64)
    time_cal = frame.loc[cal_mask, "time_id"].to_numpy(dtype=np.int64)

    candidate_rows = []
    all_best_candidate = None
    eligible_best_candidate = None
    for top_k in args.top_k_candidates:
        selected_features = ranking.head(top_k)["feature_name"].astype(str).tolist()
        fit_x_raw = frame.loc[fit_train_mask, selected_features].to_numpy(dtype=np.float32)
        cal_x_raw = frame.loc[cal_mask, selected_features].to_numpy(dtype=np.float32)
        fit_x, cal_x, _, _ = standardize(fit_x_raw, cal_x_raw)
        for leaves in args.lgbm_num_leaves_candidates:
            ridge_pred, lgbm_pred, _, _ = fit_predict_pair(
                fit_x,
                y_fit,
                w_fit,
                cal_x,
                selected_features,
                args.ridge_alpha,
                leaves,
                args.lgbm_estimators,
                args.lgbm_learning_rate,
                args.lgbm_min_child_samples,
                args.lgbm_reg_lambda,
                args.lgbm_seeds,
                args.lgbm_subsample,
                args.lgbm_colsample_bytree,
            )
            blend = find_best_blend(
                y_cal,
                w_cal,
                asset_cal,
                time_cal,
                ridge_pred,
                lgbm_pred,
                args.blend_step,
                args.shrink_mode,
                args.shrink_cap_candidates,
                args.candidate_score_mode,
            )
            shrink_summary = blend["shrink_summary"]
            score_info = blend["score_info"]
            row = {
                "top_k": int(top_k),
                "lgbm_num_leaves": int(leaves),
                "cal_score": float(blend["score"]),
                "cal_score_mode": args.candidate_score_mode,
                "cal_full_score": float(score_info["full_score"]),
                "cal_first_half_score": float(score_info["first_half_score"]),
                "cal_second_half_score": float(score_info["second_half_score"]),
                "cal_ridge_weight": float(blend["weights"]["ridge"]),
                "cal_lgbm_weight": float(blend["weights"]["lgbm"]),
                "cal_shrink_mode": args.shrink_mode,
                "cal_shrink_cap": float(blend["weights"]["shrink_cap"]),
                "cal_shrink": float(shrink_summary["cal_shrink"]),
                "cal_shrink_min": float(shrink_summary["cal_shrink_min"]),
                "cal_shrink_mean": float(shrink_summary["cal_shrink_mean"]),
                "cal_shrink_max": float(shrink_summary["cal_shrink_max"]),
                "cal_shrink_info": blend["shrink_info"],
            }
            candidate_rows.append(row)
            print(json.dumps(row))
            if all_best_candidate is None or row["cal_score"] > all_best_candidate["cal_score"]:
                all_best_candidate = row
            if args.max_cal_shrink is None or row["cal_shrink_max"] <= args.max_cal_shrink:
                if eligible_best_candidate is None or row["cal_score"] > eligible_best_candidate["cal_score"]:
                    eligible_best_candidate = row

    if all_best_candidate is None:
        raise ValueError("no candidate was evaluated")
    best_candidate = eligible_best_candidate if eligible_best_candidate is not None else all_best_candidate
    candidate_frame = pd.DataFrame(candidate_rows).sort_values("cal_score", ascending=False).reset_index(drop=True)
    candidate_frame.to_csv(args.results_dir / "candidate_metrics.csv", index=False)

    # 用校准段选出的 top_k/树叶数/融合参数，重新训练到 calibration 结束，然后只在未来 test 段评估一次。
    selected_features = ranking.head(int(best_candidate["top_k"]))["feature_name"].astype(str).tolist()
    weights = {
        "ridge": float(best_candidate["cal_ridge_weight"]),
        "lgbm": float(best_candidate["cal_lgbm_weight"]),
        "shrink": float(best_candidate["cal_shrink"]),
        "shrink_mode": str(best_candidate["cal_shrink_mode"]),
        "shrink_cap": float(best_candidate["cal_shrink_cap"]) if "cal_shrink_cap" in best_candidate else None,
    }
    shrink_info = best_candidate["cal_shrink_info"]

    # 保存 calibration 段预测，供后续做二层融合时使用；融合权重只能在 calibration 上学习。
    cal_fit_x_raw = frame.loc[fit_train_mask, selected_features].to_numpy(dtype=np.float32)
    cal_eval_x_raw = frame.loc[cal_mask, selected_features].to_numpy(dtype=np.float32)
    cal_fit_x, cal_eval_x, _, _ = standardize(cal_fit_x_raw, cal_eval_x_raw)
    ridge_cal, lgbm_cal, _, _ = fit_predict_pair(
        cal_fit_x,
        y_fit,
        w_fit,
        cal_eval_x,
        selected_features,
        args.ridge_alpha,
        int(best_candidate["lgbm_num_leaves"]),
        args.lgbm_estimators,
        args.lgbm_learning_rate,
        args.lgbm_min_child_samples,
        args.lgbm_reg_lambda,
        args.lgbm_seeds,
        args.lgbm_subsample,
        args.lgbm_colsample_bytree,
    )
    cal_base_prediction = weights["ridge"] * ridge_cal + weights["lgbm"] * lgbm_cal
    cal_shrink_values = shrink_values_for_assets(asset_cal, shrink_info)
    cal_prediction = cal_shrink_values * cal_base_prediction
    calibration_predictions = frame.loc[cal_mask, ["row_id", "time_id", "asset_id", "target", "weight"]].copy()
    calibration_predictions["ridge_prediction"] = ridge_cal
    calibration_predictions["lgbm_prediction"] = lgbm_cal
    calibration_predictions["shrink"] = cal_shrink_values
    calibration_predictions["prediction"] = cal_prediction
    calibration_predictions["error"] = calibration_predictions["prediction"] - calibration_predictions["target"]
    calibration_predictions.to_csv(args.results_dir / "calibration_predictions.csv", index=False)

    final_train_x_raw = frame.loc[final_train_mask, selected_features].to_numpy(dtype=np.float32)
    test_x_raw = frame.loc[test_mask, selected_features].to_numpy(dtype=np.float32)
    final_train_x, test_x, mean, scale = standardize(final_train_x_raw, test_x_raw)
    y_final_train = frame.loc[final_train_mask, "target"].to_numpy(dtype=np.float32)
    w_final_train = frame.loc[final_train_mask, "weight"].to_numpy(dtype=np.float32)
    y_test = frame.loc[test_mask, "target"].to_numpy(dtype=np.float32)
    w_test = frame.loc[test_mask, "weight"].to_numpy(dtype=np.float32)
    asset_test = frame.loc[test_mask, "asset_id"].to_numpy(dtype=np.int64)

    ridge_test, lgbm_test, ridge_model, lgbm_model = fit_predict_pair(
        final_train_x,
        y_final_train,
        w_final_train,
        test_x,
        selected_features,
        args.ridge_alpha,
        int(best_candidate["lgbm_num_leaves"]),
        args.lgbm_estimators,
        args.lgbm_learning_rate,
        args.lgbm_min_child_samples,
        args.lgbm_reg_lambda,
        args.lgbm_seeds,
        args.lgbm_subsample,
        args.lgbm_colsample_bytree,
    )
    weights = {
        "ridge": float(best_candidate["cal_ridge_weight"]),
        "lgbm": float(best_candidate["cal_lgbm_weight"]),
        "shrink": float(best_candidate["cal_shrink"]),
        "shrink_mode": str(best_candidate["cal_shrink_mode"]),
        "shrink_cap": float(best_candidate["cal_shrink_cap"]),
    }
    base_prediction = weights["ridge"] * ridge_test + weights["lgbm"] * lgbm_test
    shrink_info = best_candidate["cal_shrink_info"]
    test_shrink_values = shrink_values_for_assets(asset_test, shrink_info)
    prediction = test_shrink_values * base_prediction
    test_score = weighted_zero_mean_r2(y_test, prediction, w_test)
    test_score_by_asset = score_by_asset(y_test, prediction, w_test, asset_test)

    predictions = frame.loc[test_mask, ["row_id", "time_id", "asset_id", "target", "weight"]].copy()
    predictions["ridge_prediction"] = ridge_test
    predictions["lgbm_prediction"] = lgbm_test
    predictions["shrink"] = test_shrink_values
    predictions["prediction"] = prediction
    predictions["error"] = predictions["prediction"] - predictions["target"]
    predictions.to_csv(args.results_dir / "test_predictions.csv", index=False)
    pd.DataFrame({"feature_name": selected_features}).to_csv(args.results_dir / "selected_features.csv", index=False)

    metrics = {
        "leakage_safe": True,
        "future_function_guard": {
            "feature_screen_fit_train": f"time_id <= {args.fit_train_end_time}",
            "selection_and_shrink_calibration": f"{args.cal_start_time} <= time_id <= {args.cal_end_time}",
            "holdout_test_only_for_final_score": f"{args.test_start_time} <= time_id <= {args.test_end_time}",
        },
        "data_dir": str(args.data_dir),
        "feature_source": feature_source,
        "fixed_features_file": str(args.fixed_features_file) if args.fixed_features_file is not None else None,
        "fit_train_rows": int(fit_train_mask.sum()),
        "cal_rows": int(cal_mask.sum()),
        "final_train_rows": int(final_train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "best_candidate": best_candidate,
        "unconstrained_best_candidate": all_best_candidate,
        "candidate_selection_policy": {
            "candidate_score_mode": args.candidate_score_mode,
            "max_cal_shrink": args.max_cal_shrink,
            "shrink_cap_candidates": [float(value) for value in args.shrink_cap_candidates],
            "lgbm_seeds": [int(seed) for seed in args.lgbm_seeds],
            "lgbm_subsample": float(args.lgbm_subsample),
            "lgbm_colsample_bytree": float(args.lgbm_colsample_bytree),
            "shrink_limit_field": "cal_shrink_max",
            "fallback_to_unconstrained_if_no_eligible_candidate": True,
        },
        "test_score": float(test_score),
        "test_score_by_asset": test_score_by_asset,
        "negative_asset_count": int(sum(score < 0 for score in test_score_by_asset.values())),
        "selected_feature_count": int(len(selected_features)),
        "selected_features": selected_features,
        "feature_mean": mean.astype(float).tolist(),
        "feature_scale": scale.astype(float).tolist(),
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
