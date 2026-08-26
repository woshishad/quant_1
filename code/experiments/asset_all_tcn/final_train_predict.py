from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from walk_forward_tabular import (
    find_best_blend,
    fit_predict_pair,
    score_candidate_on_calibration,
    standardize,
    weighted_zero_mean_r2,
)


BASE_COLUMNS_TRAIN = ["row_id", "time_id", "asset_id", "weight", "target"]
BASE_COLUMNS_TEST = ["row_id", "time_id", "asset_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the current best leakage-safe final protocol and predict official test rows."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset_all_final_best_protocol"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/asset_all_final_best_protocol"))
    parser.add_argument(
        "--fixed-features-file",
        type=Path,
        default=Path("results/asset_all_stable_features_100k/selected_features_stable_top128.csv"),
    )
    parser.add_argument("--sample-submission", type=Path, default=None)
    parser.add_argument("--cal-time-points", type=int, default=100_000)
    parser.add_argument(
        "--train-start-time",
        type=int,
        default=None,
        help="只读取不早于该 time_id 的训练数据；默认使用 raw train 全部历史。",
    )
    parser.add_argument(
        "--train-lookback-time-points",
        type=int,
        default=None,
        help="只使用 train_end_time 往前数的最近 N 个 time_id，便于在内存有限时做正式预测。",
    )
    parser.add_argument("--max-train-time-id", type=int, default=None)
    parser.add_argument("--test-start-time", type=int, default=None)
    parser.add_argument("--test-end-time", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--top-k-candidates", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--lgbm-num-leaves-candidates", type=int, nargs="+", default=[15, 31])
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.005)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=8000)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=500.0)
    parser.add_argument("--lgbm-subsample", type=float, default=0.7)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.7)
    parser.add_argument("--lgbm-seeds", type=int, nargs="+", default=[11, 42, 73])
    parser.add_argument("--global-candidate-score-mode", choices=["full", "mean_halves", "min_halves"], default="full")
    parser.add_argument("--per-asset-candidate-score-mode", choices=["full", "mean_halves", "min_halves"], default="min_halves")
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[1.2])
    parser.add_argument("--model-blend-step", type=float, default=0.01)
    parser.add_argument("--final-blend-min-global-weight", type=float, default=0.8)
    parser.add_argument("--final-blend-max-global-weight", type=float, default=1.0)
    parser.add_argument("--final-blend-step", type=float, default=0.01)
    parser.add_argument(
        "--per-asset-feature-mode",
        choices=["fixed_ranking", "screen"],
        default="fixed_ranking",
        help="fixed_ranking 直接复用全局特征排序；screen 会在最终 calibration 上为每个 asset 单独筛 323 个特征，较慢但更贴近 walk-forward 辅助模型。",
    )
    parser.add_argument("--per-asset-top-k-candidates", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--per-asset-lgbm-num-leaves-candidates", type=int, nargs="+", default=[7, 15])
    parser.add_argument("--per-asset-lgbm-min-child-samples", type=int, default=1000)
    parser.add_argument("--per-asset-lgbm-reg-lambda", type=float, default=300.0)
    parser.add_argument(
        "--per-asset-lgbm-seeds",
        type=int,
        nargs="+",
        default=[42],
        help="per-asset 辅助模型的 LightGBM seeds；默认单 seed，避免 15 个标的重复 bagging 过慢。",
    )
    parser.add_argument("--no-save-models", action="store_true")
    return parser.parse_args()


def parquet_paths(raw_data_dir: Path, split: str) -> list[Path]:
    paths = sorted((raw_data_dir / split).glob(f"{split}_partition_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet partitions found for split={split} under {raw_data_dir}")
    return paths


def schema_columns(paths: list[Path]) -> list[str]:
    return pq.ParquetFile(paths[0]).schema_arrow.names


def time_range(paths: list[Path]) -> tuple[int, int]:
    mins = []
    maxs = []
    for path in paths:
        table = pq.read_table(path, columns=["time_id"])
        values = table.column("time_id").to_numpy()
        mins.append(int(np.min(values)))
        maxs.append(int(np.max(values)))
    return min(mins), max(maxs)


def time_filter(min_time: int | None, max_time: int | None):
    expression = None
    if min_time is not None:
        expression = ds.field("time_id") >= int(min_time)
    if max_time is not None:
        upper = ds.field("time_id") <= int(max_time)
        expression = upper if expression is None else expression & upper
    return expression


def read_partitioned_frame(
    paths: list[Path],
    columns: list[str],
    min_time: int | None = None,
    max_time: int | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    table = dataset.to_table(columns=columns, filter=time_filter(min_time, max_time))
    if max_rows is not None:
        table = table.slice(0, int(max_rows))
    frame = table.to_pandas(split_blocks=True, self_destruct=True)
    sort_columns = ["time_id", "asset_id"] if "time_id" in frame.columns and "asset_id" in frame.columns else ["row_id"]
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def load_feature_ranking(path: Path, available_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "feature_name" in frame.columns:
        names = frame["feature_name"].astype(str).tolist()
    elif "feature_index" in frame.columns:
        feature_columns = [column for column in available_columns if column.startswith("feature_")]
        names = [feature_columns[index] for index in frame["feature_index"].astype(int).tolist()]
    else:
        raise ValueError(f"{path} must contain feature_name or feature_index")
    missing = sorted(set(names) - set(available_columns))
    if missing:
        raise ValueError(f"{path} contains unknown features: {missing[:10]}")
    ranking = pd.DataFrame({"feature_name": names})
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def normalized_weight(weight: np.ndarray) -> np.ndarray:
    return weight / max(float(np.mean(weight)), 1e-12)


def optimal_shrink(y_true: np.ndarray, prediction: np.ndarray, weight: np.ndarray, cap: float) -> float:
    denominator = float(np.sum(weight * prediction * prediction))
    if denominator <= 1e-18:
        return 0.0
    shrink = float(np.sum(weight * y_true * prediction) / denominator)
    return min(float(cap), max(0.0, shrink))


def apply_per_asset_shrink(prediction: np.ndarray, asset_id: np.ndarray, shrink_by_asset: dict[str, float]) -> np.ndarray:
    values = np.zeros(len(asset_id), dtype=np.float64)
    for asset_name, shrink in shrink_by_asset.items():
        values[asset_id == int(asset_name)] = float(shrink)
    return values * prediction


def fit_predict_pair_with_args(
    train_x: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    predict_x: np.ndarray,
    selected_features: list[str],
    leaves: int,
    args: argparse.Namespace,
    min_child_samples: int | None = None,
    reg_lambda: float | None = None,
    lgbm_seeds: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, object, list[object]]:
    return fit_predict_pair(
        train_x,
        y_train,
        w_train,
        predict_x,
        selected_features,
        args.ridge_alpha,
        int(leaves),
        args.lgbm_estimators,
        args.lgbm_learning_rate,
        min_child_samples if min_child_samples is not None else args.lgbm_min_child_samples,
        reg_lambda if reg_lambda is not None else args.lgbm_reg_lambda,
        lgbm_seeds if lgbm_seeds is not None else args.lgbm_seeds,
        args.lgbm_subsample,
        args.lgbm_colsample_bytree,
    )


def choose_global_candidate(
    frame: pd.DataFrame,
    ranking: pd.DataFrame,
    fit_mask: np.ndarray,
    cal_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y_fit = frame.loc[fit_mask, "target"].to_numpy(dtype=np.float32)
    w_fit = frame.loc[fit_mask, "weight"].to_numpy(dtype=np.float32)
    y_cal = frame.loc[cal_mask, "target"].to_numpy(dtype=np.float32)
    w_cal = frame.loc[cal_mask, "weight"].to_numpy(dtype=np.float32)
    asset_cal = frame.loc[cal_mask, "asset_id"].to_numpy(dtype=np.int64)
    time_cal = frame.loc[cal_mask, "time_id"].to_numpy(dtype=np.int64)

    rows = []
    best: dict | None = None
    for top_k in args.top_k_candidates:
        selected = ranking.head(int(top_k))["feature_name"].astype(str).tolist()
        fit_x_raw = frame.loc[fit_mask, selected].to_numpy(dtype=np.float32)
        cal_x_raw = frame.loc[cal_mask, selected].to_numpy(dtype=np.float32)
        fit_x, cal_x, _, _ = standardize(fit_x_raw, cal_x_raw)
        for leaves in args.lgbm_num_leaves_candidates:
            ridge_pred, lgbm_pred, _, _ = fit_predict_pair_with_args(
                fit_x, y_fit, w_fit, cal_x, selected, int(leaves), args
            )
            blend = find_best_blend(
                y_cal,
                w_cal,
                asset_cal,
                time_cal,
                ridge_pred,
                lgbm_pred,
                args.model_blend_step,
                "per_asset",
                args.shrink_cap_candidates,
                args.global_candidate_score_mode,
            )
            score_info = blend["score_info"]
            shrink_summary = blend["shrink_summary"]
            row = {
                "scope": "global",
                "top_k": int(top_k),
                "lgbm_num_leaves": int(leaves),
                "cal_score": float(blend["score"]),
                "cal_full_score": float(score_info["full_score"]),
                "cal_first_half_score": float(score_info["first_half_score"]),
                "cal_second_half_score": float(score_info["second_half_score"]),
                "cal_ridge_weight": float(blend["weights"]["ridge"]),
                "cal_lgbm_weight": float(blend["weights"]["lgbm"]),
                "cal_shrink": float(shrink_summary["cal_shrink"]),
                "cal_shrink_mean": float(shrink_summary["cal_shrink_mean"]),
                "cal_shrink_max": float(shrink_summary["cal_shrink_max"]),
                "cal_shrink_info": blend["shrink_info"],
                "selected_features": selected,
            }
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items() if key not in {"selected_features", "cal_shrink_info"}}))
            if best is None or row["cal_score"] > best["cal_score"]:
                best = row
    if best is None:
        raise ValueError("no global candidate evaluated")

    cal_predictions = predict_global_for_frame(
        frame,
        fit_mask,
        cal_mask,
        best,
        args,
        include_target=True,
    )
    rows_for_csv = [{key: value for key, value in row.items() if key not in {"selected_features", "cal_shrink_info"}} for row in rows]
    return best, pd.DataFrame(rows_for_csv), cal_predictions


def predict_global_for_frame(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    predict_mask: np.ndarray,
    candidate: dict,
    args: argparse.Namespace,
    include_target: bool,
    return_models: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    selected = [str(feature) for feature in candidate["selected_features"]]
    train_x_raw = frame.loc[train_mask, selected].to_numpy(dtype=np.float32)
    predict_x_raw = frame.loc[predict_mask, selected].to_numpy(dtype=np.float32)
    train_x, predict_x, _, _ = standardize(train_x_raw, predict_x_raw)
    y_train = frame.loc[train_mask, "target"].to_numpy(dtype=np.float32)
    w_train = frame.loc[train_mask, "weight"].to_numpy(dtype=np.float32)
    ridge_pred, lgbm_pred, ridge_model, lgbm_models = fit_predict_pair_with_args(
        train_x,
        y_train,
        w_train,
        predict_x,
        selected,
        int(candidate["lgbm_num_leaves"]),
        args,
    )
    base_prediction = float(candidate["cal_ridge_weight"]) * ridge_pred + float(candidate["cal_lgbm_weight"]) * lgbm_pred
    shrink_info = candidate["cal_shrink_info"]
    shrink_values = np.full(len(base_prediction), float(shrink_info["global"]), dtype=np.float64)
    if shrink_info.get("mode") == "per_asset":
        asset_predict = frame.loc[predict_mask, "asset_id"].to_numpy(dtype=np.int64)
        for asset_name, shrink in shrink_info.get("by_asset", {}).items():
            shrink_values[asset_predict == int(asset_name)] = float(shrink)
    columns = BASE_COLUMNS_TRAIN if include_target else BASE_COLUMNS_TEST
    output = frame.loc[predict_mask, columns].copy()
    output["global_ridge_prediction"] = ridge_pred
    output["global_lgbm_prediction"] = lgbm_pred
    output["global_base_prediction"] = base_prediction
    output["global_shrink_value"] = shrink_values
    output["global_prediction"] = shrink_values * base_prediction
    if return_models:
        return output, {
            "ridge_model": ridge_model,
            "lgbm_models": lgbm_models,
            "selected_features": selected,
            "candidate": candidate,
        }
    return output


def read_feature_values_for_screening(
    train_paths: list[Path],
    feature_name: str,
    train_end_time: int,
) -> pd.DataFrame:
    frame = read_partitioned_frame(
        train_paths,
        ["row_id", "time_id", "asset_id", feature_name],
        max_time=train_end_time,
    )
    return frame[["row_id", feature_name]]


def screen_features_by_asset_partitioned(
    train_paths: list[Path],
    feature_columns: list[str],
    base: pd.DataFrame,
    fit_mask: np.ndarray,
    cal_mask: np.ndarray,
    train_end_time: int,
) -> pd.DataFrame:
    # 这个筛选严格只看 fit/cal 两段：fit 段估计单因子方向，cal 段打分；不接触 test。
    assets = sorted(base["asset_id"].unique().astype(int).tolist())
    row_id = base["row_id"].to_numpy(dtype=np.int64)
    asset_values = base["asset_id"].to_numpy(dtype=np.int64)
    y_values = base["target"].to_numpy(dtype=np.float64)
    w_values = base["weight"].to_numpy(dtype=np.float64)
    rows = []

    for feature_index, feature_name in enumerate(feature_columns):
        feature_frame = read_feature_values_for_screening(train_paths, feature_name, train_end_time)
        merged = pd.DataFrame({"row_id": row_id}).merge(feature_frame, on="row_id", how="left")
        values = merged[feature_name].to_numpy(dtype=np.float64)

        for asset in assets:
            asset_mask = asset_values == asset
            asset_fit = fit_mask & asset_mask
            asset_cal = cal_mask & asset_mask
            train_x_raw = values[asset_fit]
            cal_x_raw = values[asset_cal]
            if len(train_x_raw) == 0 or len(cal_x_raw) == 0:
                continue
            mean = float(np.nanmean(train_x_raw))
            scale = float(np.nanstd(train_x_raw))
            if not np.isfinite(scale) or scale < 1e-6:
                scale = 1.0
            train_x = np.nan_to_num((train_x_raw - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
            cal_x = np.nan_to_num((cal_x_raw - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
            train_weight = normalized_weight(w_values[asset_fit])
            denominator = float(np.sum(train_weight * train_x * train_x))
            coef = 0.0 if denominator <= 1e-18 else float(np.sum(train_weight * train_x * y_values[asset_fit]) / denominator)
            prediction = coef * cal_x
            rows.append(
                {
                    "asset_id": int(asset),
                    "feature_index": int(feature_index),
                    "feature_name": feature_name,
                    "cal_raw_score": weighted_zero_mean_r2(y_values[asset_cal], prediction, w_values[asset_cal]),
                    "coef": coef,
                }
            )
        if (feature_index + 1) % 50 == 0 or feature_index + 1 == len(feature_columns):
            print(f"screened {feature_index + 1}/{len(feature_columns)} features for per-asset auxiliary")

    ranking = pd.DataFrame(rows).sort_values(["asset_id", "cal_raw_score"], ascending=[True, False])
    ranking["rank"] = ranking.groupby("asset_id").cumcount() + 1
    return ranking.reset_index(drop=True)


def fixed_per_asset_ranking(global_ranking: pd.DataFrame, assets: list[int]) -> pd.DataFrame:
    rows = []
    for asset in assets:
        for _, row in global_ranking.iterrows():
            rows.append({"asset_id": int(asset), "feature_name": str(row["feature_name"]), "rank": int(row["rank"])})
    return pd.DataFrame(rows)


def choose_per_asset_candidates(
    frame: pd.DataFrame,
    ranking: pd.DataFrame,
    fit_mask: np.ndarray,
    cal_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, dict], pd.DataFrame, pd.DataFrame]:
    candidates: dict[str, dict] = {}
    candidate_rows = []
    prediction_parts = []
    assets = sorted(frame["asset_id"].unique().astype(int).tolist())

    for asset in assets:
        asset_mask = frame["asset_id"].to_numpy(dtype=np.int64) == asset
        asset_fit = fit_mask & asset_mask
        asset_cal = cal_mask & asset_mask
        asset_ranking = ranking.loc[ranking["asset_id"] == asset].sort_values("rank")
        y_fit = frame.loc[asset_fit, "target"].to_numpy(dtype=np.float32)
        w_fit = frame.loc[asset_fit, "weight"].to_numpy(dtype=np.float32)
        y_cal = frame.loc[asset_cal, "target"].to_numpy(dtype=np.float32)
        w_cal = frame.loc[asset_cal, "weight"].to_numpy(dtype=np.float32)
        time_cal = frame.loc[asset_cal, "time_id"].to_numpy(dtype=np.int64)
        asset_cal_values = frame.loc[asset_cal, "asset_id"].to_numpy(dtype=np.int64)

        best: dict | None = None
        for top_k in args.per_asset_top_k_candidates:
            selected = asset_ranking.head(int(top_k))["feature_name"].astype(str).tolist()
            fit_x_raw = frame.loc[asset_fit, selected].to_numpy(dtype=np.float32)
            cal_x_raw = frame.loc[asset_cal, selected].to_numpy(dtype=np.float32)
            fit_x, cal_x, _, _ = standardize(fit_x_raw, cal_x_raw)
            for leaves in args.per_asset_lgbm_num_leaves_candidates:
                ridge_pred, lgbm_pred, _, _ = fit_predict_pair_with_args(
                    fit_x,
                    y_fit,
                    w_fit,
                    cal_x,
                    selected,
                    int(leaves),
                    args,
                    min_child_samples=args.per_asset_lgbm_min_child_samples,
                    reg_lambda=args.per_asset_lgbm_reg_lambda,
                    lgbm_seeds=args.per_asset_lgbm_seeds,
                )
                blend = find_best_blend(
                    y_cal,
                    w_cal,
                    asset_cal_values,
                    time_cal,
                    ridge_pred,
                    lgbm_pred,
                    args.model_blend_step,
                    "global",
                    args.shrink_cap_candidates,
                    args.per_asset_candidate_score_mode,
                )
                score_info = blend["score_info"]
                row = {
                    "scope": "per_asset",
                    "asset_id": int(asset),
                    "top_k": int(top_k),
                    "lgbm_num_leaves": int(leaves),
                    "cal_score": float(blend["score"]),
                    "cal_full_score": float(score_info["full_score"]),
                    "cal_first_half_score": float(score_info["first_half_score"]),
                    "cal_second_half_score": float(score_info["second_half_score"]),
                    "cal_ridge_weight": float(blend["weights"]["ridge"]),
                    "cal_lgbm_weight": float(blend["weights"]["lgbm"]),
                    "cal_shrink": float(blend["weights"]["shrink"]),
                    "selected_features": selected,
                }
                candidate_rows.append({key: value for key, value in row.items() if key != "selected_features"})
                if best is None or row["cal_score"] > best["cal_score"]:
                    best = row
        if best is None:
            raise ValueError(f"no per-asset candidate evaluated for asset={asset}")
        candidates[str(asset)] = best
        prediction_parts.append(predict_per_asset_for_frame(frame, asset_fit, asset_cal, best, args, include_target=True))
        print(json.dumps({key: value for key, value in best.items() if key != "selected_features"}))

    return candidates, pd.DataFrame(candidate_rows), pd.concat(prediction_parts, ignore_index=True)


def predict_per_asset_for_frame(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    predict_mask: np.ndarray,
    candidate: dict,
    args: argparse.Namespace,
    include_target: bool,
    return_models: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    selected = [str(feature) for feature in candidate["selected_features"]]
    train_x_raw = frame.loc[train_mask, selected].to_numpy(dtype=np.float32)
    predict_x_raw = frame.loc[predict_mask, selected].to_numpy(dtype=np.float32)
    train_x, predict_x, _, _ = standardize(train_x_raw, predict_x_raw)
    y_train = frame.loc[train_mask, "target"].to_numpy(dtype=np.float32)
    w_train = frame.loc[train_mask, "weight"].to_numpy(dtype=np.float32)
    ridge_pred, lgbm_pred, ridge_model, lgbm_models = fit_predict_pair_with_args(
        train_x,
        y_train,
        w_train,
        predict_x,
        selected,
        int(candidate["lgbm_num_leaves"]),
        args,
        min_child_samples=args.per_asset_lgbm_min_child_samples,
        reg_lambda=args.per_asset_lgbm_reg_lambda,
        lgbm_seeds=args.per_asset_lgbm_seeds,
    )
    base_prediction = float(candidate["cal_ridge_weight"]) * ridge_pred + float(candidate["cal_lgbm_weight"]) * lgbm_pred
    prediction = float(candidate["cal_shrink"]) * base_prediction
    columns = BASE_COLUMNS_TRAIN if include_target else BASE_COLUMNS_TEST
    output = frame.loc[predict_mask, columns].copy()
    output["per_asset_ridge_prediction"] = ridge_pred
    output["per_asset_lgbm_prediction"] = lgbm_pred
    output["per_asset_base_prediction"] = base_prediction
    output["per_asset_shrink"] = float(candidate["cal_shrink"])
    output["per_asset_prediction"] = prediction
    if return_models:
        return output, {
            "ridge_model": ridge_model,
            "lgbm_models": lgbm_models,
            "selected_features": selected,
            "candidate": candidate,
        }
    return output


def predict_all_per_asset_for_frame(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    predict_mask: np.ndarray,
    candidates: dict[str, dict],
    args: argparse.Namespace,
    include_target: bool,
    return_models: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    parts = []
    model_payload = {}
    asset_values = frame["asset_id"].to_numpy(dtype=np.int64)
    for asset_name, candidate in candidates.items():
        asset = int(asset_name)
        asset_train = train_mask & (asset_values == asset)
        asset_predict = predict_mask & (asset_values == asset)
        if not np.any(asset_predict):
            continue
        if return_models:
            prediction_part, asset_models = predict_per_asset_for_frame(
                frame, asset_train, asset_predict, candidate, args, include_target, return_models=True
            )
            parts.append(prediction_part)
            model_payload[str(asset)] = asset_models
        else:
            parts.append(predict_per_asset_for_frame(frame, asset_train, asset_predict, candidate, args, include_target))
    predictions = pd.concat(parts, ignore_index=True).sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    if return_models:
        return predictions, model_payload
    return predictions


def search_final_blend(cal_frame: pd.DataFrame, args: argparse.Namespace) -> dict:
    y = cal_frame["target"].to_numpy(dtype=np.float64)
    w = cal_frame["weight"].to_numpy(dtype=np.float64)
    global_pred = cal_frame["global_prediction"].to_numpy(dtype=np.float64)
    aux_pred = cal_frame["per_asset_prediction"].to_numpy(dtype=np.float64)
    best = {"cal_score": -np.inf, "global_weight": args.final_blend_min_global_weight}
    for global_weight in np.arange(
        args.final_blend_min_global_weight,
        args.final_blend_max_global_weight + 1e-12,
        args.final_blend_step,
    ):
        prediction = global_weight * global_pred + (1.0 - global_weight) * aux_pred
        score = weighted_zero_mean_r2(y, prediction, w)
        if score > best["cal_score"]:
            best = {"cal_score": float(score), "global_weight": float(global_weight)}
    best["per_asset_weight"] = float(1.0 - best["global_weight"])
    return best


def merge_prediction_components(global_frame: pd.DataFrame, per_asset_frame: pd.DataFrame, include_target: bool) -> pd.DataFrame:
    base_cols = BASE_COLUMNS_TRAIN if include_target else BASE_COLUMNS_TEST
    per_asset_cols = ["row_id", "per_asset_prediction", "per_asset_base_prediction", "per_asset_shrink"]
    merged = global_frame.merge(per_asset_frame[per_asset_cols], on="row_id", how="left")
    merged["per_asset_prediction"] = merged["per_asset_prediction"].fillna(0.0)
    merged["per_asset_base_prediction"] = merged["per_asset_base_prediction"].fillna(0.0)
    merged["per_asset_shrink"] = merged["per_asset_shrink"].fillna(0.0)
    return merged[base_cols + [column for column in merged.columns if column not in base_cols]]


def reorder_like_sample(submission: pd.DataFrame, sample_path: Path | None) -> pd.DataFrame:
    if sample_path is None or not sample_path.exists():
        return submission.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    sample = pd.read_csv(sample_path, usecols=["row_id"])
    if len(sample) != len(submission):
        # smoke 子集不强制对齐完整 sample，只按 row_id 输出。
        return submission.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    return sample.merge(submission, on="row_id", how="left")


def save_pickle(path: Path, payload: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
    train_end_time = min(train_max_time_available, args.max_train_time_id) if args.max_train_time_id is not None else train_max_time_available
    if args.train_lookback_time_points is not None:
        train_start_time = max(train_min_time, train_end_time - int(args.train_lookback_time_points) + 1)
    elif args.train_start_time is not None:
        train_start_time = max(train_min_time, int(args.train_start_time))
    else:
        train_start_time = train_min_time
    fit_train_end_time = train_end_time - int(args.cal_time_points)
    cal_start_time = fit_train_end_time + 1
    if fit_train_end_time < train_start_time:
        raise ValueError("cal-time-points is too large for the selected train time range")

    available_columns = schema_columns(train_paths)
    feature_columns = [column for column in available_columns if column.startswith("feature_")]
    global_ranking = load_feature_ranking(args.fixed_features_file, available_columns)
    max_global_top_k = max(args.top_k_candidates)
    max_aux_top_k = max(args.per_asset_top_k_candidates)
    max_feature_count = max(max_global_top_k, max_aux_top_k, len(global_ranking))
    if max_global_top_k > len(global_ranking):
        raise ValueError(f"global top_k requires {max_global_top_k} features, but ranking has {len(global_ranking)}")
    fixed_features = global_ranking.head(max_feature_count)["feature_name"].astype(str).tolist()

    print(
        f"Final protocol split: fit<= {fit_train_end_time}, cal={cal_start_time}..{train_end_time}, "
        f"test={args.test_start_time or test_min_time}..{args.test_end_time or test_max_time_available}"
    )
    train_frame = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + fixed_features,
        min_time=train_start_time,
        max_time=train_end_time,
    )
    train_time = train_frame["time_id"].to_numpy(dtype=np.int64)
    fit_mask = train_time <= fit_train_end_time
    cal_mask = (train_time >= cal_start_time) & (train_time <= train_end_time)
    final_train_mask = train_time <= train_end_time

    best_global, global_candidates, global_cal = choose_global_candidate(train_frame, global_ranking, fit_mask, cal_mask, args)
    global_candidates.to_csv(args.results_dir / "global_candidate_metrics.csv", index=False)
    pd.DataFrame({"feature_name": best_global["selected_features"]}).to_csv(
        args.results_dir / "global_selected_features.csv", index=False
    )

    assets = sorted(train_frame["asset_id"].unique().astype(int).tolist())
    if args.per_asset_feature_mode == "screen":
        aux_ranking = screen_features_by_asset_partitioned(
            train_paths,
            feature_columns,
            train_frame[BASE_COLUMNS_TRAIN],
            fit_mask,
            cal_mask,
            train_end_time,
        )
    else:
        aux_ranking = fixed_per_asset_ranking(global_ranking.head(max_aux_top_k), assets)
    aux_ranking.to_csv(args.results_dir / "per_asset_feature_ranking.csv", index=False)

    aux_candidates, aux_candidate_frame, aux_cal = choose_per_asset_candidates(
        train_frame,
        aux_ranking,
        fit_mask,
        cal_mask,
        args,
    )
    aux_candidate_frame.to_csv(args.results_dir / "per_asset_candidate_metrics.csv", index=False)
    per_asset_selected_rows = []
    for asset_name, candidate in aux_candidates.items():
        for rank, feature_name in enumerate(candidate["selected_features"], start=1):
            per_asset_selected_rows.append({"asset_id": int(asset_name), "rank": rank, "feature_name": feature_name})
    pd.DataFrame(per_asset_selected_rows).to_csv(args.results_dir / "per_asset_selected_features.csv", index=False)

    cal_components = merge_prediction_components(global_cal, aux_cal, include_target=True)
    final_blend = search_final_blend(cal_components, args)
    cal_components["prediction"] = (
        final_blend["global_weight"] * cal_components["global_prediction"]
        + final_blend["per_asset_weight"] * cal_components["per_asset_prediction"]
    )
    cal_components["error"] = cal_components["prediction"] - cal_components["target"]
    cal_components.to_csv(args.results_dir / "calibration_predictions.csv", index=False)

    test_min = args.test_start_time if args.test_start_time is not None else test_min_time
    test_max = args.test_end_time if args.test_end_time is not None else test_max_time_available
    test_frame = read_partitioned_frame(
        test_paths,
        BASE_COLUMNS_TEST + fixed_features,
        min_time=test_min,
        max_time=test_max,
        max_rows=args.max_test_rows,
    )
    # test 没有 target/weight；这里填 0 只是为了和 train 拼成同一张表，训练掩码不会选中这些 test 行。
    test_for_prediction = test_frame.copy()
    test_for_prediction["weight"] = 0.0
    test_for_prediction["target"] = 0.0
    combined_frame = pd.concat(
        [
            train_frame.loc[final_train_mask, BASE_COLUMNS_TRAIN + fixed_features],
            test_for_prediction[BASE_COLUMNS_TRAIN + fixed_features],
        ],
        ignore_index=True,
    ).sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    combined_time = combined_frame["time_id"].to_numpy(dtype=np.int64)
    combined_train_mask = combined_time <= train_end_time
    combined_test_mask = combined_time > train_end_time

    if args.no_save_models:
        global_test = predict_global_for_frame(
            combined_frame,
            combined_train_mask,
            combined_test_mask,
            best_global,
            args,
            include_target=False,
        )
        aux_test = predict_all_per_asset_for_frame(
            combined_frame,
            combined_train_mask,
            combined_test_mask,
            aux_candidates,
            args,
            include_target=False,
        )
        trained_model_payload = None
    else:
        global_test, global_models = predict_global_for_frame(
            combined_frame,
            combined_train_mask,
            combined_test_mask,
            best_global,
            args,
            include_target=False,
            return_models=True,
        )
        aux_test, aux_models = predict_all_per_asset_for_frame(
            combined_frame,
            combined_train_mask,
            combined_test_mask,
            aux_candidates,
            args,
            include_target=False,
            return_models=True,
        )
        trained_model_payload = {"global": global_models, "per_asset": aux_models}
    test_predictions = merge_prediction_components(global_test, aux_test, include_target=False)
    test_predictions["prediction"] = (
        final_blend["global_weight"] * test_predictions["global_prediction"]
        + final_blend["per_asset_weight"] * test_predictions["per_asset_prediction"]
    )
    test_predictions.to_csv(args.results_dir / "final_test_predictions.csv", index=False)
    submission = test_predictions[["row_id", "prediction"]].rename(columns={"prediction": "target"})
    submission = reorder_like_sample(submission, args.sample_submission)
    submission.to_csv(args.results_dir / "submission.csv", index=False)

    metrics = {
        "leakage_safe": True,
        "environment": "quant-competition-wsl",
        "raw_data_dir": str(args.raw_data_dir),
        "future_function_guard": {
            "fit_train": f"{train_start_time} <= time_id <= {fit_train_end_time}",
            "calibration": f"{cal_start_time} <= time_id <= {train_end_time}",
            "official_test_prediction_only": f"{test_min} <= time_id <= {test_max}",
        },
        "train_window": {
            "raw_train_min_time": int(train_min_time),
            "raw_train_max_time": int(train_max_time_available),
            "used_train_start_time": int(train_start_time),
            "used_train_end_time": int(train_end_time),
            "train_lookback_time_points": (
                int(args.train_lookback_time_points) if args.train_lookback_time_points is not None else None
            ),
        },
        "rows": {
            "fit_train": int(fit_mask.sum()),
            "calibration": int(cal_mask.sum()),
            "final_train": int(final_train_mask.sum()),
            "test_predicted": int(len(test_predictions)),
        },
        "global_candidate": {key: value for key, value in best_global.items() if key not in {"selected_features", "cal_shrink_info"}},
        "global_selected_feature_count": int(len(best_global["selected_features"])),
        "per_asset_feature_mode": args.per_asset_feature_mode,
        "per_asset_candidate_count": int(len(aux_candidates)),
        "final_blend": final_blend,
        "calibration_score": weighted_zero_mean_r2(
            cal_components["target"].to_numpy(dtype=np.float64),
            cal_components["prediction"].to_numpy(dtype=np.float64),
            cal_components["weight"].to_numpy(dtype=np.float64),
        ),
        "calibration_global_score": weighted_zero_mean_r2(
            cal_components["target"].to_numpy(dtype=np.float64),
            cal_components["global_prediction"].to_numpy(dtype=np.float64),
            cal_components["weight"].to_numpy(dtype=np.float64),
        ),
        "calibration_per_asset_score": weighted_zero_mean_r2(
            cal_components["target"].to_numpy(dtype=np.float64),
            cal_components["per_asset_prediction"].to_numpy(dtype=np.float64),
            cal_components["weight"].to_numpy(dtype=np.float64),
        ),
        "output_files": {
            "submission": str(args.results_dir / "submission.csv"),
            "final_test_predictions": str(args.results_dir / "final_test_predictions.csv"),
            "calibration_predictions": str(args.results_dir / "calibration_predictions.csv"),
        },
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.model_dir / "metadata.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not args.no_save_models:
        save_pickle(args.model_dir / "final_protocol_candidates.pkl", {"global": best_global, "per_asset": aux_candidates})
        save_pickle(args.model_dir / "final_trained_models.pkl", trained_model_payload)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
