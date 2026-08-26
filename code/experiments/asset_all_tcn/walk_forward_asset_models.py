from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from walk_forward_tabular import (
    find_best_blend,
    fit_predict_pair,
    score_by_asset,
    standardize,
    weighted_zero_mean_r2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe per-asset tabular walk-forward experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/asset_all_time100000"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset_all_walk_forward_asset_models_100k"))
    parser.add_argument("--fit-train-end-time", type=int, default=59_999)
    parser.add_argument("--cal-start-time", type=int, default=60_000)
    parser.add_argument("--cal-end-time", type=int, default=79_999)
    parser.add_argument("--test-start-time", type=int, default=80_000)
    parser.add_argument("--test-end-time", type=int, default=99_999)
    parser.add_argument("--top-k-candidates", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--lgbm-num-leaves-candidates", type=int, nargs="+", default=[7, 15, 31])
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.005)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=1000)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=300.0)
    parser.add_argument("--blend-step", type=float, default=0.02)
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    parser.add_argument(
        "--candidate-score-mode",
        choices=["full", "mean_halves", "min_halves"],
        default="full",
        help="每个标的在 calibration 段上选择候选参数的方式。",
    )
    return parser.parse_args()


def normalize_weight(weight: np.ndarray) -> np.ndarray:
    return weight / max(float(np.mean(weight)), 1e-12)


def screen_features_by_asset(
    data_path: Path,
    feature_columns: list[str],
    base: pd.DataFrame,
    fit_train_mask: np.ndarray,
    cal_mask: np.ndarray,
) -> pd.DataFrame:
    # 这里按标的单独筛因子：同一个 feature 在不同 asset 上可能方向和有效性完全不同。
    # 筛选只用 fit_train 拟合单因子方向，只用 calibration 打分，不触碰未来 holdout。
    asset_values = base["asset_id"].to_numpy(dtype=np.int64)
    y_values = base["target"].to_numpy(dtype=np.float64)
    w_values = base["weight"].to_numpy(dtype=np.float64)
    assets = sorted(np.unique(asset_values))
    asset_masks = {
        int(asset): {
            "fit": fit_train_mask & (asset_values == asset),
            "cal": cal_mask & (asset_values == asset),
        }
        for asset in assets
    }

    rows = []
    sorted_index = base.index.to_numpy()
    for feature_index, feature_name in enumerate(feature_columns):
        feature_series = pd.read_parquet(data_path, columns=[feature_name])[feature_name]
        values = feature_series.loc[sorted_index].to_numpy(dtype=np.float64)

        for asset in assets:
            masks = asset_masks[int(asset)]
            fit_mask = masks["fit"]
            asset_cal_mask = masks["cal"]
            train_x_raw = values[fit_mask]
            cal_x_raw = values[asset_cal_mask]
            y_train = y_values[fit_mask]
            y_cal = y_values[asset_cal_mask]
            w_train = w_values[fit_mask]
            w_cal = w_values[asset_cal_mask]

            mean = float(np.nanmean(train_x_raw))
            scale = float(np.nanstd(train_x_raw))
            if not np.isfinite(scale) or scale < 1e-6:
                scale = 1.0
            train_x = np.nan_to_num((train_x_raw - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
            cal_x = np.nan_to_num((cal_x_raw - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)

            train_weight = normalize_weight(w_train)
            denominator = float(np.sum(train_weight * train_x * train_x))
            coef = 0.0 if denominator <= 1e-18 else float(np.sum(train_weight * train_x * y_train) / denominator)
            prediction = coef * cal_x
            score = weighted_zero_mean_r2(y_cal, prediction, w_cal)
            rows.append(
                {
                    "asset_id": int(asset),
                    "feature_index": int(feature_index),
                    "feature_name": feature_name,
                    "cal_raw_score": float(score),
                    "coef": float(coef),
                }
            )

        if (feature_index + 1) % 50 == 0 or feature_index + 1 == len(feature_columns):
            print(f"screened {feature_index + 1}/{len(feature_columns)} features for {len(assets)} assets")

    ranking = pd.DataFrame(rows).sort_values(["asset_id", "cal_raw_score"], ascending=[True, False])
    ranking["rank"] = ranking.groupby("asset_id").cumcount() + 1
    return ranking.reset_index(drop=True)


def selected_feature_union(ranking: pd.DataFrame, max_top_k: int) -> list[str]:
    selected = ranking.loc[ranking["rank"] <= max_top_k, "feature_name"].astype(str).tolist()
    # 保留第一次出现的顺序，避免 set 排序改变 LightGBM 的列顺序记录。
    return list(dict.fromkeys(selected))


def choose_asset_candidate(
    asset: int,
    ranking: pd.DataFrame,
    frame: pd.DataFrame,
    fit_train_mask: np.ndarray,
    cal_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    # 对单个 asset 选择 top_k、树复杂度、Ridge/LightGBM 融合权重和 shrink。
    asset_mask = frame["asset_id"].to_numpy(dtype=np.int64) == asset
    fit_mask = fit_train_mask & asset_mask
    asset_cal_mask = cal_mask & asset_mask
    y_fit = frame.loc[fit_mask, "target"].to_numpy(dtype=np.float32)
    w_fit = frame.loc[fit_mask, "weight"].to_numpy(dtype=np.float32)
    y_cal = frame.loc[asset_cal_mask, "target"].to_numpy(dtype=np.float32)
    w_cal = frame.loc[asset_cal_mask, "weight"].to_numpy(dtype=np.float32)
    asset_cal = frame.loc[asset_cal_mask, "asset_id"].to_numpy(dtype=np.int64)
    time_cal = frame.loc[asset_cal_mask, "time_id"].to_numpy(dtype=np.int64)

    asset_ranking = ranking.loc[ranking["asset_id"] == asset].sort_values("rank")
    candidate_rows = []
    best_candidate: dict | None = None
    for top_k in args.top_k_candidates:
        selected_features = asset_ranking.head(int(top_k))["feature_name"].astype(str).tolist()
        fit_x_raw = frame.loc[fit_mask, selected_features].to_numpy(dtype=np.float32)
        cal_x_raw = frame.loc[asset_cal_mask, selected_features].to_numpy(dtype=np.float32)
        fit_x, cal_x, _, _ = standardize(fit_x_raw, cal_x_raw)

        for leaves in args.lgbm_num_leaves_candidates:
            ridge_pred, lgbm_pred, _, _ = fit_predict_pair(
                fit_x,
                y_fit,
                w_fit,
                cal_x,
                selected_features,
                args.ridge_alpha,
                int(leaves),
                args.lgbm_estimators,
                args.lgbm_learning_rate,
                args.lgbm_min_child_samples,
                args.lgbm_reg_lambda,
            )
            blend = find_best_blend(
                y_cal,
                w_cal,
                asset_cal,
                time_cal,
                ridge_pred,
                lgbm_pred,
                args.blend_step,
                "global",
                args.shrink_cap_candidates,
                args.candidate_score_mode,
            )
            score_info = blend["score_info"]
            row = {
                "asset_id": int(asset),
                "top_k": int(top_k),
                "lgbm_num_leaves": int(leaves),
                "cal_score": float(blend["score"]),
                "cal_score_mode": args.candidate_score_mode,
                "cal_full_score": float(score_info["full_score"]),
                "cal_first_half_score": float(score_info["first_half_score"]),
                "cal_second_half_score": float(score_info["second_half_score"]),
                "cal_ridge_weight": float(blend["weights"]["ridge"]),
                "cal_lgbm_weight": float(blend["weights"]["lgbm"]),
                "cal_shrink_cap": float(blend["weights"]["shrink_cap"]),
                "cal_shrink": float(blend["weights"]["shrink"]),
                "selected_features": selected_features,
            }
            candidate_rows.append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "selected_features"}))
            if best_candidate is None or row["cal_score"] > best_candidate["cal_score"]:
                best_candidate = row

    if best_candidate is None:
        raise ValueError(f"asset {asset} has no evaluated candidate")
    return best_candidate, candidate_rows


def fit_predict_asset_holdout(
    asset: int,
    frame: pd.DataFrame,
    final_train_mask: np.ndarray,
    test_mask: np.ndarray,
    candidate: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    # 用 calibration 结束前的全部历史重新训练，再把 calibration 里选出的融合/shrink 原样应用到未来 holdout。
    asset_mask = frame["asset_id"].to_numpy(dtype=np.int64) == asset
    train_mask = final_train_mask & asset_mask
    asset_test_mask = test_mask & asset_mask
    selected_features = [str(feature) for feature in candidate["selected_features"]]

    train_x_raw = frame.loc[train_mask, selected_features].to_numpy(dtype=np.float32)
    test_x_raw = frame.loc[asset_test_mask, selected_features].to_numpy(dtype=np.float32)
    train_x, test_x, _, _ = standardize(train_x_raw, test_x_raw)
    y_train = frame.loc[train_mask, "target"].to_numpy(dtype=np.float32)
    w_train = frame.loc[train_mask, "weight"].to_numpy(dtype=np.float32)

    ridge_pred, lgbm_pred, _, _ = fit_predict_pair(
        train_x,
        y_train,
        w_train,
        test_x,
        selected_features,
        args.ridge_alpha,
        int(candidate["lgbm_num_leaves"]),
        args.lgbm_estimators,
        args.lgbm_learning_rate,
        args.lgbm_min_child_samples,
        args.lgbm_reg_lambda,
    )

    prediction = (
        float(candidate["cal_shrink"])
        * (float(candidate["cal_ridge_weight"]) * ridge_pred + float(candidate["cal_lgbm_weight"]) * lgbm_pred)
    )
    output = frame.loc[asset_test_mask, ["row_id", "time_id", "asset_id", "target", "weight"]].copy()
    output["ridge_prediction"] = ridge_pred
    output["lgbm_prediction"] = lgbm_pred
    output["shrink"] = float(candidate["cal_shrink"])
    output["prediction"] = prediction
    output["error"] = output["prediction"] - output["target"]
    return output


def fit_predict_asset_calibration(
    asset: int,
    frame: pd.DataFrame,
    fit_train_mask: np.ndarray,
    cal_mask: np.ndarray,
    candidate: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    # calibration 预测必须由 fit_train 段训练得到，不能用 calibration 自己参与拟合。
    # 这个文件用于后续二层融合学习权重，仍然保持时间顺序无泄漏。
    asset_mask = frame["asset_id"].to_numpy(dtype=np.int64) == asset
    train_mask = fit_train_mask & asset_mask
    asset_cal_mask = cal_mask & asset_mask
    selected_features = [str(feature) for feature in candidate["selected_features"]]

    train_x_raw = frame.loc[train_mask, selected_features].to_numpy(dtype=np.float32)
    cal_x_raw = frame.loc[asset_cal_mask, selected_features].to_numpy(dtype=np.float32)
    train_x, cal_x, _, _ = standardize(train_x_raw, cal_x_raw)
    y_train = frame.loc[train_mask, "target"].to_numpy(dtype=np.float32)
    w_train = frame.loc[train_mask, "weight"].to_numpy(dtype=np.float32)

    ridge_pred, lgbm_pred, _, _ = fit_predict_pair(
        train_x,
        y_train,
        w_train,
        cal_x,
        selected_features,
        args.ridge_alpha,
        int(candidate["lgbm_num_leaves"]),
        args.lgbm_estimators,
        args.lgbm_learning_rate,
        args.lgbm_min_child_samples,
        args.lgbm_reg_lambda,
    )

    prediction = (
        float(candidate["cal_shrink"])
        * (float(candidate["cal_ridge_weight"]) * ridge_pred + float(candidate["cal_lgbm_weight"]) * lgbm_pred)
    )
    output = frame.loc[asset_cal_mask, ["row_id", "time_id", "asset_id", "target", "weight"]].copy()
    output["ridge_prediction"] = ridge_pred
    output["lgbm_prediction"] = lgbm_pred
    output["shrink"] = float(candidate["cal_shrink"])
    output["prediction"] = prediction
    output["error"] = output["prediction"] - output["target"]
    return output


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

    ranking = screen_features_by_asset(data_path, feature_columns, base, fit_train_mask, cal_mask)
    ranking.to_csv(args.results_dir / "feature_ranking_by_asset.csv", index=False)

    max_top_k = max(args.top_k_candidates)
    feature_names = selected_feature_union(ranking, max_top_k)
    frame = pd.read_parquet(data_path, columns=base_columns + feature_names).sort_values(
        ["time_id", "asset_id"], kind="mergesort"
    )

    assets = sorted(frame["asset_id"].astype(int).unique().tolist())
    best_candidates = {}
    all_candidate_rows = []
    calibration_parts = []
    prediction_parts = []
    selected_feature_rows = []
    for asset in assets:
        best_candidate, candidate_rows = choose_asset_candidate(asset, ranking, frame, fit_train_mask, cal_mask, args)
        best_candidates[str(asset)] = {key: value for key, value in best_candidate.items() if key != "selected_features"}
        all_candidate_rows.extend(candidate_rows)
        calibration_parts.append(fit_predict_asset_calibration(asset, frame, fit_train_mask, cal_mask, best_candidate, args))
        prediction_parts.append(fit_predict_asset_holdout(asset, frame, final_train_mask, test_mask, best_candidate, args))
        for rank, feature_name in enumerate(best_candidate["selected_features"], start=1):
            selected_feature_rows.append({"asset_id": int(asset), "rank": int(rank), "feature_name": feature_name})

    candidate_frame = pd.DataFrame(all_candidate_rows)
    candidate_frame.drop(columns=["selected_features"]).to_csv(args.results_dir / "candidate_metrics_by_asset.csv", index=False)
    pd.DataFrame(selected_feature_rows).to_csv(args.results_dir / "selected_features_by_asset.csv", index=False)

    calibration_predictions = pd.concat(calibration_parts, ignore_index=True).sort_values(
        ["time_id", "asset_id"], kind="mergesort"
    )
    calibration_predictions.to_csv(args.results_dir / "calibration_predictions.csv", index=False)

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(["time_id", "asset_id"], kind="mergesort")
    predictions.to_csv(args.results_dir / "test_predictions.csv", index=False)

    y_test = predictions["target"].to_numpy(dtype=np.float64)
    pred_test = predictions["prediction"].to_numpy(dtype=np.float64)
    w_test = predictions["weight"].to_numpy(dtype=np.float64)
    asset_test = predictions["asset_id"].to_numpy(dtype=np.int64)
    test_score = weighted_zero_mean_r2(y_test, pred_test, w_test)
    test_score_by_asset = score_by_asset(y_test, pred_test, w_test, asset_test)

    metrics = {
        "leakage_safe": True,
        "future_function_guard": {
            "feature_screen_fit_train": f"time_id <= {args.fit_train_end_time}",
            "selection_and_shrink_calibration": f"{args.cal_start_time} <= time_id <= {args.cal_end_time}",
            "holdout_test_only_for_final_score": f"{args.test_start_time} <= time_id <= {args.test_end_time}",
        },
        "data_dir": str(args.data_dir),
        "model_scope": "per_asset",
        "fit_train_rows": int(fit_train_mask.sum()),
        "cal_rows": int(cal_mask.sum()),
        "final_train_rows": int(final_train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "candidate_selection_policy": {
            "candidate_score_mode": args.candidate_score_mode,
            "top_k_candidates": [int(value) for value in args.top_k_candidates],
            "lgbm_num_leaves_candidates": [int(value) for value in args.lgbm_num_leaves_candidates],
            "shrink_cap_candidates": [float(value) for value in args.shrink_cap_candidates],
        },
        "best_candidate_by_asset": best_candidates,
        "test_score": float(test_score),
        "test_score_by_asset": test_score_by_asset,
        "negative_asset_count": int(sum(score < 0 for score in test_score_by_asset.values())),
        "selected_feature_count_by_asset": {
            str(asset): int(len([row for row in selected_feature_rows if row["asset_id"] == asset])) for asset in assets
        },
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
