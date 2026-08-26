from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge

from walk_forward_tabular import (
    apply_shrink,
    calibrate_shrink_info,
    load_fixed_feature_ranking,
    score_by_asset,
    score_candidate_on_calibration,
    screen_features,
    shrink_values_for_assets,
    standardize,
    summarize_shrink_info,
    weighted_zero_mean_r2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe walk-forward Ridge + CatBoost GPU experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/asset_all_time100000"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset_all_walk_forward_catboost_gpu"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/asset_all_walk_forward_catboost_gpu"))
    parser.add_argument("--fit-train-end-time", type=int, default=59_999)
    parser.add_argument("--cal-start-time", type=int, default=60_000)
    parser.add_argument("--cal-end-time", type=int, default=79_999)
    parser.add_argument("--test-start-time", type=int, default=80_000)
    parser.add_argument("--test-end-time", type=int, default=99_999)
    parser.add_argument("--top-k-candidates", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--catboost-depth-candidates", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.03)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=200.0)
    parser.add_argument("--catboost-random-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--catboost-task-type", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--catboost-devices", type=str, default="0")
    parser.add_argument("--blend-step", type=float, default=0.01)
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[1.2])
    parser.add_argument(
        "--candidate-score-mode",
        choices=["full", "mean_halves", "min_halves"],
        default="full",
        help="候选参数在 calibration 段上的打分方式；min_halves 会更偏向前后半段都稳定的参数。",
    )
    parser.add_argument(
        "--shrink-mode",
        choices=["global", "per_asset"],
        default="per_asset",
        help="global 共用一个 shrink；per_asset 对每个标的单独校准预测幅度。",
    )
    parser.add_argument("--fixed-features-file", type=Path, default=None)
    parser.add_argument(
        "--no-asset-id-feature",
        action="store_true",
        help="默认把 asset_id 作为 CatBoost 类别特征；加这个参数则不使用 asset_id 类别特征。",
    )
    return parser.parse_args()


def make_catboost_frame(
    x_values: np.ndarray,
    asset_id: np.ndarray,
    feature_names: list[str],
    include_asset_id: bool,
) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.DataFrame(x_values, columns=feature_names)
    cat_features: list[str] = []
    if include_asset_id:
        # asset_id 是强横截面信息，让 CatBoost 自己学习不同标的的偏移和交互；Ridge 仍只吃连续 feature。
        frame["asset_id"] = asset_id.astype(str)
        cat_features.append("asset_id")
    return frame, cat_features


def fit_predict_pair(
    train_x: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    train_asset: np.ndarray,
    predict_x: np.ndarray,
    predict_asset: np.ndarray,
    feature_names: list[str],
    ridge_alpha: float,
    catboost_depth: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, Ridge, list[CatBoostRegressor]]:
    # Ridge 作为低方差线性底座；CatBoost GPU 作为非线性候选，二者后面只在 calibration 上融合。
    sample_weight = w_train / max(float(np.mean(w_train)), 1e-12)
    ridge = Ridge(alpha=ridge_alpha, solver="lsqr", max_iter=500)
    ridge.fit(train_x, y_train, sample_weight=sample_weight)
    ridge_pred = ridge.predict(predict_x)

    include_asset_id = not args.no_asset_id_feature
    train_frame, cat_features = make_catboost_frame(train_x, train_asset, feature_names, include_asset_id)
    predict_frame, _ = make_catboost_frame(predict_x, predict_asset, feature_names, include_asset_id)
    train_pool = Pool(train_frame, y_train, weight=sample_weight, cat_features=cat_features)
    predict_pool = Pool(predict_frame, cat_features=cat_features)

    catboost_models: list[CatBoostRegressor] = []
    catboost_predictions = []
    for seed in args.catboost_random_seeds:
        params = {
            "loss_function": "RMSE",
            "iterations": int(args.catboost_iterations),
            "learning_rate": float(args.catboost_learning_rate),
            "depth": int(catboost_depth),
            "l2_leaf_reg": float(args.catboost_l2_leaf_reg),
            "random_seed": int(seed),
            "task_type": args.catboost_task_type,
            "verbose": False,
            "allow_writing_files": False,
        }
        if args.catboost_task_type == "GPU":
            params["devices"] = args.catboost_devices
        model = CatBoostRegressor(**params)
        model.fit(train_pool)
        catboost_models.append(model)
        catboost_predictions.append(model.predict(predict_pool))

    catboost_pred = np.mean(np.vstack(catboost_predictions), axis=0)
    return ridge_pred, catboost_pred, ridge, catboost_models


def find_best_blend(
    y_true: np.ndarray,
    weight: np.ndarray,
    asset_id: np.ndarray,
    time_id: np.ndarray,
    ridge_pred: np.ndarray,
    catboost_pred: np.ndarray,
    step: float,
    shrink_mode: str,
    shrink_cap_candidates: list[float],
    score_mode: str,
) -> dict:
    # 这里搜索的是 Ridge/CatBoost 融合权重和 shrink；只允许看 calibration，不碰未来 holdout。
    best = {"score": -np.inf}
    for ridge_weight in np.arange(0.0, 1.0 + 1e-12, step):
        catboost_weight = 1.0 - ridge_weight
        base_prediction = ridge_weight * ridge_pred + catboost_weight * catboost_pred
        for shrink_cap in shrink_cap_candidates:
            shrink_info = calibrate_shrink_info(y_true, base_prediction, weight, asset_id, shrink_mode, shrink_cap)
            prediction = apply_shrink(base_prediction, asset_id, shrink_info)
            score_info = score_candidate_on_calibration(y_true, prediction, weight, time_id, score_mode)
            score = score_info["selection_score"]
            if score > best["score"]:
                best = {
                    "score": float(score),
                    "score_info": score_info,
                    "weights": {
                        "ridge": float(ridge_weight),
                        "catboost": float(catboost_weight),
                        "shrink": float(shrink_info["global"]),
                        "shrink_mode": shrink_mode,
                        "shrink_cap": float(shrink_cap),
                    },
                    "shrink_info": shrink_info,
                    "shrink_summary": summarize_shrink_info(shrink_info),
                }
    return best


def write_ridge_metadata(path: Path, ridge: Ridge, selected_features: list[str]) -> None:
    coef_frame = pd.DataFrame({"feature_name": selected_features, "coefficient": ridge.coef_.astype(float)})
    coef_frame.to_csv(path / "ridge_coefficients.csv", index=False)
    (path / "ridge_metadata.json").write_text(
        json.dumps({"intercept": float(ridge.intercept_)}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

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
    asset_fit = frame.loc[fit_train_mask, "asset_id"].to_numpy(dtype=np.int64)
    asset_cal = frame.loc[cal_mask, "asset_id"].to_numpy(dtype=np.int64)
    time_cal = frame.loc[cal_mask, "time_id"].to_numpy(dtype=np.int64)

    candidate_rows = []
    best_candidate = None
    for top_k in args.top_k_candidates:
        selected_features = ranking.head(top_k)["feature_name"].astype(str).tolist()
        fit_x_raw = frame.loc[fit_train_mask, selected_features].to_numpy(dtype=np.float32)
        cal_x_raw = frame.loc[cal_mask, selected_features].to_numpy(dtype=np.float32)
        fit_x, cal_x, _, _ = standardize(fit_x_raw, cal_x_raw)

        for depth in args.catboost_depth_candidates:
            ridge_pred, catboost_pred, _, _ = fit_predict_pair(
                fit_x,
                y_fit,
                w_fit,
                asset_fit,
                cal_x,
                asset_cal,
                selected_features,
                args.ridge_alpha,
                int(depth),
                args,
            )
            blend = find_best_blend(
                y_cal,
                w_cal,
                asset_cal,
                time_cal,
                ridge_pred,
                catboost_pred,
                args.blend_step,
                args.shrink_mode,
                args.shrink_cap_candidates,
                args.candidate_score_mode,
            )
            score_info = blend["score_info"]
            shrink_summary = blend["shrink_summary"]
            row = {
                "top_k": int(top_k),
                "catboost_depth": int(depth),
                "cal_score": float(blend["score"]),
                "cal_score_mode": args.candidate_score_mode,
                "cal_full_score": float(score_info["full_score"]),
                "cal_first_half_score": float(score_info["first_half_score"]),
                "cal_second_half_score": float(score_info["second_half_score"]),
                "cal_ridge_weight": float(blend["weights"]["ridge"]),
                "cal_catboost_weight": float(blend["weights"]["catboost"]),
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
            if best_candidate is None or row["cal_score"] > best_candidate["cal_score"]:
                best_candidate = row

    if best_candidate is None:
        raise ValueError("no candidate was evaluated")
    pd.DataFrame(candidate_rows).sort_values("cal_score", ascending=False).to_csv(
        args.results_dir / "candidate_metrics.csv", index=False
    )

    selected_features = ranking.head(int(best_candidate["top_k"]))["feature_name"].astype(str).tolist()
    weights = {
        "ridge": float(best_candidate["cal_ridge_weight"]),
        "catboost": float(best_candidate["cal_catboost_weight"]),
        "shrink": float(best_candidate["cal_shrink"]),
        "shrink_mode": str(best_candidate["cal_shrink_mode"]),
        "shrink_cap": float(best_candidate["cal_shrink_cap"]),
    }
    shrink_info = best_candidate["cal_shrink_info"]

    # 保存 calibration 段预测，后续可以和 LightGBM 全局底座做二层融合；融合权重仍只能在 calibration 上学习。
    cal_fit_x_raw = frame.loc[fit_train_mask, selected_features].to_numpy(dtype=np.float32)
    cal_eval_x_raw = frame.loc[cal_mask, selected_features].to_numpy(dtype=np.float32)
    cal_fit_x, cal_eval_x, _, _ = standardize(cal_fit_x_raw, cal_eval_x_raw)
    ridge_cal, catboost_cal, _, _ = fit_predict_pair(
        cal_fit_x,
        y_fit,
        w_fit,
        asset_fit,
        cal_eval_x,
        asset_cal,
        selected_features,
        args.ridge_alpha,
        int(best_candidate["catboost_depth"]),
        args,
    )
    cal_base_prediction = weights["ridge"] * ridge_cal + weights["catboost"] * catboost_cal
    cal_shrink_values = shrink_values_for_assets(asset_cal, shrink_info)
    cal_prediction = cal_shrink_values * cal_base_prediction
    calibration_predictions = frame.loc[cal_mask, base_columns].copy()
    calibration_predictions["ridge_prediction"] = ridge_cal
    calibration_predictions["catboost_prediction"] = catboost_cal
    calibration_predictions["base_prediction"] = cal_base_prediction
    calibration_predictions["shrink_value"] = cal_shrink_values
    calibration_predictions["prediction"] = cal_prediction
    calibration_predictions["error"] = calibration_predictions["prediction"] - calibration_predictions["target"]
    calibration_predictions.to_csv(args.results_dir / "calibration_predictions.csv", index=False)

    # 最终模型只用 calibration 结束前的历史重新训练，然后评估未来 holdout。
    final_train_x_raw = frame.loc[final_train_mask, selected_features].to_numpy(dtype=np.float32)
    test_x_raw = frame.loc[test_mask, selected_features].to_numpy(dtype=np.float32)
    final_train_x, test_x, mean, scale = standardize(final_train_x_raw, test_x_raw)
    y_final_train = frame.loc[final_train_mask, "target"].to_numpy(dtype=np.float32)
    w_final_train = frame.loc[final_train_mask, "weight"].to_numpy(dtype=np.float32)
    asset_final_train = frame.loc[final_train_mask, "asset_id"].to_numpy(dtype=np.int64)
    y_test = frame.loc[test_mask, "target"].to_numpy(dtype=np.float32)
    w_test = frame.loc[test_mask, "weight"].to_numpy(dtype=np.float32)
    asset_test = frame.loc[test_mask, "asset_id"].to_numpy(dtype=np.int64)

    ridge_test, catboost_test, ridge_model, catboost_models = fit_predict_pair(
        final_train_x,
        y_final_train,
        w_final_train,
        asset_final_train,
        test_x,
        asset_test,
        selected_features,
        args.ridge_alpha,
        int(best_candidate["catboost_depth"]),
        args,
    )
    base_prediction = weights["ridge"] * ridge_test + weights["catboost"] * catboost_test
    test_shrink_values = shrink_values_for_assets(asset_test, shrink_info)
    prediction = test_shrink_values * base_prediction
    test_score = weighted_zero_mean_r2(y_test, prediction, w_test)

    predictions = frame.loc[test_mask, base_columns].copy()
    predictions["ridge_prediction"] = ridge_test
    predictions["catboost_prediction"] = catboost_test
    predictions["base_prediction"] = base_prediction
    predictions["shrink_value"] = test_shrink_values
    predictions["prediction"] = prediction
    predictions["error"] = predictions["prediction"] - predictions["target"]
    predictions.to_csv(args.results_dir / "test_predictions.csv", index=False)
    pd.DataFrame({"feature_name": selected_features}).to_csv(args.results_dir / "selected_features.csv", index=False)

    write_ridge_metadata(args.model_dir, ridge_model, selected_features)
    for index, model in enumerate(catboost_models):
        seed = int(args.catboost_random_seeds[index])
        model.save_model(args.model_dir / f"catboost_seed{seed}.cbm")

    test_score_by_asset = score_by_asset(y_test, prediction, w_test, asset_test)
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
        "catboost_config": {
            "task_type": args.catboost_task_type,
            "devices": args.catboost_devices,
            "iterations": int(args.catboost_iterations),
            "learning_rate": float(args.catboost_learning_rate),
            "l2_leaf_reg": float(args.catboost_l2_leaf_reg),
            "random_seeds": [int(seed) for seed in args.catboost_random_seeds],
            "include_asset_id_feature": not args.no_asset_id_feature,
        },
        "best_candidate": best_candidate,
        "candidate_selection_policy": {
            "candidate_score_mode": args.candidate_score_mode,
            "shrink_cap_candidates": [float(value) for value in args.shrink_cap_candidates],
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
    (args.model_dir / "metadata.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
