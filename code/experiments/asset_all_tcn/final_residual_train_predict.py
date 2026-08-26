from __future__ import annotations

import argparse
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
        description="正式训练 Ridge 基线 + LightGBM 残差模型，并预测 official test。"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset_all_residual_lgbm_final"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/asset_all_residual_lgbm_final"))
    parser.add_argument(
        "--fixed-features-file",
        type=Path,
        default=Path("results/asset_all_stable_features_100k/selected_features_stable_top128.csv"),
    )
    parser.add_argument("--sample-submission", type=Path, default=None)
    parser.add_argument("--train-lookback-time-points", type=int, default=90_000)
    parser.add_argument("--cal-time-points", type=int, default=20_000)
    parser.add_argument("--max-train-time-id", type=int, default=None)
    parser.add_argument("--test-start-time", type=int, default=None)
    parser.add_argument("--test-end-time", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)

    # 这些默认值来自当前 calibration 最优候选。
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--lgbm-num-leaves", type=int, default=31)
    parser.add_argument("--lgbm-estimators", type=int, default=200)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.005)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=8000)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=1000.0)
    parser.add_argument("--lgbm-subsample", type=float, default=0.7)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.7)
    parser.add_argument("--lgbm-seeds", type=int, nargs="+", default=[11, 42, 73])
    parser.add_argument("--lgbm-n-jobs", type=int, default=-1)

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


def fit_ridge(train_x: np.ndarray, y_train: np.ndarray, w_train: np.ndarray, alpha: float) -> Ridge:
    """Ridge 负责学习最稳定的线性信号，是整个残差框架的低方差底座。"""
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
    """LightGBM 只预测 Ridge 没解释掉的残差，并用多 seed 平均降低方差。"""
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
    """把多个 seed 的 LightGBM 残差预测做平均。"""
    predict_frame = pd.DataFrame(predict_x, columns=feature_names)
    predictions = [model.predict(predict_frame) for model in models]
    return np.mean(np.vstack(predictions), axis=0)


def search_residual_weight_and_shrink(
    y_cal: np.ndarray,
    w_cal: np.ndarray,
    asset_cal: np.ndarray,
    time_cal: np.ndarray,
    ridge_cal: np.ndarray,
    residual_cal: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """在 calibration 上同时选择残差权重和 shrink，选择规则默认看 min_halves 稳定性。"""
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


def fit_predict_residual_protocol(
    train_x_raw: np.ndarray,
    predict_x_raw: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, Ridge, list[LGBMRegressor]]:
    """训练 Ridge，再训练 LightGBM 预测 Ridge 残差，返回两部分预测。"""
    train_x, predict_x, _, _ = standardize(train_x_raw, predict_x_raw)
    ridge_model = fit_ridge(train_x, y_train, w_train, args.ridge_alpha)
    ridge_train = ridge_model.predict(train_x)
    ridge_predict = ridge_model.predict(predict_x)
    residual_train = y_train - ridge_train
    lgbm_models = fit_lgbm_residual_models(train_x, residual_train, w_train, feature_names, args)
    residual_predict = predict_lgbm_average(lgbm_models, predict_x, feature_names)
    return ridge_predict, residual_predict, ridge_model, lgbm_models


def save_zip(csv_path: Path, zip_path: Path) -> None:
    """把 submission.csv 压缩成官方常用的 zip 格式。"""
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=csv_path.name)


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
    selected_features = ranking.head(int(args.top_k))["feature_name"].astype(str).tolist()
    print(
        f"Residual final split: fit<= {fit_train_end_time}, cal={cal_start_time}..{train_end_time}, "
        f"test={args.test_start_time or test_min_time}..{args.test_end_time or test_max_time_available}"
    )

    train_frame = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + selected_features,
        min_time=train_start_time,
        max_time=train_end_time,
    )
    train_time = train_frame["time_id"].to_numpy(dtype=np.int64)
    fit_mask = train_time <= fit_train_end_time
    cal_mask = (train_time >= cal_start_time) & (train_time <= train_end_time)
    final_train_mask = train_time <= train_end_time

    fit_x_raw = train_frame.loc[fit_mask, selected_features].to_numpy(dtype=np.float32)
    cal_x_raw = train_frame.loc[cal_mask, selected_features].to_numpy(dtype=np.float32)
    y_fit = train_frame.loc[fit_mask, "target"].to_numpy(dtype=np.float32)
    w_fit = train_frame.loc[fit_mask, "weight"].to_numpy(dtype=np.float32)
    y_cal = train_frame.loc[cal_mask, "target"].to_numpy(dtype=np.float32)
    w_cal = train_frame.loc[cal_mask, "weight"].to_numpy(dtype=np.float32)
    asset_cal = train_frame.loc[cal_mask, "asset_id"].to_numpy(dtype=np.int64)
    time_cal = train_frame.loc[cal_mask, "time_id"].to_numpy(dtype=np.int64)

    ridge_cal, residual_cal, _, _ = fit_predict_residual_protocol(
        fit_x_raw,
        cal_x_raw,
        y_fit,
        w_fit,
        selected_features,
        args,
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

    test_min = args.test_start_time if args.test_start_time is not None else test_min_time
    test_max = args.test_end_time if args.test_end_time is not None else test_max_time_available
    test_frame = read_partitioned_frame(
        test_paths,
        BASE_COLUMNS_TEST + selected_features,
        min_time=test_min,
        max_time=test_max,
        max_rows=args.max_test_rows,
    )
    test_for_prediction = test_frame.copy()
    test_for_prediction["weight"] = 0.0
    test_for_prediction["target"] = 0.0
    combined_frame = pd.concat(
        [
            train_frame.loc[final_train_mask, BASE_COLUMNS_TRAIN + selected_features],
            test_for_prediction[BASE_COLUMNS_TRAIN + selected_features],
        ],
        ignore_index=True,
    ).sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    combined_time = combined_frame["time_id"].to_numpy(dtype=np.int64)
    combined_train_mask = combined_time <= train_end_time
    combined_test_mask = combined_time > train_end_time

    final_x_raw = combined_frame.loc[combined_train_mask, selected_features].to_numpy(dtype=np.float32)
    test_x_raw = combined_frame.loc[combined_test_mask, selected_features].to_numpy(dtype=np.float32)
    y_final = combined_frame.loc[combined_train_mask, "target"].to_numpy(dtype=np.float32)
    w_final = combined_frame.loc[combined_train_mask, "weight"].to_numpy(dtype=np.float32)
    ridge_test, residual_test, ridge_model, lgbm_models = fit_predict_residual_protocol(
        final_x_raw,
        test_x_raw,
        y_final,
        w_final,
        selected_features,
        args,
    )
    raw_test = ridge_test + float(best["residual_weight"]) * residual_test
    asset_test = combined_frame.loc[combined_test_mask, "asset_id"].to_numpy(dtype=np.int64)
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

    metrics = {
        "leakage_safe": True,
        "official_test_used_for_training": False,
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
            "train_lookback_time_points": int(args.train_lookback_time_points),
        },
        "rows": {
            "fit_train": int(fit_mask.sum()),
            "calibration": int(cal_mask.sum()),
            "final_train": int(final_train_mask.sum()),
            "test_predicted": int(len(test_predictions)),
        },
        "model": {
            "base_model": "Ridge",
            "residual_model": "LightGBM",
            "top_k": int(args.top_k),
            "ridge_alpha": float(args.ridge_alpha),
            "lgbm_num_leaves": int(args.lgbm_num_leaves),
            "lgbm_estimators": int(args.lgbm_estimators),
            "lgbm_learning_rate": float(args.lgbm_learning_rate),
            "lgbm_min_child_samples": int(args.lgbm_min_child_samples),
            "lgbm_reg_lambda": float(args.lgbm_reg_lambda),
            "lgbm_seeds": [int(seed) for seed in args.lgbm_seeds],
            "selected_features": selected_features,
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
        "test_prediction_stats": {
            "mean": float(np.mean(prediction_test)),
            "std": float(np.std(prediction_test)),
            "min": float(np.min(prediction_test)),
            "max": float(np.max(prediction_test)),
            "null_count": int(np.sum(~np.isfinite(prediction_test))),
            "finite_count": int(np.sum(np.isfinite(prediction_test))),
        },
        "output_files": {
            "calibration_predictions": str(args.results_dir / "calibration_predictions.csv"),
            "final_test_predictions": str(args.results_dir / "final_test_predictions.csv"),
            "submission": str(submission_path),
            "submission_zip": str(zip_path),
        },
    }
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    (args.model_dir / "metadata.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    pd.DataFrame({"feature_name": selected_features}).to_csv(args.results_dir / "selected_features.csv", index=False)
    if not args.no_save_models:
        save_pickle(
            args.model_dir / "residual_final_models.pkl",
            {
                "ridge_model": ridge_model,
                "lgbm_models": lgbm_models,
                "selected_features": selected_features,
                "calibration": metrics["calibration"],
            },
        )
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
