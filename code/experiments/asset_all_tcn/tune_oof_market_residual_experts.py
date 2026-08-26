from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from final_train_predict import (
    BASE_COLUMNS_TRAIN,
    load_feature_ranking,
    parquet_paths,
    read_partitioned_frame,
    schema_columns,
    time_range,
)
from market_mean_ts_model import (
    build_time_feature_frame,
    map_time_predictions_to_rows,
    normalized_weight,
    prepare_matrix,
    score_time_blocks,
    weighted_market_target,
)
from walk_forward_tabular import (
    apply_shrink,
    calibrate_shrink_info,
    score_candidate_on_calibration,
    weighted_zero_mean_r2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用时间 OOF 残差训练市场方向/幅度模型与极端专家。"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=Path(
            "results/blend_best_panel_market32_75k_cal20k/"
            "best_blend_calibration_predictions.csv"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/oof_market_residual_experts_75k_probe"),
    )
    parser.add_argument(
        "--fixed-features-file",
        type=Path,
        default=Path(
            "results/asset_all_stable_features_100k/selected_features_stable_top128.csv"
        ),
    )
    parser.add_argument("--train-lookback-time-points", type=int, default=75_000)
    parser.add_argument("--cal-time-points", type=int, default=20_000)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-train-time-id", type=int, default=None)
    parser.add_argument("--oof-initial-time-points", type=int, default=25_000)
    parser.add_argument("--oof-block-time-points", type=int, default=10_000)
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.01)
    parser.add_argument("--lgbm-n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta-min", type=float, default=-0.50)
    parser.add_argument("--beta-max", type=float, default=1.00)
    parser.add_argument("--beta-step", type=float, default=0.05)
    parser.add_argument("--shrink-cap", type=float, default=1.6)
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


def model_params(args: argparse.Namespace, objective: str) -> dict:
    params = {
        "objective": objective,
        "n_estimators": int(args.lgbm_estimators),
        "learning_rate": float(args.lgbm_learning_rate),
        "num_leaves": 7,
        "min_child_samples": 300,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 300.0,
        "random_state": int(args.seed),
        "n_jobs": int(args.lgbm_n_jobs),
        "verbose": -1,
    }
    if objective == "huber":
        params["alpha"] = 0.85
    return params


def fit_regressor(
    x: pd.DataFrame,
    y: np.ndarray,
    weight: np.ndarray,
    args: argparse.Namespace,
    objective: str = "regression",
) -> LGBMRegressor:
    model = LGBMRegressor(**model_params(args, objective))
    model.fit(x, y, sample_weight=normalized_weight(weight))
    return model


def fit_classifier(
    x: pd.DataFrame,
    y: np.ndarray,
    weight: np.ndarray,
    args: argparse.Namespace,
) -> LGBMClassifier:
    model = LGBMClassifier(**model_params(args, "binary"))
    model.fit(x, y.astype(np.int8), sample_weight=normalized_weight(weight))
    return model


def soft_direction(probability: np.ndarray, strength: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-5, 1.0 - 1e-5)
    logit = np.log(clipped / (1.0 - clipped))
    return np.tanh(float(strength) * logit)


def generate_oof_direct_prediction(
    fit_x: pd.DataFrame,
    y_fit: np.ndarray,
    w_fit: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    """扩张窗口 OOF：每个残差标签都来自只看过更早时间的模型。"""
    row_count = len(fit_x)
    initial = int(args.oof_initial_time_points)
    block_size = int(args.oof_block_time_points)
    if initial >= row_count:
        raise ValueError("oof initial window 必须小于 fit 时间点数量")
    prediction = np.full(row_count, np.nan, dtype=np.float64)
    fold_rows = []
    block_start = initial
    fold = 1
    while block_start < row_count:
        block_end = min(row_count, block_start + block_size)
        model = fit_regressor(
            fit_x.iloc[:block_start],
            y_fit[:block_start],
            w_fit[:block_start],
            args,
            objective="regression",
        )
        prediction[block_start:block_end] = model.predict(
            fit_x.iloc[block_start:block_end]
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(block_start),
                "valid_start_index": int(block_start),
                "valid_end_index": int(block_end - 1),
                "valid_rows": int(block_end - block_start),
            }
        )
        block_start = block_end
        fold += 1
    return prediction, fold_rows


def build_residual_candidates(
    residual_x: pd.DataFrame,
    residual_y: np.ndarray,
    residual_weight: np.ndarray,
    cal_x: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    candidates: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}

    residual_model = fit_regressor(
        residual_x, residual_y, residual_weight, args, objective="huber"
    )
    residual_direct = residual_model.predict(cal_x).astype(np.float64)
    candidates["residual_huber"] = residual_direct

    sign_target = (residual_y > 0.0).astype(np.int8)
    sign_model = fit_classifier(residual_x, sign_target, residual_weight, args)
    sign_probability_fit = sign_model.predict_proba(residual_x)[:, 1]
    sign_probability_cal = sign_model.predict_proba(cal_x)[:, 1]

    for objective in ["regression_l1", "huber"]:
        magnitude_model = fit_regressor(
            residual_x,
            np.abs(residual_y),
            residual_weight,
            args,
            objective=objective,
        )
        magnitude_cal = np.clip(magnitude_model.predict(cal_x), 0.0, None)
        for strength in [0.25, 0.50, 1.0, 2.0]:
            candidates[f"residual_signmag_{objective}_s{strength:g}"] = (
                soft_direction(sign_probability_cal, strength) * magnitude_cal
            )

    for quantile in [0.70, 0.80, 0.90]:
        threshold = float(np.quantile(np.abs(residual_y), quantile))
        extreme_label = (np.abs(residual_y) >= threshold).astype(np.int8)
        gate_model = fit_classifier(residual_x, extreme_label, residual_weight, args)
        gate_probability = gate_model.predict_proba(cal_x)[:, 1]
        extreme_mask = extreme_label.astype(bool)
        expert_model = fit_regressor(
            residual_x.loc[extreme_mask].reset_index(drop=True),
            residual_y[extreme_mask],
            residual_weight[extreme_mask],
            args,
            objective="huber",
        )
        expert_prediction = expert_model.predict(cal_x).astype(np.float64)
        for strength in [0.25, 0.50, 1.0]:
            gate = np.clip(float(strength) * gate_probability, 0.0, 1.0)
            candidates[f"residual_expert_q{quantile:.2f}_g{strength:g}"] = (
                (1.0 - gate) * residual_direct + gate * expert_prediction
            )
        diagnostics[f"residual_expert_q{quantile:.2f}"] = {
            "threshold": threshold,
            "fit_positive_rate": float(np.average(extreme_label, weights=residual_weight)),
            "cal_gate_mean": float(np.mean(gate_probability)),
        }

    diagnostics["residual_direction"] = {
        "fit_weighted_accuracy": float(
            np.average((sign_probability_fit >= 0.5) == sign_target, weights=residual_weight)
        ),
        "fit_probability_mean": float(
            np.average(sign_probability_fit, weights=residual_weight)
        ),
        "cal_probability_mean": float(np.mean(sign_probability_cal)),
    }
    return candidates, diagnostics


def align_base_predictions(raw_cal: pd.DataFrame, path: Path) -> np.ndarray:
    base = pd.read_csv(path, usecols=["time_id", "asset_id", "target", "weight", "prediction"])
    base = base.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    if not np.array_equal(
        base[["time_id", "asset_id"]].to_numpy(),
        raw_cal[["time_id", "asset_id"]].to_numpy(),
    ):
        raise ValueError("base predictions 与 calibration 键不一致")
    return base["prediction"].to_numpy(dtype=np.float64)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train_paths = parquet_paths(args.raw_data_dir, "train")
    train_min, train_max = time_range(train_paths)
    train_end = (
        min(train_max, int(args.max_train_time_id))
        if args.max_train_time_id is not None
        else train_max
    )
    train_start = max(train_min, train_end - int(args.train_lookback_time_points) + 1)
    fit_end = train_end - int(args.cal_time_points)
    cal_start = fit_end + 1

    available = schema_columns(train_paths)
    ranking = load_feature_ranking(args.fixed_features_file, available)
    features = ranking.head(int(args.top_k))["feature_name"].astype(str).tolist()
    print(f"OOF residual split: fit={train_start}..{fit_end}, cal={cal_start}..{train_end}")

    raw_train = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + features,
        min_time=train_start,
        max_time=train_end,
    )
    market_targets = weighted_market_target(raw_train)
    time_features, feature_columns = build_time_feature_frame(
        raw_train, features, [], [], [], []
    )
    time_frame = time_features.merge(market_targets, on="time_id", how="left")
    time_values = time_frame["time_id"].to_numpy(dtype=np.int64)
    fit_mask = (time_values >= train_start) & (time_values <= fit_end)
    cal_mask = (time_values >= cal_start) & (time_values <= train_end)
    fit_time = time_frame.loc[fit_mask].copy()
    cal_time = time_frame.loc[cal_mask].copy()
    fit_x, cal_x, _, _ = prepare_matrix(fit_time, cal_time, feature_columns)
    fit_x_frame = pd.DataFrame(fit_x, columns=feature_columns)
    cal_x_frame = pd.DataFrame(cal_x, columns=feature_columns)
    y_fit = fit_time["market_target"].to_numpy(dtype=np.float64)
    w_fit = fit_time["weight_sum"].to_numpy(dtype=np.float64)

    oof_direct, fold_rows = generate_oof_direct_prediction(
        fit_x_frame, y_fit, w_fit, args
    )
    oof_mask = np.isfinite(oof_direct)
    residual_y = y_fit[oof_mask] - oof_direct[oof_mask]
    residual_x = fit_x_frame.loc[oof_mask].reset_index(drop=True)
    residual_weight = w_fit[oof_mask]

    direct_full = fit_regressor(
        fit_x_frame, y_fit, w_fit, args, objective="regression"
    )
    direct_cal = direct_full.predict(cal_x_frame).astype(np.float64)
    residual_candidates, diagnostics = build_residual_candidates(
        residual_x,
        residual_y,
        residual_weight,
        cal_x_frame,
        args,
    )

    raw_cal = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN,
        min_time=cal_start,
        max_time=train_end,
    )
    y_cal = raw_cal["target"].to_numpy(dtype=np.float64)
    w_cal = raw_cal["weight"].to_numpy(dtype=np.float64)
    asset_cal = raw_cal["asset_id"].to_numpy(dtype=np.int64)
    row_time = raw_cal["time_id"].to_numpy(dtype=np.int64)
    cal_time_id = cal_time["time_id"].to_numpy(dtype=np.int64)
    base_prediction = align_base_predictions(raw_cal, args.base_predictions)

    rows = []
    payloads: dict[str, dict] = {}
    beta_values = np.arange(args.beta_min, args.beta_max + 1e-12, args.beta_step)
    for name, residual_prediction in residual_candidates.items():
        residual_row = map_time_predictions_to_rows(
            row_time, cal_time_id, residual_prediction
        )
        for beta in beta_values:
            raw_prediction = base_prediction + float(beta) * residual_row
            shrink_info = calibrate_shrink_info(
                y_cal,
                raw_prediction,
                w_cal,
                asset_cal,
                "per_asset",
                float(args.shrink_cap),
            )
            prediction = apply_shrink(raw_prediction, asset_cal, shrink_info)
            score_info = score_candidate_on_calibration(
                y_cal, prediction, w_cal, row_time, "full"
            )
            selection_score = min(
                float(score_info["first_half_score"]),
                float(score_info["second_half_score"]),
            )
            rows.append(
                {
                    "candidate": name,
                    "beta": float(beta),
                    "selection_score": selection_score,
                    "full_score": float(score_info["full_score"]),
                    "first_half_score": float(score_info["first_half_score"]),
                    "second_half_score": float(score_info["second_half_score"]),
                    "prediction_std": float(np.std(prediction)),
                }
            )
            payloads[f"{name}|{beta:.8f}"] = {
                "residual_row": residual_row,
                "raw_prediction": raw_prediction,
                "prediction": prediction,
                "shrink_info": shrink_info,
            }

    candidate_metrics = pd.DataFrame(rows).sort_values(
        ["selection_score", "full_score"], ascending=False
    ).reset_index(drop=True)
    best_row = candidate_metrics.iloc[0]
    best = payloads[f"{best_row['candidate']}|{float(best_row['beta']):.8f}"]

    output = raw_cal.copy()
    output["base_prediction"] = base_prediction
    output["direct_market_prediction"] = map_time_predictions_to_rows(
        row_time, cal_time_id, direct_cal
    )
    output["residual_prediction"] = best["residual_row"]
    output["raw_prediction"] = best["raw_prediction"]
    output["prediction"] = best["prediction"]
    output["error"] = output["prediction"] - output["target"]
    candidate_metrics.to_csv(args.results_dir / "candidate_metrics.csv", index=False)
    candidate_metrics.head(100).to_csv(args.results_dir / "candidate_top100.csv", index=False)
    output.to_csv(args.results_dir / "calibration_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.results_dir / "oof_folds.csv", index=False)

    base_score = float(weighted_zero_mean_r2(y_cal, base_prediction, w_cal))
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_safe_residual_labels": True,
        "official_test_used": False,
        "split": {
            "fit": f"{train_start} <= time_id <= {fit_end}",
            "calibration": f"{cal_start} <= time_id <= {train_end}",
        },
        "oof": {
            "residual_rows": int(oof_mask.sum()),
            "folds": fold_rows,
            "direct_oof_market_r2": float(
                weighted_zero_mean_r2(
                    y_fit[oof_mask], oof_direct[oof_mask], w_fit[oof_mask]
                )
            ),
            "residual_std": float(np.std(residual_y)),
        },
        "base_score": base_score,
        "best": {
            "candidate": str(best_row["candidate"]),
            "beta": float(best_row["beta"]),
            "selection_score": float(best_row["selection_score"]),
            "full_score": float(best_row["full_score"]),
            "first_half_score": float(best_row["first_half_score"]),
            "second_half_score": float(best_row["second_half_score"]),
            "improvement_over_base": float(best_row["full_score"] - base_score),
            **score_time_blocks(y_cal, best["prediction"], w_cal, row_time, 4),
            **score_time_blocks(y_cal, best["prediction"], w_cal, row_time, 8),
        },
        "model_diagnostics": diagnostics,
        "output_files": {
            "candidate_metrics": str(args.results_dir / "candidate_metrics.csv"),
            "calibration_predictions": str(args.results_dir / "calibration_predictions.csv"),
            "oof_folds": str(args.results_dir / "oof_folds.csv"),
        },
    }
    with (args.results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, default=json_default)
    print(json.dumps(metrics["best"], ensure_ascii=False, indent=2, default=json_default))
    print(f"Saved outputs to {args.results_dir}")


if __name__ == "__main__":
    main()
