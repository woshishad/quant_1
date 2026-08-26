from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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
        description="市场方向/幅度两阶段、极端行情专家和条件 shrink 校准实验。"
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
        default=Path("results/asset_all_market_regime_experts_75k_probe"),
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
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.01)
    parser.add_argument("--lgbm-n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta-min", type=float, default=-0.50)
    parser.add_argument("--beta-max", type=float, default=0.80)
    parser.add_argument("--beta-step", type=float, default=0.05)
    parser.add_argument("--shrink-cap", type=float, default=1.6)
    parser.add_argument(
        "--conditional-quantiles",
        type=float,
        nargs="+",
        default=[0.0, 0.50, 0.80, 0.95, 1.0],
    )
    parser.add_argument("--conditional-scale-cap", type=float, default=2.5)
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


def lgbm_common(args: argparse.Namespace) -> dict:
    """市场级样本只有几万行，因此使用小叶子和强正则，减少记忆噪声。"""
    return {
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


def fit_regressor(
    fit_x: pd.DataFrame,
    target: np.ndarray,
    weight: np.ndarray,
    args: argparse.Namespace,
    *,
    objective: str = "regression",
) -> LGBMRegressor:
    params = lgbm_common(args)
    params["objective"] = objective
    if objective == "huber":
        params["alpha"] = 0.85
    model = LGBMRegressor(**params)
    model.fit(fit_x, target, sample_weight=normalized_weight(weight))
    return model


def fit_classifier(
    fit_x: pd.DataFrame,
    target: np.ndarray,
    weight: np.ndarray,
    args: argparse.Namespace,
) -> LGBMClassifier:
    params = lgbm_common(args)
    params["objective"] = "binary"
    model = LGBMClassifier(**params)
    model.fit(fit_x, target.astype(np.int8), sample_weight=normalized_weight(weight))
    return model


def confidence_signed_probability(probability: np.ndarray, strength: float) -> np.ndarray:
    """
    把上涨概率转换为 [-1, 1] 的软方向。
    tanh(logit) 比直接使用硬分类更稳，低置信度样本会自然收缩到 0。
    """
    clipped = np.clip(probability, 1e-5, 1.0 - 1e-5)
    logit = np.log(clipped / (1.0 - clipped))
    return np.tanh(float(strength) * logit)


def build_market_candidates(
    fit_x: np.ndarray,
    cal_x: np.ndarray,
    feature_columns: list[str],
    y_fit: np.ndarray,
    w_fit: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    fit_frame = pd.DataFrame(fit_x, columns=feature_columns)
    cal_frame = pd.DataFrame(cal_x, columns=feature_columns)
    diagnostics: dict[str, object] = {}

    # 直接回归是参照组，用于判断两阶段模型是否真正产生了新增信息。
    direct_model = fit_regressor(fit_frame, y_fit, w_fit, args)
    direct_fit = direct_model.predict(fit_frame).astype(np.float64)
    direct_cal = direct_model.predict(cal_frame).astype(np.float64)
    candidates: dict[str, np.ndarray] = {"direct_market": direct_cal}

    # 第一阶段预测方向，第二阶段预测绝对幅度。
    sign_target = (y_fit > 0.0).astype(np.int8)
    sign_model = fit_classifier(fit_frame, sign_target, w_fit, args)
    sign_probability_fit = sign_model.predict_proba(fit_frame)[:, 1]
    sign_probability_cal = sign_model.predict_proba(cal_frame)[:, 1]

    magnitude_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for objective in ["regression_l1", "huber"]:
        magnitude_model = fit_regressor(
            fit_frame,
            np.abs(y_fit),
            w_fit,
            args,
            objective=objective,
        )
        magnitude_predictions[objective] = (
            np.clip(magnitude_model.predict(fit_frame), 0.0, None),
            np.clip(magnitude_model.predict(cal_frame), 0.0, None),
        )

    for objective, (_, magnitude_cal) in magnitude_predictions.items():
        for strength in [0.25, 0.50, 1.0, 2.0]:
            soft_sign = confidence_signed_probability(sign_probability_cal, strength)
            candidates[f"signmag_{objective}_s{strength:g}"] = soft_sign * magnitude_cal

    # 极端专家：只用训练段的大幅市场变动训练，并由极端概率做连续门控。
    for quantile in [0.70, 0.80, 0.90]:
        threshold = float(np.quantile(np.abs(y_fit), quantile))
        extreme_label = (np.abs(y_fit) >= threshold).astype(np.int8)
        extreme_classifier = fit_classifier(fit_frame, extreme_label, w_fit, args)
        extreme_probability_cal = extreme_classifier.predict_proba(cal_frame)[:, 1]
        extreme_probability_fit = extreme_classifier.predict_proba(fit_frame)[:, 1]

        extreme_mask = extreme_label.astype(bool)
        expert_model = fit_regressor(
            fit_frame.loc[extreme_mask].reset_index(drop=True),
            y_fit[extreme_mask],
            w_fit[extreme_mask],
            args,
            objective="huber",
        )
        expert_cal = expert_model.predict(cal_frame).astype(np.float64)
        for gate_strength in [0.25, 0.50, 1.0]:
            gate = np.clip(gate_strength * extreme_probability_cal, 0.0, 1.0)
            candidates[f"expert_q{quantile:.2f}_g{gate_strength:g}"] = (
                (1.0 - gate) * direct_cal + gate * expert_cal
            )

        diagnostics[f"expert_q{quantile:.2f}"] = {
            "target_threshold": threshold,
            "fit_positive_rate": float(np.average(extreme_label, weights=w_fit)),
            "fit_probability_mean": float(np.average(extreme_probability_fit, weights=w_fit)),
            "cal_probability_mean": float(np.mean(extreme_probability_cal)),
        }

    sign_prediction_fit = sign_probability_fit >= 0.5
    diagnostics["direction"] = {
        "fit_weighted_accuracy": float(
            np.average(sign_prediction_fit == sign_target, weights=w_fit)
        ),
        "fit_probability_mean": float(np.average(sign_probability_fit, weights=w_fit)),
        "cal_probability_mean": float(np.mean(sign_probability_cal)),
        "direct_fit_std": float(np.std(direct_fit)),
    }
    return candidates, diagnostics


def make_edges(regime: np.ndarray, quantiles: list[float]) -> np.ndarray:
    values = np.abs(np.asarray(regime, dtype=np.float64))
    edges = np.quantile(values, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    # 低波动预测可能产生重复分位点；nextafter 保证 bucket 边界严格递增。
    for index in range(1, len(edges) - 1):
        if edges[index] <= edges[index - 1]:
            edges[index] = np.nextafter(edges[index - 1], np.inf)
    return edges


def bucket_ids(regime: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(np.abs(regime), edges[1:-1], right=False).astype(np.int16)


def fit_conditional_scales(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    regime: np.ndarray,
    quantiles: list[float],
    scale_cap: float,
) -> dict:
    """每个市场预测幅度区间只拟合一个缩放系数，参数量保持很小。"""
    edges = make_edges(regime, quantiles)
    buckets = bucket_ids(regime, edges)
    scales = []
    counts = []
    for bucket in range(len(edges) - 1):
        mask = buckets == bucket
        denominator = float(np.sum(weight[mask] * prediction[mask] ** 2))
        if denominator <= 1e-18:
            scale = 1.0
        else:
            scale = float(
                np.sum(weight[mask] * y_true[mask] * prediction[mask]) / denominator
            )
        scales.append(float(np.clip(scale, 0.0, scale_cap)))
        counts.append(int(mask.sum()))
    return {"edges": edges, "scales": np.asarray(scales), "counts": counts}


def apply_conditional_scales(
    prediction: np.ndarray,
    regime: np.ndarray,
    info: dict,
) -> tuple[np.ndarray, np.ndarray]:
    buckets = bucket_ids(regime, np.asarray(info["edges"], dtype=np.float64))
    scales = np.asarray(info["scales"], dtype=np.float64)
    return prediction * scales[buckets], buckets


def forward_conditional_score(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    regime: np.ndarray,
    time_id: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """只用 calibration 前半段拟合条件 shrink，在后半段检验。"""
    unique_times = np.unique(time_id)
    split_time = int(unique_times[len(unique_times) // 2])
    fit_mask = time_id < split_time
    test_mask = ~fit_mask
    info = fit_conditional_scales(
        y_true[fit_mask],
        prediction[fit_mask],
        weight[fit_mask],
        regime[fit_mask],
        args.conditional_quantiles,
        args.conditional_scale_cap,
    )
    adjusted, _ = apply_conditional_scales(prediction[test_mask], regime[test_mask], info)
    return {
        "split_time": split_time,
        "second_half_score": float(
            weighted_zero_mean_r2(y_true[test_mask], adjusted, weight[test_mask])
        ),
        "second_half_plain_score": float(
            weighted_zero_mean_r2(y_true[test_mask], prediction[test_mask], weight[test_mask])
        ),
        "scales": np.asarray(info["scales"]).tolist(),
        "edges": np.asarray(info["edges"]).tolist(),
    }


def align_base_predictions(raw_cal: pd.DataFrame, path: Path) -> np.ndarray:
    base = pd.read_csv(path, usecols=["time_id", "asset_id", "target", "weight", "prediction"])
    base = base.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    keys_match = np.array_equal(
        base[["time_id", "asset_id"]].to_numpy(),
        raw_cal[["time_id", "asset_id"]].to_numpy(),
    )
    if not keys_match:
        raise ValueError("base predictions 与当前 calibration 的 time_id/asset_id 不一致")
    if float(np.max(np.abs(base["target"].to_numpy() - raw_cal["target"].to_numpy()))) > 1e-6:
        raise ValueError("base predictions 与当前 calibration 的 target 不一致")
    return base["prediction"].to_numpy(dtype=np.float64)


def target_error_buckets(
    frame: pd.DataFrame,
    base_prediction: np.ndarray,
    improved_prediction: np.ndarray,
) -> pd.DataFrame:
    absolute_target = np.abs(frame["target"].to_numpy(dtype=np.float64))
    quantile_edges = np.unique(np.quantile(absolute_target, np.linspace(0.0, 1.0, 11)))
    bucket = np.digitize(absolute_target, quantile_edges[1:-1], right=False)
    rows = []
    y_true = frame["target"].to_numpy(dtype=np.float64)
    weight = frame["weight"].to_numpy(dtype=np.float64)
    for index in range(len(quantile_edges) - 1):
        mask = bucket == index
        rows.append(
            {
                "bucket": index,
                "abs_target_min": float(quantile_edges[index]),
                "abs_target_max": float(quantile_edges[index + 1]),
                "rows": int(mask.sum()),
                "base_r2": float(weighted_zero_mean_r2(y_true[mask], base_prediction[mask], weight[mask])),
                "improved_r2": float(
                    weighted_zero_mean_r2(y_true[mask], improved_prediction[mask], weight[mask])
                ),
                "base_mae": float(np.average(np.abs(y_true[mask] - base_prediction[mask]), weights=weight[mask])),
                "improved_mae": float(
                    np.average(np.abs(y_true[mask] - improved_prediction[mask]), weights=weight[mask])
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_bucket_scores(frame: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(frame))
    width = 0.38
    plt.figure(figsize=(10, 4.5))
    plt.bar(x - width / 2, frame["base_r2"], width=width, label="base")
    plt.bar(x + width / 2, frame["improved_r2"], width=width, label="improved")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, [f"Q{value + 1}" for value in frame["bucket"]])
    plt.xlabel("|target| decile")
    plt.ylabel("weighted zero-mean R2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train_paths = parquet_paths(args.raw_data_dir, "train")
    train_min_time, train_max_time = time_range(train_paths)
    train_end = (
        min(train_max_time, int(args.max_train_time_id))
        if args.max_train_time_id is not None
        else train_max_time
    )
    train_start = max(train_min_time, train_end - int(args.train_lookback_time_points) + 1)
    fit_end = train_end - int(args.cal_time_points)
    cal_start = fit_end + 1

    available_columns = schema_columns(train_paths)
    ranking = load_feature_ranking(args.fixed_features_file, available_columns)
    features = ranking.head(int(args.top_k))["feature_name"].astype(str).tolist()
    print(f"Regime split: fit={train_start}..{fit_end}, cal={cal_start}..{train_end}")

    raw_train = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + features,
        min_time=train_start,
        max_time=train_end,
    )
    market_targets = weighted_market_target(raw_train)
    time_features, feature_columns = build_time_feature_frame(
        raw_train,
        features,
        [],
        [],
        [],
        [],
    )
    time_frame = time_features.merge(market_targets, on="time_id", how="left")
    time_values = time_frame["time_id"].to_numpy(dtype=np.int64)
    fit_mask = (time_values >= train_start) & (time_values <= fit_end)
    cal_mask = (time_values >= cal_start) & (time_values <= train_end)
    fit_time = time_frame.loc[fit_mask].copy()
    cal_time = time_frame.loc[cal_mask].copy()
    fit_x, cal_x, _, _ = prepare_matrix(fit_time, cal_time, feature_columns)

    market_candidates, model_diagnostics = build_market_candidates(
        fit_x,
        cal_x,
        feature_columns,
        fit_time["market_target"].to_numpy(dtype=np.float32),
        fit_time["weight_sum"].to_numpy(dtype=np.float32),
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
    row_time_cal = raw_cal["time_id"].to_numpy(dtype=np.int64)
    cal_time_id = cal_time["time_id"].to_numpy(dtype=np.int64)
    base_prediction = align_base_predictions(raw_cal, args.base_predictions)

    candidate_rows: list[dict] = []
    candidate_payload: dict[str, dict] = {}
    beta_values = np.arange(args.beta_min, args.beta_max + 1e-12, args.beta_step)

    for name, market_prediction in market_candidates.items():
        market_row = map_time_predictions_to_rows(row_time_cal, cal_time_id, market_prediction)
        for beta in beta_values:
            raw_prediction = base_prediction + float(beta) * market_row
            shrink_info = calibrate_shrink_info(
                y_cal,
                raw_prediction,
                w_cal,
                asset_cal,
                "per_asset",
                float(args.shrink_cap),
            )
            plain_prediction = apply_shrink(raw_prediction, asset_cal, shrink_info)
            plain_info = score_candidate_on_calibration(
                y_cal,
                plain_prediction,
                w_cal,
                row_time_cal,
                "full",
            )

            conditional_info = fit_conditional_scales(
                y_cal,
                plain_prediction,
                w_cal,
                market_row,
                args.conditional_quantiles,
                args.conditional_scale_cap,
            )
            conditional_prediction, conditional_bucket = apply_conditional_scales(
                plain_prediction,
                market_row,
                conditional_info,
            )
            conditional_score = weighted_zero_mean_r2(
                y_cal,
                conditional_prediction,
                w_cal,
            )
            forward_info = forward_conditional_score(
                y_cal,
                plain_prediction,
                w_cal,
                market_row,
                row_time_cal,
                args,
            )
            candidate_rows.append(
                {
                    "market_candidate": name,
                    "beta": float(beta),
                    "plain_full_score": float(plain_info["full_score"]),
                    "plain_first_half_score": float(plain_info["first_half_score"]),
                    "plain_second_half_score": float(plain_info["second_half_score"]),
                    "conditional_full_score": float(conditional_score),
                    "conditional_forward_second_score": float(forward_info["second_half_score"]),
                    "conditional_forward_plain_second_score": float(
                        forward_info["second_half_plain_score"]
                    ),
                    "prediction_std": float(np.std(conditional_prediction)),
                    "conditional_scales": json.dumps(
                        np.asarray(conditional_info["scales"]).tolist()
                    ),
                }
            )
            key = f"{name}|{beta:.8f}"
            candidate_payload[key] = {
                "market_row": market_row,
                "raw_prediction": raw_prediction,
                "plain_prediction": plain_prediction,
                "conditional_prediction": conditional_prediction,
                "conditional_bucket": conditional_bucket,
                "shrink_info": shrink_info,
                "conditional_info": conditional_info,
                "forward_info": forward_info,
            }

    candidate_metrics = pd.DataFrame(candidate_rows)
    # 主排序先看完整 calibration，再用 forward second-half 作为稳定性参考。
    candidate_metrics = candidate_metrics.sort_values(
        ["conditional_full_score", "conditional_forward_second_score"],
        ascending=False,
    ).reset_index(drop=True)
    best_row = candidate_metrics.iloc[0]
    best_key = f"{best_row['market_candidate']}|{float(best_row['beta']):.8f}"
    best = candidate_payload[best_key]

    base_score = float(weighted_zero_mean_r2(y_cal, base_prediction, w_cal))
    error_buckets = target_error_buckets(
        raw_cal,
        base_prediction,
        best["conditional_prediction"],
    )
    calibration_output = raw_cal.copy()
    calibration_output["base_prediction"] = base_prediction
    calibration_output["market_prediction"] = best["market_row"]
    calibration_output["raw_prediction"] = best["raw_prediction"]
    calibration_output["plain_prediction"] = best["plain_prediction"]
    calibration_output["conditional_bucket"] = best["conditional_bucket"]
    calibration_output["prediction"] = best["conditional_prediction"]
    calibration_output["error"] = calibration_output["prediction"] - calibration_output["target"]

    candidate_metrics.to_csv(args.results_dir / "candidate_metrics.csv", index=False)
    candidate_metrics.head(100).to_csv(args.results_dir / "candidate_top100.csv", index=False)
    calibration_output.to_csv(args.results_dir / "calibration_predictions.csv", index=False)
    error_buckets.to_csv(args.results_dir / "error_by_abs_target_bucket.csv", index=False)
    plot_bucket_scores(error_buckets, args.results_dir / "score_by_abs_target_bucket.png")

    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_safe_model_fit": True,
        "official_test_used": False,
        "environment": "quant-competition-wsl",
        "split": {
            "fit": f"{train_start} <= time_id <= {fit_end}",
            "calibration": f"{cal_start} <= time_id <= {train_end}",
        },
        "feature_count": int(len(feature_columns)),
        "market_candidate_count": int(len(market_candidates)),
        "base": {
            "prediction_file": str(args.base_predictions),
            "full_score": base_score,
            "prediction_std": float(np.std(base_prediction)),
            **score_time_blocks(y_cal, base_prediction, w_cal, row_time_cal, 4),
            **score_time_blocks(y_cal, base_prediction, w_cal, row_time_cal, 8),
        },
        "best": {
            "market_candidate": str(best_row["market_candidate"]),
            "beta": float(best_row["beta"]),
            "plain_full_score": float(best_row["plain_full_score"]),
            "conditional_full_score": float(best_row["conditional_full_score"]),
            "conditional_forward_second_score": float(
                best_row["conditional_forward_second_score"]
            ),
            "conditional_forward_plain_second_score": float(
                best_row["conditional_forward_plain_second_score"]
            ),
            "conditional_edges": np.asarray(best["conditional_info"]["edges"]).tolist(),
            "conditional_scales": np.asarray(best["conditional_info"]["scales"]).tolist(),
            "conditional_counts": best["conditional_info"]["counts"],
            "prediction_std": float(np.std(best["conditional_prediction"])),
            "improvement_over_base": float(
                best_row["conditional_full_score"] - base_score
            ),
            **score_time_blocks(
                y_cal,
                best["conditional_prediction"],
                w_cal,
                row_time_cal,
                4,
            ),
            **score_time_blocks(
                y_cal,
                best["conditional_prediction"],
                w_cal,
                row_time_cal,
                8,
            ),
        },
        "model_diagnostics": model_diagnostics,
        "output_files": {
            "candidate_metrics": str(args.results_dir / "candidate_metrics.csv"),
            "calibration_predictions": str(args.results_dir / "calibration_predictions.csv"),
            "error_by_abs_target_bucket": str(
                args.results_dir / "error_by_abs_target_bucket.csv"
            ),
            "score_by_abs_target_bucket": str(
                args.results_dir / "score_by_abs_target_bucket.png"
            ),
        },
    }
    with (args.results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, default=json_default)

    print(json.dumps(metrics["best"], ensure_ascii=False, indent=2, default=json_default))
    print(f"Saved outputs to {args.results_dir}")


if __name__ == "__main__":
    main()
