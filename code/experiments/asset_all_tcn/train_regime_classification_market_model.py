from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from final_train_predict import BASE_COLUMNS_TRAIN, parquet_paths, read_partitioned_frame, time_range
from market_mean_ts_model import (
    build_time_feature_frame,
    map_time_predictions_to_rows,
    normalized_weight,
    prepare_matrix,
    score_time_blocks,
    weighted_market_target,
)
from walk_forward_tabular import weighted_zero_mean_r2


CLASS_NAMES = {
    0: "strong_negative",
    1: "mixed",
    2: "strong_positive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="先分类市场横截面状态，再用分类条件幅度头预测市场共同 target。"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--aggregate-ranking-file",
        type=Path,
        default=Path("results/market_target_feature_screen_75k/aggregate_feature_ranking.csv"),
    )
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=Path(
            "results/forward_conditional_exactmarket_blend/calibration_predictions.csv"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/regime_classification_market_75k_probe"),
    )
    parser.add_argument("--train-lookback-time-points", type=int, default=75_000)
    parser.add_argument("--cal-time-points", type=int, default=20_000)
    parser.add_argument("--top-k-candidates", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--logistic-c-candidates", type=float, nargs="+", default=[0.01, 0.10])
    parser.add_argument("--magnitude-ridge-alpha", type=float, default=10_000.0)
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.01)
    parser.add_argument("--lgbm-n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta-min", type=float, default=-0.10)
    parser.add_argument("--beta-max", type=float, default=0.20)
    parser.add_argument("--beta-step", type=float, default=0.01)
    parser.add_argument("--max-train-time-id", type=int, default=None)
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


def raw_feature_name(aggregate_column: str) -> str:
    for suffix in ["_xmean", "_xstd", "_xmin", "_xmax", "_xrange"]:
        if aggregate_column.endswith(suffix):
            return aggregate_column[: -len(suffix)]
    raise ValueError(f"无法识别聚合特征：{aggregate_column}")


def build_regime_labels(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """
    strong_negative: 正 target <= 3；strong_positive: 正 target >= 12；其余为 mixed。
    标签只在训练和验证时由 target 构造，正式预测时由 feature 分类器输出概率。
    """
    working = raw_frame[["time_id", "target"]].copy()
    working["is_positive"] = (working["target"] > 0.0).astype(np.int8)
    summary = (
        working.groupby("time_id", sort=True)
        .agg(
            positive_count=("is_positive", "sum"),
            asset_count=("target", "size"),
        )
        .reset_index()
    )
    summary["regime_class"] = np.where(
        summary["positive_count"] <= 3,
        0,
        np.where(summary["positive_count"] >= 12, 2, 1),
    ).astype(np.int8)
    return summary


def balanced_sample_weight(
    labels: np.ndarray,
    base_weight: np.ndarray,
) -> np.ndarray:
    """让三个市场状态在分类损失中权重接近，避免分类器偏向样本较多的状态。"""
    result = base_weight.astype(np.float64).copy()
    total = float(np.sum(base_weight))
    class_count = len(np.unique(labels))
    for class_id in sorted(np.unique(labels)):
        mask = labels == class_id
        class_weight_sum = max(float(np.sum(base_weight[mask])), 1e-12)
        result[mask] *= total / (class_count * class_weight_sum)
    return normalized_weight(result)


def fit_ridge_head(
    fit_x: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    alpha: float,
) -> Ridge:
    model = Ridge(alpha=float(alpha), solver="lsqr", max_iter=800)
    model.fit(fit_x, target, sample_weight=normalized_weight(weight))
    return model


def weighted_constant(values: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(values * weight) / max(float(np.sum(weight)), 1e-12))


def fit_magnitude_heads(
    fit_z: np.ndarray,
    y_fit: np.ndarray,
    regime_fit: np.ndarray,
    w_fit: np.ndarray,
    cal_z: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    """分别预测一致下跌幅度、混合市场均值和一致上涨幅度。"""
    negative_mask = regime_fit == 0
    mixed_mask = regime_fit == 1
    positive_mask = regime_fit == 2

    negative_model = fit_ridge_head(
        fit_z[negative_mask], np.abs(y_fit[negative_mask]), w_fit[negative_mask], alpha
    )
    mixed_model = fit_ridge_head(
        fit_z[mixed_mask], y_fit[mixed_mask], w_fit[mixed_mask], alpha
    )
    positive_model = fit_ridge_head(
        fit_z[positive_mask], np.abs(y_fit[positive_mask]), w_fit[positive_mask], alpha
    )
    pooled_magnitude_model = fit_ridge_head(fit_z, np.abs(y_fit), w_fit, alpha)
    direct_model = fit_ridge_head(fit_z, y_fit, w_fit, alpha)

    return {
        "negative_magnitude": np.clip(negative_model.predict(cal_z), 0.0, 2.3),
        "mixed_value": np.clip(mixed_model.predict(cal_z), -2.3, 2.3),
        "positive_magnitude": np.clip(positive_model.predict(cal_z), 0.0, 2.3),
        "pooled_magnitude": np.clip(pooled_magnitude_model.predict(cal_z), 0.0, 2.3),
        "direct_value": np.clip(direct_model.predict(cal_z), -2.3, 2.3),
        "negative_constant": weighted_constant(
            np.abs(y_fit[negative_mask]), w_fit[negative_mask]
        ),
        "mixed_constant": weighted_constant(y_fit[mixed_mask], w_fit[mixed_mask]),
        "positive_constant": weighted_constant(
            np.abs(y_fit[positive_mask]), w_fit[positive_mask]
        ),
    }


def fit_classifier_candidates(
    fit_x: np.ndarray,
    fit_z: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    cal_x: np.ndarray,
    cal_z: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    probabilities: dict[str, np.ndarray] = {}
    for c_value in args.logistic_c_candidates:
        model = LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            max_iter=800,
            random_state=int(args.seed),
        )
        model.fit(fit_z, labels, sample_weight=sample_weight)
        probabilities[f"logistic_c{c_value:g}"] = model.predict_proba(cal_z)

    lgbm = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=int(args.lgbm_estimators),
        learning_rate=float(args.lgbm_learning_rate),
        num_leaves=7,
        min_child_samples=300,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=300.0,
        random_state=int(args.seed),
        n_jobs=int(args.lgbm_n_jobs),
        verbose=-1,
    )
    fit_frame = pd.DataFrame(fit_x, columns=feature_names)
    cal_frame = pd.DataFrame(cal_x, columns=feature_names)
    lgbm.fit(fit_frame, labels, sample_weight=sample_weight)
    probabilities["lgbm"] = lgbm.predict_proba(cal_frame)
    return probabilities


def build_market_predictions(
    probability: np.ndarray,
    heads: dict[str, np.ndarray | float],
) -> dict[str, np.ndarray]:
    p_negative = probability[:, 0]
    p_mixed = probability[:, 1]
    p_positive = probability[:, 2]
    negative_magnitude = np.asarray(heads["negative_magnitude"])
    mixed_value = np.asarray(heads["mixed_value"])
    positive_magnitude = np.asarray(heads["positive_magnitude"])
    pooled_magnitude = np.asarray(heads["pooled_magnitude"])
    direct_value = np.asarray(heads["direct_value"])

    soft_conditional = (
        -p_negative * negative_magnitude
        + p_mixed * mixed_value
        + p_positive * positive_magnitude
    )
    soft_constant = (
        -p_negative * float(heads["negative_constant"])
        + p_mixed * float(heads["mixed_constant"])
        + p_positive * float(heads["positive_constant"])
    )
    signed_pooled = (p_positive - p_negative) * pooled_magnitude

    hard_class = np.argmax(probability, axis=1)
    hard_conditional = np.where(
        hard_class == 0,
        -negative_magnitude,
        np.where(hard_class == 2, positive_magnitude, mixed_value),
    )
    confidence = np.max(probability, axis=1)
    # 低置信度硬分类向直接 Ridge 回退，防止把不确定状态输出成大幅共同波动。
    confidence_gate = np.clip((confidence - 1.0 / 3.0) / (2.0 / 3.0), 0.0, 1.0)
    gated_hard = confidence_gate * hard_conditional + (1.0 - confidence_gate) * direct_value
    return {
        "soft_conditional": soft_conditional,
        "soft_constant": soft_constant,
        "signed_pooled": signed_pooled,
        "gated_hard": gated_hard,
        "direct_ridge": direct_value,
    }


def classifier_diagnostics(
    probability: np.ndarray,
    true_class: np.ndarray,
    weight: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    predicted = np.argmax(probability, axis=1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    for actual, pred in zip(true_class, predicted):
        matrix[int(actual), int(pred)] += 1
    rows = []
    recalls = {}
    for class_id in range(3):
        mask = true_class == class_id
        recalls[CLASS_NAMES[class_id]] = float(np.mean(predicted[mask] == class_id))
        for predicted_id in range(3):
            rows.append(
                {
                    "actual_class": CLASS_NAMES[class_id],
                    "predicted_class": CLASS_NAMES[predicted_id],
                    "count": int(matrix[class_id, predicted_id]),
                }
            )
    diagnostics = {
        "accuracy": float(np.mean(predicted == true_class)),
        "weighted_accuracy": float(
            np.sum(weight * (predicted == true_class)) / max(float(np.sum(weight)), 1e-12)
        ),
        "recall_by_class": recalls,
        "predicted_class_share": {
            CLASS_NAMES[class_id]: float(np.mean(predicted == class_id))
            for class_id in range(3)
        },
        "mean_probability": {
            CLASS_NAMES[class_id]: float(np.mean(probability[:, class_id]))
            for class_id in range(3)
        },
    }
    return diagnostics, pd.DataFrame(rows)


def load_base_predictions(raw_cal: pd.DataFrame, path: Path) -> np.ndarray:
    base = pd.read_csv(path, usecols=["time_id", "asset_id", "target", "prediction"])
    base = base.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    if not np.array_equal(
        base[["time_id", "asset_id"]].to_numpy(),
        raw_cal[["time_id", "asset_id"]].to_numpy(),
    ):
        raise ValueError("base predictions 与 calibration 键不一致")
    return base["prediction"].to_numpy(dtype=np.float64)


def plot_confusion(matrix_frame: pd.DataFrame, output_path: Path) -> None:
    matrix = matrix_frame.pivot(
        index="actual_class", columns="predicted_class", values="count"
    ).reindex(index=list(CLASS_NAMES.values()), columns=list(CLASS_NAMES.values()))
    values = matrix.to_numpy(dtype=np.float64)
    row_sum = np.maximum(values.sum(axis=1, keepdims=True), 1.0)
    normalized = values / row_sum
    plt.figure(figsize=(7, 6))
    plt.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(label="row-normalized share")
    for row in range(3):
        for column in range(3):
            plt.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center")
    plt.xticks(range(3), list(CLASS_NAMES.values()), rotation=20)
    plt.yticks(range(3), list(CLASS_NAMES.values()))
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    aggregate_ranking = pd.read_csv(args.aggregate_ranking_file)
    max_top_k = min(max(args.top_k_candidates), len(aggregate_ranking))
    aggregate_columns = (
        aggregate_ranking.head(max_top_k)["aggregate_feature"].astype(str).tolist()
    )
    raw_features = list(dict.fromkeys(raw_feature_name(name) for name in aggregate_columns))

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
    print(
        f"Regime classification split: fit={train_start}..{fit_end}, "
        f"cal={cal_start}..{train_end}"
    )

    raw_train = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + raw_features,
        min_time=train_start,
        max_time=train_end,
    )
    market_targets = weighted_market_target(raw_train)
    regime_labels = build_regime_labels(raw_train)
    time_features, available_aggregate = build_time_feature_frame(
        raw_train, raw_features, [], [], [], []
    )
    missing = sorted(set(aggregate_columns) - set(available_aggregate))
    if missing:
        raise ValueError(f"缺少聚合特征：{missing[:10]}")
    time_frame = (
        time_features.merge(market_targets, on="time_id", how="left")
        .merge(regime_labels, on="time_id", how="left")
        .sort_values("time_id")
        .reset_index(drop=True)
    )
    time_values = time_frame["time_id"].to_numpy(dtype=np.int64)
    fit_mask = (time_values >= train_start) & (time_values <= fit_end)
    cal_mask = (time_values >= cal_start) & (time_values <= train_end)
    fit_time = time_frame.loc[fit_mask].copy()
    cal_time = time_frame.loc[cal_mask].copy()

    raw_cal = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN,
        min_time=cal_start,
        max_time=train_end,
    )
    y_cal = raw_cal["target"].to_numpy(dtype=np.float64)
    w_cal = raw_cal["weight"].to_numpy(dtype=np.float64)
    row_time = raw_cal["time_id"].to_numpy(dtype=np.int64)
    cal_time_id = cal_time["time_id"].to_numpy(dtype=np.int64)
    base_prediction = load_base_predictions(raw_cal, args.base_predictions)
    base_full_score = float(weighted_zero_mean_r2(y_cal, base_prediction, w_cal))

    unique_cal_times = np.unique(row_time)
    split_time = int(unique_cal_times[len(unique_cal_times) // 2])
    first_row_mask = row_time < split_time
    second_row_mask = ~first_row_mask
    first_time_mask = cal_time_id < split_time
    second_time_mask = ~first_time_mask

    candidate_rows = []
    payloads: dict[str, dict] = {}
    classifier_rows = []
    confusion_payloads: dict[str, pd.DataFrame] = {}
    beta_values = np.arange(args.beta_min, args.beta_max + 1e-12, args.beta_step)

    for top_k in args.top_k_candidates:
        columns = aggregate_columns[: int(top_k)]
        fit_x, cal_x, fit_z, cal_z = prepare_matrix(fit_time, cal_time, columns)
        regime_fit = fit_time["regime_class"].to_numpy(dtype=np.int64)
        regime_cal = cal_time["regime_class"].to_numpy(dtype=np.int64)
        y_fit = fit_time["market_target"].to_numpy(dtype=np.float64)
        w_fit = fit_time["weight_sum"].to_numpy(dtype=np.float64)
        classifier_weight = balanced_sample_weight(regime_fit, w_fit)
        heads = fit_magnitude_heads(
            fit_z,
            y_fit,
            regime_fit,
            w_fit,
            cal_z,
            args.magnitude_ridge_alpha,
        )
        classifier_probabilities = fit_classifier_candidates(
            fit_x,
            fit_z,
            regime_fit,
            classifier_weight,
            cal_x,
            cal_z,
            columns,
            args,
        )

        for classifier_name, probability in classifier_probabilities.items():
            classifier_key = f"top{top_k}_{classifier_name}"
            diagnostics, confusion = classifier_diagnostics(
                probability,
                regime_cal,
                cal_time["weight_sum"].to_numpy(dtype=np.float64),
            )
            classifier_rows.append(
                {
                    "classifier": classifier_key,
                    "top_k": int(top_k),
                    "accuracy": diagnostics["accuracy"],
                    "weighted_accuracy": diagnostics["weighted_accuracy"],
                    "negative_recall": diagnostics["recall_by_class"]["strong_negative"],
                    "mixed_recall": diagnostics["recall_by_class"]["mixed"],
                    "positive_recall": diagnostics["recall_by_class"]["strong_positive"],
                }
            )
            confusion_payloads[classifier_key] = confusion
            market_predictions = build_market_predictions(probability, heads)
            for market_name, market_prediction in market_predictions.items():
                market_row = map_time_predictions_to_rows(
                    row_time, cal_time_id, market_prediction
                )
                market_time_r2 = weighted_zero_mean_r2(
                    cal_time["market_target"].to_numpy(dtype=np.float64),
                    market_prediction,
                    cal_time["weight_sum"].to_numpy(dtype=np.float64),
                )
                for beta in beta_values:
                    # 市场共同项直接加到所有 asset，不经过 per-asset shrink。
                    prediction = base_prediction + float(beta) * market_row
                    first_score = weighted_zero_mean_r2(
                        y_cal[first_row_mask],
                        prediction[first_row_mask],
                        w_cal[first_row_mask],
                    )
                    second_score = weighted_zero_mean_r2(
                        y_cal[second_row_mask],
                        prediction[second_row_mask],
                        w_cal[second_row_mask],
                    )
                    full_score = weighted_zero_mean_r2(y_cal, prediction, w_cal)
                    key = (
                        f"top{top_k}|{classifier_name}|{market_name}|{float(beta):.8f}"
                    )
                    candidate_rows.append(
                        {
                            "top_k": int(top_k),
                            "classifier": classifier_name,
                            "market_head": market_name,
                            "beta": float(beta),
                            "first_half_score": float(first_score),
                            "second_half_score": float(second_score),
                            "full_score": float(full_score),
                            "market_time_r2": float(market_time_r2),
                            "prediction_std": float(np.std(prediction)),
                        }
                    )
                    payloads[key] = {
                        "prediction": prediction,
                        "market_row": market_row,
                        "market_prediction": market_prediction,
                        "probability": probability,
                        "regime_cal": regime_cal,
                        "classifier_key": classifier_key,
                    }

    candidates = pd.DataFrame(candidate_rows).sort_values(
        "first_half_score", ascending=False
    ).reset_index(drop=True)
    selected = candidates.iloc[0]
    selected_key = (
        f"top{int(selected['top_k'])}|{selected['classifier']}|"
        f"{selected['market_head']}|{float(selected['beta']):.8f}"
    )
    best = payloads[selected_key]
    selected_probability = best["probability"]
    predicted_class = np.argmax(selected_probability, axis=1)

    output = raw_cal.copy()
    output["base_prediction"] = base_prediction
    output["market_regime_prediction"] = best["market_row"]
    output["prediction"] = best["prediction"]
    output["error"] = output["prediction"] - output["target"]
    time_output = cal_time[
        ["time_id", "market_target", "positive_count", "regime_class", "weight_sum"]
    ].copy()
    time_output["prob_strong_negative"] = selected_probability[:, 0]
    time_output["prob_mixed"] = selected_probability[:, 1]
    time_output["prob_strong_positive"] = selected_probability[:, 2]
    time_output["predicted_regime_class"] = predicted_class
    time_output["market_prediction"] = best["market_prediction"]

    classifier_metrics = pd.DataFrame(classifier_rows).sort_values(
        "weighted_accuracy", ascending=False
    )
    selected_confusion = confusion_payloads[best["classifier_key"]]
    candidates.to_csv(args.results_dir / "candidate_metrics.csv", index=False)
    candidates.head(100).to_csv(args.results_dir / "candidate_top100_by_first_half.csv", index=False)
    classifier_metrics.to_csv(args.results_dir / "classifier_metrics.csv", index=False)
    selected_confusion.to_csv(args.results_dir / "selected_confusion_matrix.csv", index=False)
    output.to_csv(args.results_dir / "calibration_predictions.csv", index=False)
    time_output.to_csv(args.results_dir / "time_regime_predictions.csv", index=False)
    plot_confusion(selected_confusion, args.results_dir / "selected_confusion_matrix.png")

    classifier_diagnostic, _ = classifier_diagnostics(
        selected_probability,
        best["regime_cal"],
        cal_time["weight_sum"].to_numpy(dtype=np.float64),
    )
    base_first_score = weighted_zero_mean_r2(
        y_cal[first_row_mask], base_prediction[first_row_mask], w_cal[first_row_mask]
    )
    base_second_score = weighted_zero_mean_r2(
        y_cal[second_row_mask], base_prediction[second_row_mask], w_cal[second_row_mask]
    )
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_safe_model_fit": True,
        "selection_rule": "classifier/head/beta selected only by calibration first-half R2",
        "regime_definition": {
            "strong_negative": "positive_count <= 3",
            "mixed": "4 <= positive_count <= 11",
            "strong_positive": "positive_count >= 12",
        },
        "base": {
            "prediction_file": str(args.base_predictions),
            "full_score": float(base_full_score),
            "first_half_score": float(base_first_score),
            "second_half_score": float(base_second_score),
        },
        "selected": {
            "top_k": int(selected["top_k"]),
            "classifier": str(selected["classifier"]),
            "market_head": str(selected["market_head"]),
            "beta": float(selected["beta"]),
            "market_time_r2": float(selected["market_time_r2"]),
            "full_score": float(selected["full_score"]),
            "first_half_score": float(selected["first_half_score"]),
            "second_half_score": float(selected["second_half_score"]),
            "second_half_improvement_over_base": float(
                selected["second_half_score"] - base_second_score
            ),
            "classifier_diagnostics": classifier_diagnostic,
            **score_time_blocks(
                y_cal, best["prediction"], w_cal, row_time, 4
            ),
            **score_time_blocks(
                y_cal, best["prediction"], w_cal, row_time, 8
            ),
        },
        "output_files": {
            "candidate_metrics": str(args.results_dir / "candidate_metrics.csv"),
            "classifier_metrics": str(args.results_dir / "classifier_metrics.csv"),
            "calibration_predictions": str(args.results_dir / "calibration_predictions.csv"),
            "time_regime_predictions": str(args.results_dir / "time_regime_predictions.csv"),
            "confusion_plot": str(args.results_dir / "selected_confusion_matrix.png"),
        },
    }
    with (args.results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=json_default)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
