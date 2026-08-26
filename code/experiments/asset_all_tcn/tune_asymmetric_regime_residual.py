from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from market_mean_ts_model import map_time_predictions_to_rows, score_time_blocks
from walk_forward_tabular import weighted_zero_mean_r2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用嵌套时间切分学习正向/负向市场状态概率的非对称残差校准。"
    )
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=Path(
            "results/forward_conditional_exactmarket_blend/calibration_predictions.csv"
        ),
    )
    parser.add_argument(
        "--regime-time-predictions",
        type=Path,
        default=Path(
            "results/regime_classification_market_75k_probe/time_regime_predictions.csv"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/asymmetric_regime_residual_nested_forward"),
    )
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1000.0, 10000.0],
    )
    parser.add_argument(
        "--coefficient-shrink-candidates",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 1.0],
    )
    return parser.parse_args()


def build_feature_variants(time_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    p_negative = time_frame["prob_strong_negative"].to_numpy(dtype=np.float64)
    p_mixed = time_frame["prob_mixed"].to_numpy(dtype=np.float64)
    p_positive = time_frame["prob_strong_positive"].to_numpy(dtype=np.float64)
    market_prediction = time_frame["market_prediction"].to_numpy(dtype=np.float64)
    return {
        "probability_contrasts": np.column_stack(
            [p_positive - p_mixed, p_negative - p_mixed]
        ),
        "asymmetric_excess": np.column_stack(
            [
                np.maximum(p_positive - 1.0 / 3.0, 0.0),
                np.maximum(p_negative - 1.0 / 3.0, 0.0),
                p_mixed - 1.0 / 3.0,
            ]
        ),
        "probabilities_and_market": np.column_stack(
            [p_positive, p_negative, p_mixed, market_prediction]
        ),
        "market_only": market_prediction[:, None],
    }


def standardize_fit_predict(
    train_x: np.ndarray,
    predict_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (train_x - mean) / std, (predict_x - mean) / std


def fit_predict_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    predict_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    train_z, predict_z = standardize_fit_predict(train_x, predict_x)
    model = Ridge(alpha=float(alpha), solver="lsqr", max_iter=800)
    normalized_weight = train_weight / max(float(np.mean(train_weight)), 1e-12)
    model.fit(train_z, train_y, sample_weight=normalized_weight)
    return model.predict(predict_z).astype(np.float64)


def row_score_with_time_correction(
    row_frame: pd.DataFrame,
    correction_time_ids: np.ndarray,
    correction: np.ndarray,
    mask: np.ndarray,
) -> float:
    row_time = row_frame["time_id"].to_numpy(dtype=np.int64)
    correction_row = map_time_predictions_to_rows(row_time, correction_time_ids, correction)
    prediction = row_frame["base_prediction"].to_numpy(dtype=np.float64) + correction_row
    return float(
        weighted_zero_mean_r2(
            row_frame.loc[mask, "target"].to_numpy(dtype=np.float64),
            prediction[mask],
            row_frame.loc[mask, "weight"].to_numpy(dtype=np.float64),
        )
    )


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(
        args.base_predictions,
        usecols=["time_id", "asset_id", "target", "weight", "prediction"],
    ).rename(columns={"prediction": "base_prediction"})
    rows = rows.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    rows["weighted_target"] = rows["weight"] * rows["target"]
    rows["weighted_base"] = rows["weight"] * rows["base_prediction"]
    time_target = (
        rows.groupby("time_id", sort=True)
        .agg(
            weighted_target_sum=("weighted_target", "sum"),
            weighted_base_sum=("weighted_base", "sum"),
            weight_sum=("weight", "sum"),
        )
        .reset_index()
    )
    time_target["target_market"] = (
        time_target["weighted_target_sum"] / time_target["weight_sum"]
    )
    time_target["base_market"] = (
        time_target["weighted_base_sum"] / time_target["weight_sum"]
    )
    time_target["residual_market"] = time_target["target_market"] - time_target["base_market"]

    regime = pd.read_csv(args.regime_time_predictions)
    time_frame = time_target.merge(regime, on="time_id", how="inner", suffixes=("", "_regime"))
    time_frame = time_frame.sort_values("time_id").reset_index(drop=True)
    feature_variants = build_feature_variants(time_frame)
    time_ids = time_frame["time_id"].to_numpy(dtype=np.int64)
    target_residual = time_frame["residual_market"].to_numpy(dtype=np.float64)
    time_weight = time_frame["weight_sum"].to_numpy(dtype=np.float64)

    # 20k calibration: 前 5k 拟合，第二个 5k 选参数，最后 10k 只做外层验证。
    inner_train_end = len(time_frame) // 4
    inner_valid_end = len(time_frame) // 2
    inner_train = np.arange(len(time_frame)) < inner_train_end
    inner_valid = (
        (np.arange(len(time_frame)) >= inner_train_end)
        & (np.arange(len(time_frame)) < inner_valid_end)
    )
    outer_train = np.arange(len(time_frame)) < inner_valid_end
    outer_test = ~outer_train

    row_time = rows["time_id"].to_numpy(dtype=np.int64)
    inner_valid_row = (row_time >= time_ids[inner_train_end]) & (
        row_time <= time_ids[inner_valid_end - 1]
    )
    outer_test_row = row_time >= time_ids[inner_valid_end]
    candidates = []
    for variant_name, features in feature_variants.items():
        for alpha in args.ridge_alphas:
            raw_correction = fit_predict_ridge(
                features[inner_train],
                target_residual[inner_train],
                time_weight[inner_train],
                features[inner_valid],
                float(alpha),
            )
            for coefficient_shrink in args.coefficient_shrink_candidates:
                correction = float(coefficient_shrink) * raw_correction
                full_correction = np.zeros(len(time_frame), dtype=np.float64)
                full_correction[inner_valid] = correction
                inner_score = row_score_with_time_correction(
                    rows,
                    time_ids,
                    full_correction,
                    inner_valid_row,
                )
                candidates.append(
                    {
                        "variant": variant_name,
                        "alpha": float(alpha),
                        "coefficient_shrink": float(coefficient_shrink),
                        "inner_validation_score": float(inner_score),
                    }
                )

    candidate_frame = pd.DataFrame(candidates).sort_values(
        "inner_validation_score", ascending=False
    ).reset_index(drop=True)
    selected = candidate_frame.iloc[0]
    selected_features = feature_variants[str(selected["variant"])]
    outer_raw_correction = fit_predict_ridge(
        selected_features[outer_train],
        target_residual[outer_train],
        time_weight[outer_train],
        selected_features[outer_test],
        float(selected["alpha"]),
    )
    outer_correction = float(selected["coefficient_shrink"]) * outer_raw_correction
    correction_all = np.zeros(len(time_frame), dtype=np.float64)
    correction_all[outer_test] = outer_correction
    correction_row = map_time_predictions_to_rows(row_time, time_ids, correction_all)
    base_prediction = rows["base_prediction"].to_numpy(dtype=np.float64)
    forward_prediction = base_prediction + correction_row

    base_outer_score = weighted_zero_mean_r2(
        rows.loc[outer_test_row, "target"].to_numpy(dtype=np.float64),
        base_prediction[outer_test_row],
        rows.loc[outer_test_row, "weight"].to_numpy(dtype=np.float64),
    )
    new_outer_score = weighted_zero_mean_r2(
        rows.loc[outer_test_row, "target"].to_numpy(dtype=np.float64),
        forward_prediction[outer_test_row],
        rows.loc[outer_test_row, "weight"].to_numpy(dtype=np.float64),
    )

    output = rows[["time_id", "asset_id", "target", "weight", "base_prediction"]].copy()
    output["regime_residual_correction"] = correction_row
    output["prediction"] = forward_prediction
    output["is_outer_test"] = outer_test_row.astype(np.int8)
    output["error"] = output["prediction"] - output["target"]
    time_output = time_frame[["time_id", "target_market", "base_market", "residual_market"]].copy()
    time_output["correction"] = correction_all
    time_output["is_outer_test"] = outer_test.astype(np.int8)

    candidate_frame.to_csv(args.results_dir / "inner_candidate_metrics.csv", index=False)
    output.to_csv(args.results_dir / "forward_calibration_predictions.csv", index=False)
    time_output.to_csv(args.results_dir / "time_residual_corrections.csv", index=False)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_protocol": {
            "inner_train": [int(time_ids[0]), int(time_ids[inner_train_end - 1])],
            "inner_validation": [
                int(time_ids[inner_train_end]),
                int(time_ids[inner_valid_end - 1]),
            ],
            "outer_test": [int(time_ids[inner_valid_end]), int(time_ids[-1])],
        },
        "selected": {
            "variant": str(selected["variant"]),
            "alpha": float(selected["alpha"]),
            "coefficient_shrink": float(selected["coefficient_shrink"]),
            "inner_validation_score": float(selected["inner_validation_score"]),
        },
        "outer_test": {
            "base_score": float(base_outer_score),
            "new_score": float(new_outer_score),
            "improvement": float(new_outer_score - base_outer_score),
            "correction_mean": float(np.mean(outer_correction)),
            "correction_std": float(np.std(outer_correction)),
        },
        "output_files": {
            "candidate_metrics": str(args.results_dir / "inner_candidate_metrics.csv"),
            "forward_predictions": str(
                args.results_dir / "forward_calibration_predictions.csv"
            ),
            "time_corrections": str(args.results_dir / "time_residual_corrections.csv"),
        },
    }
    with (args.results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
