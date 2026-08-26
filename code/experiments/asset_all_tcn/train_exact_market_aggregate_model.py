from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from final_train_predict import (
    BASE_COLUMNS_TRAIN,
    parquet_paths,
    read_partitioned_frame,
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
        description="只使用 market OOF 筛选出的精确横截面统计量训练市场模型。"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--aggregate-ranking-file",
        type=Path,
        default=Path(
            "results/market_target_feature_screen_75k/aggregate_feature_ranking.csv"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/exact_market_aggregate_75k_probe"),
    )
    parser.add_argument("--train-lookback-time-points", type=int, default=75_000)
    parser.add_argument("--cal-time-points", type=int, default=20_000)
    parser.add_argument("--top-k-candidates", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[10.0, 100.0, 1000.0, 10000.0])
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.01)
    parser.add_argument("--lgbm-n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shrink-cap", type=float, default=1.4)
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


def fit_lgbm(
    fit_x: np.ndarray,
    y_fit: np.ndarray,
    w_fit: np.ndarray,
    cal_x: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
) -> np.ndarray:
    model = LGBMRegressor(
        objective="regression",
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
    model.fit(fit_frame, y_fit, sample_weight=normalized_weight(w_fit))
    return model.predict(cal_frame).astype(np.float64)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(args.aggregate_ranking_file)
    if "aggregate_feature" not in ranking.columns:
        raise ValueError("aggregate ranking 必须包含 aggregate_feature 列")
    max_top_k = min(max(args.top_k_candidates), len(ranking))
    selected_aggregate = ranking.head(max_top_k)["aggregate_feature"].astype(str).tolist()
    raw_features = list(dict.fromkeys(raw_feature_name(name) for name in selected_aggregate))

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
        f"Exact aggregate split: fit={train_start}..{fit_end}, cal={cal_start}..{train_end}, "
        f"raw_features={len(raw_features)}, aggregate_candidates={len(selected_aggregate)}"
    )

    raw_train = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + raw_features,
        min_time=train_start,
        max_time=train_end,
    )
    market_targets = weighted_market_target(raw_train)
    time_features, available_aggregate = build_time_feature_frame(
        raw_train, raw_features, [], [], [], []
    )
    missing = sorted(set(selected_aggregate) - set(available_aggregate))
    if missing:
        raise ValueError(f"聚合特征未生成：{missing[:10]}")
    time_frame = time_features.merge(market_targets, on="time_id", how="left")
    time_values = time_frame["time_id"].to_numpy(dtype=np.int64)
    fit_mask = (time_values >= train_start) & (time_values <= fit_end)
    cal_mask = (time_values >= cal_start) & (time_values <= train_end)
    fit_time = time_frame.loc[fit_mask].copy()
    cal_time = time_frame.loc[cal_mask].copy()
    y_fit = fit_time["market_target"].to_numpy(dtype=np.float64)
    w_fit = fit_time["weight_sum"].to_numpy(dtype=np.float64)

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

    rows = []
    payloads: dict[str, dict] = {}
    for top_k in args.top_k_candidates:
        columns = selected_aggregate[: int(top_k)]
        fit_x, cal_x, fit_z, cal_z = prepare_matrix(fit_time, cal_time, columns)
        time_predictions: dict[str, np.ndarray] = {}
        for alpha in args.ridge_alphas:
            model = Ridge(alpha=float(alpha), solver="lsqr", max_iter=800)
            model.fit(fit_z, y_fit, sample_weight=normalized_weight(w_fit))
            time_predictions[f"ridge_a{alpha:g}"] = model.predict(cal_z).astype(np.float64)
        time_predictions["lgbm"] = fit_lgbm(
            fit_x, y_fit, w_fit, cal_x, columns, args
        )

        # Ridge 与 LightGBM 的简单融合，权重网格保持低维，减少校准过拟合。
        ridge_names = [name for name in time_predictions if name.startswith("ridge_")]
        for ridge_name in ridge_names:
            for lgbm_weight in np.arange(0.0, 1.0 + 1e-12, 0.10):
                name = f"{ridge_name}_lgbm_w{lgbm_weight:.1f}"
                market_time_prediction = (
                    (1.0 - float(lgbm_weight)) * time_predictions[ridge_name]
                    + float(lgbm_weight) * time_predictions["lgbm"]
                )
                market_row = map_time_predictions_to_rows(
                    row_time, cal_time_id, market_time_prediction
                )
                shrink_info = calibrate_shrink_info(
                    y_cal,
                    market_row,
                    w_cal,
                    asset_cal,
                    "per_asset",
                    float(args.shrink_cap),
                )
                prediction = apply_shrink(market_row, asset_cal, shrink_info)
                score_info = score_candidate_on_calibration(
                    y_cal, prediction, w_cal, row_time, "full"
                )
                selection_score = min(
                    float(score_info["first_half_score"]),
                    float(score_info["second_half_score"]),
                )
                key = f"{top_k}|{name}"
                rows.append(
                    {
                        "top_k": int(top_k),
                        "model": name,
                        "selection_score": selection_score,
                        "full_score": float(score_info["full_score"]),
                        "first_half_score": float(score_info["first_half_score"]),
                        "second_half_score": float(score_info["second_half_score"]),
                        "prediction_std": float(np.std(prediction)),
                        "market_time_raw_r2": float(
                            weighted_zero_mean_r2(
                                cal_time["market_target"].to_numpy(dtype=np.float64),
                                market_time_prediction,
                                cal_time["weight_sum"].to_numpy(dtype=np.float64),
                            )
                        ),
                    }
                )
                payloads[key] = {
                    "market_row": market_row,
                    "prediction": prediction,
                    "market_time_prediction": market_time_prediction,
                    "columns": columns,
                    "shrink_info": shrink_info,
                }

    candidate_metrics = pd.DataFrame(rows).sort_values(
        ["selection_score", "full_score"], ascending=False
    ).reset_index(drop=True)
    best_row = candidate_metrics.iloc[0]
    best = payloads[f"{int(best_row['top_k'])}|{best_row['model']}"]

    output = raw_cal.copy()
    output["raw_market_prediction"] = best["market_row"]
    output["prediction"] = best["prediction"]
    output["error"] = output["prediction"] - output["target"]
    time_output = cal_time[["time_id", "market_target", "weight_sum"]].copy()
    time_output["market_prediction"] = best["market_time_prediction"]
    candidate_metrics.to_csv(args.results_dir / "candidate_metrics.csv", index=False)
    candidate_metrics.head(100).to_csv(args.results_dir / "candidate_top100.csv", index=False)
    output.to_csv(args.results_dir / "calibration_predictions.csv", index=False)
    time_output.to_csv(args.results_dir / "market_calibration_predictions.csv", index=False)
    pd.DataFrame({"feature_name": best["columns"]}).to_csv(
        args.results_dir / "selected_aggregate_features.csv", index=False
    )

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_safe": True,
        "aggregate_ranking_file": str(args.aggregate_ranking_file),
        "base_raw_feature_count": int(len(raw_features)),
        "best": {
            "top_k": int(best_row["top_k"]),
            "model": str(best_row["model"]),
            "selection_score": float(best_row["selection_score"]),
            "full_score": float(best_row["full_score"]),
            "first_half_score": float(best_row["first_half_score"]),
            "second_half_score": float(best_row["second_half_score"]),
            "market_time_raw_r2": float(best_row["market_time_raw_r2"]),
            "prediction_std": float(best_row["prediction_std"]),
            **score_time_blocks(
                y_cal, best["prediction"], w_cal, row_time, 4
            ),
            **score_time_blocks(
                y_cal, best["prediction"], w_cal, row_time, 8
            ),
        },
        "output_files": {
            "candidate_metrics": str(args.results_dir / "candidate_metrics.csv"),
            "calibration_predictions": str(args.results_dir / "calibration_predictions.csv"),
            "market_calibration_predictions": str(
                args.results_dir / "market_calibration_predictions.csv"
            ),
            "selected_aggregate_features": str(
                args.results_dir / "selected_aggregate_features.csv"
            ),
        },
    }
    with (args.results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=json_default)
    print(json.dumps(result["best"], ensure_ascii=False, indent=2, default=json_default))
    print(f"Saved outputs to {args.results_dir}")


if __name__ == "__main__":
    main()
