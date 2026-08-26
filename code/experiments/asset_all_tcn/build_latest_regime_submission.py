from __future__ import annotations

import argparse
import gc
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from final_train_predict import (
    BASE_COLUMNS_TEST,
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
    weighted_market_target,
)
from train_regime_classification_market_model import (
    balanced_sample_weight,
    build_market_predictions,
    build_regime_labels,
    fit_magnitude_heads,
    raw_feature_name,
)
from walk_forward_tabular import apply_shrink, calibrate_shrink_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按当前最佳多层模型重训缺失组件，并生成官方提交 CSV/ZIP。"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--stable-features-file",
        type=Path,
        default=Path(
            "results/asset_all_stable_features_100k/selected_features_stable_top128.csv"
        ),
    )
    parser.add_argument(
        "--aggregate-ranking-file",
        type=Path,
        default=Path("results/market_target_feature_screen_75k/aggregate_feature_ranking.csv"),
    )
    parser.add_argument(
        "--exact-calibration-predictions",
        type=Path,
        default=Path("results/exact_market_aggregate_75k_probe/calibration_predictions.csv"),
    )
    parser.add_argument(
        "--threeway-summary",
        type=Path,
        default=Path("results/blend_best_panel_market32_75k_cal20k/summary.json"),
    )
    parser.add_argument(
        "--market32-metrics",
        type=Path,
        default=Path("results/asset_all_market_mean_simple_top32_75k_probe/metrics.json"),
    )
    parser.add_argument(
        "--neutralized-test-predictions",
        type=Path,
        default=Path(
            "results/final_neutralize_continuous_full_alpha_refined_lgbm/"
            "final_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--panel-test-predictions",
        type=Path,
        default=Path(
            "results/asset_all_panel_market_relative_75k_final/final_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/final_latest_regime_classification_model"),
    )
    parser.add_argument("--train-lookback-time-points", type=int, default=75_000)
    parser.add_argument("--market32-top-k", type=int, default=32)
    parser.add_argument("--regime-top-k", type=int, default=32)
    parser.add_argument("--exact-top-k", type=int, default=8)
    parser.add_argument("--lgbm-estimators", type=int, default=500)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.01)
    parser.add_argument("--lgbm-n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exact-cutoff", type=float, default=0.0470988437533378)
    parser.add_argument("--exact-high-weight", type=float, default=0.40)
    parser.add_argument("--regime-beta", type=float, default=0.20)
    parser.add_argument("--save-component-predictions", action="store_true")
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


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_zip(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")


def prediction_stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "finite_count": int(np.sum(np.isfinite(values))),
        "null_count": int(np.sum(~np.isfinite(values))),
    }


def load_component_prediction(path: Path, expected_row_id: np.ndarray, name: str) -> np.ndarray:
    frame = pd.read_csv(path, usecols=["row_id", "prediction"])
    frame = frame.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(frame["row_id"].to_numpy(dtype=np.int64), expected_row_id):
        raise ValueError(f"{name} 的 row_id 与 raw test 不一致")
    prediction = frame["prediction"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(prediction)):
        raise ValueError(f"{name} 包含非有限预测")
    return prediction


def fit_market32_prediction(
    train_time: pd.DataFrame,
    test_time: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> np.ndarray:
    """复现 calibration 最佳的 95% 小树 + 5% 稍大树 Market32。"""
    train_x, test_x, _, _ = prepare_matrix(train_time, test_time, feature_columns)
    y_train = train_time["market_target"].to_numpy(dtype=np.float64)
    w_train = train_time["weight_sum"].to_numpy(dtype=np.float64)
    train_frame = pd.DataFrame(train_x, columns=feature_columns)
    test_frame = pd.DataFrame(test_x, columns=feature_columns)
    predictions = []
    configs = [
        (7, 300, 300.0, 0.95),
        (15, 300, 100.0, 0.05),
    ]
    for leaves, child, reg_lambda, blend_weight in configs:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=int(args.lgbm_estimators),
            learning_rate=float(args.lgbm_learning_rate),
            num_leaves=int(leaves),
            min_child_samples=int(child),
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=float(reg_lambda),
            random_state=int(args.seed),
            n_jobs=int(args.lgbm_n_jobs),
            verbose=-1,
        )
        model.fit(train_frame, y_train, sample_weight=normalized_weight(w_train))
        predictions.append(float(blend_weight) * model.predict(test_frame))
    return np.sum(np.vstack(predictions), axis=0).astype(np.float64)


def exact_shrink_info(path: Path) -> dict:
    """从原 calibration 重新计算 Exact-Market 的 per-asset shrink，避免手抄参数。"""
    frame = pd.read_csv(
        path,
        usecols=["asset_id", "target", "weight", "raw_market_prediction"],
    )
    return calibrate_shrink_info(
        frame["target"].to_numpy(dtype=np.float64),
        frame["raw_market_prediction"].to_numpy(dtype=np.float64),
        frame["weight"].to_numpy(dtype=np.float64),
        frame["asset_id"].to_numpy(dtype=np.int64),
        "per_asset",
        1.4,
    )


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    sample_submission_path = args.raw_data_dir / "sample_submission.csv"

    train_paths = parquet_paths(args.raw_data_dir, "train")
    test_paths = parquet_paths(args.raw_data_dir, "test")
    available_columns = schema_columns(train_paths)
    train_min_time, train_max_time = time_range(train_paths)
    test_min_time, test_max_time = time_range(test_paths)
    train_start_time = max(
        train_min_time,
        train_max_time - int(args.train_lookback_time_points) + 1,
    )

    stable_ranking = load_feature_ranking(args.stable_features_file, available_columns)
    market32_raw_features = (
        stable_ranking.head(int(args.market32_top_k))["feature_name"].astype(str).tolist()
    )
    aggregate_ranking = pd.read_csv(args.aggregate_ranking_file)
    aggregate_columns = (
        aggregate_ranking.head(int(args.regime_top_k))["aggregate_feature"]
        .astype(str)
        .tolist()
    )
    exact_columns = aggregate_columns[: int(args.exact_top_k)]
    regime_columns = aggregate_columns[: int(args.regime_top_k)]
    aggregate_raw_features = [raw_feature_name(name) for name in regime_columns]
    read_features = sorted(set(market32_raw_features + aggregate_raw_features))
    market32_columns = [
        f"{feature}{suffix}"
        for feature in market32_raw_features
        for suffix in ["_xmean", "_xstd", "_xmin", "_xmax", "_xrange"]
    ]
    print(
        f"Final train={train_start_time}..{train_max_time}; "
        f"test={test_min_time}..{test_max_time}; raw_features={len(read_features)}"
    )

    # 只读取顶层三个市场模型真正需要的特征，控制 320 万测试行的内存占用。
    raw_train = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + read_features,
        min_time=train_start_time,
        max_time=train_max_time,
    )
    raw_test = read_partitioned_frame(
        test_paths,
        BASE_COLUMNS_TEST + read_features,
        min_time=test_min_time,
        max_time=test_max_time,
    )
    test_keys = raw_test[BASE_COLUMNS_TEST].copy()
    expected_row_id = test_keys["row_id"].to_numpy(dtype=np.int64)
    test_asset = test_keys["asset_id"].to_numpy(dtype=np.int64)
    test_time = test_keys["time_id"].to_numpy(dtype=np.int64)

    market_targets = weighted_market_target(raw_train)
    regime_labels = build_regime_labels(raw_train)
    train_time_features, train_aggregate_columns = build_time_feature_frame(
        raw_train, read_features, [], [], [], []
    )
    test_time_features, test_aggregate_columns = build_time_feature_frame(
        raw_test, read_features, [], [], [], []
    )
    required_columns = set(market32_columns + regime_columns)
    missing = sorted(
        required_columns
        - set(train_aggregate_columns)
        - set(test_aggregate_columns)
    )
    if missing:
        raise ValueError(f"缺少聚合特征：{missing[:10]}")
    train_time_frame = (
        train_time_features.merge(market_targets, on="time_id", how="left")
        .merge(regime_labels, on="time_id", how="left")
        .sort_values("time_id")
        .reset_index(drop=True)
    )
    test_time_frame = test_time_features.sort_values("time_id").reset_index(drop=True)
    del raw_train, raw_test, train_time_features, test_time_features
    gc.collect()

    # 组件一：Market32 时间预测 -> 映射回每个 asset -> 使用 calibration shrink。
    market32_time_prediction = fit_market32_prediction(
        train_time_frame,
        test_time_frame,
        market32_columns,
        args,
    )
    market32_row_raw = map_time_predictions_to_rows(
        test_time,
        test_time_frame["time_id"].to_numpy(dtype=np.int64),
        market32_time_prediction,
    )
    market32_metrics = read_json(args.market32_metrics)
    market32_shrink_info = market32_metrics["model"]["shrink_info"]
    market32_prediction = apply_shrink(market32_row_raw, test_asset, market32_shrink_info)

    # 组件二：Exact-Market，纯 Ridge alpha=10000，只使用8个精确统计量。
    _, _, exact_train_z, exact_test_z = prepare_matrix(
        train_time_frame, test_time_frame, exact_columns
    )
    exact_model = Ridge(alpha=10_000.0, solver="lsqr", max_iter=800)
    exact_model.fit(
        exact_train_z,
        train_time_frame["market_target"].to_numpy(dtype=np.float64),
        sample_weight=normalized_weight(
            train_time_frame["weight_sum"].to_numpy(dtype=np.float64)
        ),
    )
    exact_time_prediction = exact_model.predict(exact_test_z).astype(np.float64)
    exact_row_raw = map_time_predictions_to_rows(
        test_time,
        test_time_frame["time_id"].to_numpy(dtype=np.int64),
        exact_time_prediction,
    )
    exact_prediction = apply_shrink(
        exact_row_raw,
        test_asset,
        exact_shrink_info(args.exact_calibration_predictions),
    )

    # 组件三：三分类 Logistic + 三个条件幅度 Ridge，输出所有 asset 共用的市场修正。
    regime_train_x, regime_test_x, regime_train_z, regime_test_z = prepare_matrix(
        train_time_frame, test_time_frame, regime_columns
    )
    regime_class = train_time_frame["regime_class"].to_numpy(dtype=np.int64)
    train_weight_sum = train_time_frame["weight_sum"].to_numpy(dtype=np.float64)
    classifier = LogisticRegression(
        C=0.01,
        solver="lbfgs",
        max_iter=800,
        random_state=int(args.seed),
    )
    classifier.fit(
        regime_train_z,
        regime_class,
        sample_weight=balanced_sample_weight(regime_class, train_weight_sum),
    )
    regime_probability = classifier.predict_proba(regime_test_z)
    magnitude_heads = fit_magnitude_heads(
        regime_train_z,
        train_time_frame["market_target"].to_numpy(dtype=np.float64),
        regime_class,
        train_weight_sum,
        regime_test_z,
        10_000.0,
    )
    regime_time_prediction = build_market_predictions(
        regime_probability, magnitude_heads
    )["soft_conditional"]
    regime_row_prediction = map_time_predictions_to_rows(
        test_time,
        test_time_frame["time_id"].to_numpy(dtype=np.int64),
        regime_time_prediction,
    )
    del regime_train_x, regime_test_x, regime_train_z, regime_test_z
    gc.collect()

    # 已有两个高成本逐行组件直接读取其官方测试预测。
    neutralized_prediction = load_component_prediction(
        args.neutralized_test_predictions, expected_row_id, "neutralized"
    )
    panel_prediction = load_component_prediction(
        args.panel_test_predictions, expected_row_id, "panel"
    )

    # 三路基础融合，并复用 calibration 选择出的 per-asset shrink。
    threeway_raw = (
        0.54 * neutralized_prediction
        + 0.22 * panel_prediction
        + 0.24 * market32_prediction
    )
    threeway_summary = read_json(args.threeway_summary)
    threeway_shrink_info = json.loads(
        threeway_summary["best_candidate"]["shrink_info"]
    )
    base_threeway = apply_shrink(threeway_raw, test_asset, threeway_shrink_info)

    # Exact-Market 只有在幅度超过固定 calibration 阈值时进入融合。
    high_exact_regime = np.abs(exact_prediction) >= float(args.exact_cutoff)
    base_conditional = np.where(
        high_exact_regime,
        (1.0 - float(args.exact_high_weight)) * base_threeway
        + float(args.exact_high_weight) * exact_prediction,
        base_threeway,
    )
    final_prediction = (
        base_conditional + float(args.regime_beta) * regime_row_prediction
    ).astype(np.float64)
    if not np.all(np.isfinite(final_prediction)):
        raise ValueError("最终预测包含 NaN/Inf")

    # 严格按照 sample_submission 的 row_id 顺序输出。
    prediction_frame = pd.DataFrame(
        {"row_id": expected_row_id, "target": final_prediction}
    ).sort_values("row_id", kind="mergesort")
    sample = pd.read_csv(sample_submission_path, usecols=["row_id"])
    if np.array_equal(
        sample["row_id"].to_numpy(dtype=np.int64),
        prediction_frame["row_id"].to_numpy(dtype=np.int64),
    ):
        submission = pd.DataFrame(
            {"row_id": sample["row_id"], "target": prediction_frame["target"]}
        )
    else:
        submission = sample.merge(prediction_frame, on="row_id", how="left", validate="one_to_one")
    if submission["target"].isna().any():
        raise ValueError("submission 合并后存在缺失 target")

    submission_path = args.results_dir / "submission.csv"
    zip_path = args.results_dir / "submission.zip"
    submission.to_csv(submission_path, index=False)
    save_zip(submission_path, zip_path)

    component_path = None
    if args.save_component_predictions:
        component_path = args.results_dir / "final_test_predictions.csv"
        component_frame = test_keys.copy()
        component_frame["prediction_neutralized"] = neutralized_prediction.astype(np.float32)
        component_frame["prediction_panel"] = panel_prediction.astype(np.float32)
        component_frame["prediction_market32"] = market32_prediction.astype(np.float32)
        component_frame["prediction_exact"] = exact_prediction.astype(np.float32)
        component_frame["prediction_regime"] = regime_row_prediction.astype(np.float32)
        component_frame["prediction_base_threeway"] = base_threeway.astype(np.float32)
        component_frame["prediction_base_conditional"] = base_conditional.astype(np.float32)
        component_frame["prediction"] = final_prediction.astype(np.float32)
        component_frame.to_csv(component_path, index=False)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "environment": "quant-competition-wsl",
        "official_test_used_for_training": False,
        "train_time_range": [int(train_start_time), int(train_max_time)],
        "test_time_range": [int(test_min_time), int(test_max_time)],
        "row_count": int(len(submission)),
        "model": {
            "threeway_weights": {
                "neutralized": 0.54,
                "panel": 0.22,
                "market32": 0.24,
            },
            "exact_cutoff": float(args.exact_cutoff),
            "exact_high_weight": float(args.exact_high_weight),
            "regime_classifier": "LogisticRegression(C=0.01, balanced sample weight)",
            "regime_market_head": "soft_conditional Ridge heads alpha=10000",
            "regime_beta": float(args.regime_beta),
        },
        "prediction_stats": prediction_stats(final_prediction),
        "component_stats": {
            "base_threeway": prediction_stats(base_threeway),
            "exact": prediction_stats(exact_prediction),
            "regime_market": prediction_stats(regime_row_prediction),
            "high_exact_row_share": float(np.mean(high_exact_regime)),
            "regime_mean_probability": {
                "strong_negative": float(np.mean(regime_probability[:, 0])),
                "mixed": float(np.mean(regime_probability[:, 1])),
                "strong_positive": float(np.mean(regime_probability[:, 2])),
            },
        },
        "output_files": {
            "submission_csv": str(submission_path),
            "submission_zip": str(zip_path),
            "component_predictions": str(component_path) if component_path else None,
        },
    }
    with (args.results_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=json_default)
    reproduce_command = (
        f"{sys.executable} code/experiments/asset_all_tcn/"
        f"build_latest_regime_submission.py --results-dir {args.results_dir}\n"
    )
    (args.results_dir / "reproduce_command.sh").write_text(reproduce_command, encoding="utf-8")
    (args.results_dir / "reproduce_command.ps1").write_text(
        "# This result was generated in WSL; run reproduce_command.sh.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
