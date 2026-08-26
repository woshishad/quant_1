from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from final_train_predict import (
    BASE_COLUMNS_TRAIN,
    choose_per_asset_candidates,
    fixed_per_asset_ranking,
    fit_predict_pair_with_args,
    load_feature_ranking,
    merge_prediction_components,
    parquet_paths,
    predict_global_for_frame,
    read_partitioned_frame,
    schema_columns,
    screen_features_by_asset_partitioned,
    search_final_blend,
    time_range,
)
from walk_forward_tabular import apply_shrink, find_best_blend, standardize, weighted_zero_mean_r2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "只用 raw train 内部的 fit/calibration 做最终协议调参；"
            "不会读取 official test，也不会生成 submission。"
        )
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="默认写入 results/final_protocol_tuning_时间戳，避免覆盖旧实验。",
    )
    parser.add_argument(
        "--fixed-features-file",
        type=Path,
        default=Path("results/asset_all_stable_features_100k/selected_features_stable_top128.csv"),
    )
    parser.add_argument("--max-train-time-id", type=int, default=None)
    parser.add_argument("--cal-time-points", type=int, default=20_000)
    parser.add_argument("--lookback-time-points", type=int, nargs="+", default=[60_000, 90_000, 120_000, 150_000])

    # Stage A：低成本扫描全局 Ridge + LightGBM。这里的组合数会直接决定耗时。
    parser.add_argument("--top-k-candidates", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--lgbm-num-leaves-candidates", type=int, nargs="+", default=[15, 31])
    parser.add_argument("--lgbm-min-child-samples-candidates", type=int, nargs="+", default=[6000, 8000, 12000])
    parser.add_argument("--lgbm-reg-lambda-candidates", type=float, nargs="+", default=[300.0, 500.0, 800.0])
    parser.add_argument("--lgbm-learning-rate-candidates", type=float, nargs="+", default=[0.005])
    parser.add_argument("--lgbm-estimators-candidates", type=int, nargs="+", default=[500])
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--lgbm-subsample", type=float, default=0.7)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.7)
    parser.add_argument(
        "--lgbm-seeds",
        type=int,
        nargs="+",
        default=[42],
        help="广扫建议先单 seed；确认候选后再用 11 42 73 做更稳的 seed bagging。",
    )
    parser.add_argument("--model-blend-step", type=float, default=0.01)
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[1.0, 1.2, 1.4])
    parser.add_argument(
        "--global-candidate-score-mode",
        choices=["full", "mean_halves", "min_halves"],
        default="full",
    )

    # Stage B：只对 Stage A 里最好的少量候选补 per-asset 辅助模型，再搜索最终融合权重。
    parser.add_argument(
        "--stage",
        choices=["broad", "full", "two_stage"],
        default="two_stage",
        help="broad 只扫全局模型；full/two_stage 会对前 N 个候选补 per-asset 融合。",
    )
    parser.add_argument("--promote-top-n", type=int, default=3)
    parser.add_argument(
        "--promotion-score-column",
        type=str,
        default="cal_score",
        help="从 broad 候选中挑前 N 个进入 full 阶段时使用的排序列。",
    )
    parser.add_argument("--per-asset-feature-mode", choices=["fixed_ranking", "screen"], default="fixed_ranking")
    parser.add_argument("--per-asset-top-k-candidates", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--per-asset-lgbm-num-leaves-candidates", type=int, nargs="+", default=[7, 15])
    parser.add_argument("--per-asset-lgbm-min-child-samples", type=int, default=1000)
    parser.add_argument("--per-asset-lgbm-reg-lambda", type=float, default=300.0)
    parser.add_argument("--per-asset-lgbm-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--per-asset-candidate-score-mode", choices=["full", "mean_halves", "min_halves"], default="min_halves")
    parser.add_argument("--final-blend-min-global-weight", type=float, default=0.75)
    parser.add_argument("--final-blend-max-global-weight", type=float, default=1.0)
    parser.add_argument("--final-blend-step", type=float, default=0.01)
    parser.add_argument(
        "--save-full-calibration-predictions",
        action="store_true",
        help="full 阶段是否保存逐行 calibration 预测；默认只保存指标，减少磁盘占用。",
    )
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
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def make_results_dir(path: Path | None) -> Path:
    if path is not None:
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"final_protocol_tuning_{timestamp}"


def make_run_args(args: argparse.Namespace, **overrides: object) -> SimpleNamespace:
    """把调参脚本的参数转换成 final_train_predict 里函数需要的参数形状。"""
    values = {
        "ridge_alpha": args.ridge_alpha,
        "lgbm_estimators": int(overrides.get("lgbm_estimators", args.lgbm_estimators_candidates[0])),
        "lgbm_learning_rate": float(overrides.get("lgbm_learning_rate", args.lgbm_learning_rate_candidates[0])),
        "lgbm_min_child_samples": int(
            overrides.get("lgbm_min_child_samples", args.lgbm_min_child_samples_candidates[0])
        ),
        "lgbm_reg_lambda": float(overrides.get("lgbm_reg_lambda", args.lgbm_reg_lambda_candidates[0])),
        "lgbm_subsample": args.lgbm_subsample,
        "lgbm_colsample_bytree": args.lgbm_colsample_bytree,
        "lgbm_seeds": list(args.lgbm_seeds),
        "top_k_candidates": list(args.top_k_candidates),
        "lgbm_num_leaves_candidates": list(args.lgbm_num_leaves_candidates),
        "model_blend_step": args.model_blend_step,
        "shrink_cap_candidates": list(args.shrink_cap_candidates),
        "global_candidate_score_mode": args.global_candidate_score_mode,
        "per_asset_feature_mode": args.per_asset_feature_mode,
        "per_asset_top_k_candidates": list(args.per_asset_top_k_candidates),
        "per_asset_lgbm_num_leaves_candidates": list(args.per_asset_lgbm_num_leaves_candidates),
        "per_asset_lgbm_min_child_samples": args.per_asset_lgbm_min_child_samples,
        "per_asset_lgbm_reg_lambda": args.per_asset_lgbm_reg_lambda,
        "per_asset_lgbm_seeds": list(args.per_asset_lgbm_seeds),
        "per_asset_candidate_score_mode": args.per_asset_candidate_score_mode,
        "final_blend_min_global_weight": args.final_blend_min_global_weight,
        "final_blend_max_global_weight": args.final_blend_max_global_weight,
        "final_blend_step": args.final_blend_step,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def candidate_run_args(args: argparse.Namespace, candidate: dict) -> SimpleNamespace:
    """full 阶段重放某个 broad 候选时，必须复用它当时的 LightGBM 参数。"""
    return make_run_args(
        args,
        lgbm_estimators=int(candidate["lgbm_estimators"]),
        lgbm_learning_rate=float(candidate["lgbm_learning_rate"]),
        lgbm_min_child_samples=int(candidate["lgbm_min_child_samples"]),
        lgbm_reg_lambda=float(candidate["lgbm_reg_lambda"]),
        top_k_candidates=[int(candidate["top_k"])],
        lgbm_num_leaves_candidates=[int(candidate["lgbm_num_leaves"])],
    )


def max_required_feature_count(args: argparse.Namespace) -> int:
    return max(max(args.top_k_candidates), max(args.per_asset_top_k_candidates))


def sanitize_for_csv(row: dict) -> dict:
    clean = {}
    for key, value in row.items():
        if key in {"selected_features", "cal_shrink_info"}:
            clean[key] = json.dumps(value, ensure_ascii=False, default=json_default)
        else:
            clean[key] = value
    return clean


def score_time_blocks(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    block_count: int,
) -> dict[str, float]:
    """把 calibration 按时间切成若干块，观察模型是不是只在某一小段行情上偶然有效。"""
    unique_times = np.unique(time_id)
    chunks = [chunk for chunk in np.array_split(unique_times, block_count) if len(chunk) > 0]
    scores = []
    for chunk in chunks:
        left = int(chunk[0])
        right = int(chunk[-1])
        mask = (time_id >= left) & (time_id <= right)
        scores.append(float(weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])))
    if not scores:
        return {
            f"block{block_count}_mean_score": np.nan,
            f"block{block_count}_min_score": np.nan,
            f"block{block_count}_last_score": np.nan,
            f"block{block_count}_negative_count": 0,
        }
    values = np.asarray(scores, dtype=np.float64)
    return {
        f"block{block_count}_mean_score": float(np.mean(values)),
        f"block{block_count}_min_score": float(np.min(values)),
        f"block{block_count}_last_score": float(values[-1]),
        f"block{block_count}_negative_count": int(np.sum(values < 0.0)),
    }


def load_window_frame(
    train_paths: list[Path],
    ranking: pd.DataFrame,
    train_min_time: int,
    train_end_time: int,
    lookback_time_points: int,
    feature_count: int,
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    """读取某个 lookback 窗口，并生成 fit/cal 的布尔 mask。"""
    train_start_time = max(int(train_min_time), int(train_end_time) - int(lookback_time_points) + 1)
    fit_train_end_time = int(train_end_time) - int(load_window_frame.cal_time_points)
    cal_start_time = fit_train_end_time + 1
    if fit_train_end_time < train_start_time:
        raise ValueError(
            f"lookback={lookback_time_points} 太短，无法容纳 cal_time_points={load_window_frame.cal_time_points}"
        )

    selected_for_read = ranking.head(int(feature_count))["feature_name"].astype(str).tolist()
    frame = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + selected_for_read,
        min_time=train_start_time,
        max_time=int(train_end_time),
    )
    time_values = frame["time_id"].to_numpy(dtype=np.int64)
    fit_mask = (time_values >= train_start_time) & (time_values <= fit_train_end_time)
    cal_mask = (time_values >= cal_start_time) & (time_values <= int(train_end_time))
    window_info = {
        "train_start_time": int(train_start_time),
        "fit_train_end_time": int(fit_train_end_time),
        "cal_start_time": int(cal_start_time),
        "train_end_time": int(train_end_time),
        "lookback_time_points": int(lookback_time_points),
        "fit_rows": int(fit_mask.sum()),
        "cal_rows": int(cal_mask.sum()),
    }
    return frame, window_info, fit_mask, cal_mask


# 给 load_window_frame 挂一个简单属性，避免在每次调用时多传一个固定参数。
load_window_frame.cal_time_points = 20_000


def evaluate_global_grid(
    frame: pd.DataFrame,
    ranking: pd.DataFrame,
    fit_mask: np.ndarray,
    cal_mask: np.ndarray,
    run_args: SimpleNamespace,
    context: dict,
) -> list[dict]:
    """评估一个 lookback + 一组 LightGBM 正则参数下的 top_k/num_leaves 网格。"""
    y_fit = frame.loc[fit_mask, "target"].to_numpy(dtype=np.float32)
    w_fit = frame.loc[fit_mask, "weight"].to_numpy(dtype=np.float32)
    y_cal = frame.loc[cal_mask, "target"].to_numpy(dtype=np.float32)
    w_cal = frame.loc[cal_mask, "weight"].to_numpy(dtype=np.float32)
    asset_cal = frame.loc[cal_mask, "asset_id"].to_numpy(dtype=np.int64)
    time_cal = frame.loc[cal_mask, "time_id"].to_numpy(dtype=np.int64)

    rows: list[dict] = []
    for top_k in run_args.top_k_candidates:
        selected_features = ranking.head(int(top_k))["feature_name"].astype(str).tolist()
        fit_x_raw = frame.loc[fit_mask, selected_features].to_numpy(dtype=np.float32)
        cal_x_raw = frame.loc[cal_mask, selected_features].to_numpy(dtype=np.float32)
        fit_x, cal_x, _, _ = standardize(fit_x_raw, cal_x_raw)

        for leaves in run_args.lgbm_num_leaves_candidates:
            ridge_pred, lgbm_pred, _, _ = fit_predict_pair_with_args(
                fit_x,
                y_fit,
                w_fit,
                cal_x,
                selected_features,
                int(leaves),
                run_args,
            )
            blend = find_best_blend(
                y_cal,
                w_cal,
                asset_cal,
                time_cal,
                ridge_pred,
                lgbm_pred,
                run_args.model_blend_step,
                "per_asset",
                run_args.shrink_cap_candidates,
                run_args.global_candidate_score_mode,
            )
            base_prediction = (
                float(blend["weights"]["ridge"]) * ridge_pred
                + float(blend["weights"]["lgbm"]) * lgbm_pred
            )
            prediction = apply_shrink(base_prediction, asset_cal, blend["shrink_info"])
            score_info = blend["score_info"]
            shrink_summary = blend["shrink_summary"]
            row = {
                **context,
                "scope": "global",
                "top_k": int(top_k),
                "lgbm_num_leaves": int(leaves),
                "lgbm_estimators": int(run_args.lgbm_estimators),
                "lgbm_learning_rate": float(run_args.lgbm_learning_rate),
                "lgbm_min_child_samples": int(run_args.lgbm_min_child_samples),
                "lgbm_reg_lambda": float(run_args.lgbm_reg_lambda),
                "lgbm_subsample": float(run_args.lgbm_subsample),
                "lgbm_colsample_bytree": float(run_args.lgbm_colsample_bytree),
                "lgbm_seeds": ",".join(str(seed) for seed in run_args.lgbm_seeds),
                "ridge_alpha": float(run_args.ridge_alpha),
                "cal_score": float(blend["score"]),
                "cal_score_mode": run_args.global_candidate_score_mode,
                "cal_full_score": float(score_info["full_score"]),
                "cal_first_half_score": float(score_info["first_half_score"]),
                "cal_second_half_score": float(score_info["second_half_score"]),
                "cal_ridge_weight": float(blend["weights"]["ridge"]),
                "cal_lgbm_weight": float(blend["weights"]["lgbm"]),
                "cal_shrink": float(shrink_summary["cal_shrink"]),
                "cal_shrink_min": float(shrink_summary["cal_shrink_min"]),
                "cal_shrink_mean": float(shrink_summary["cal_shrink_mean"]),
                "cal_shrink_max": float(shrink_summary["cal_shrink_max"]),
                "prediction_mean": float(np.mean(prediction)),
                "prediction_std": float(np.std(prediction)),
                "prediction_abs_mean": float(np.mean(np.abs(prediction))),
                "target_std": float(np.std(y_cal)),
                "selected_features": selected_features,
                "cal_shrink_info": blend["shrink_info"],
            }
            row.update(score_time_blocks(y_cal, prediction, w_cal, time_cal, 4))
            row.update(score_time_blocks(y_cal, prediction, w_cal, time_cal, 8))
            rows.append(row)

            # 打印精简版 JSON，方便终端里边跑边看，不刷 selected_features 这种长字段。
            print(
                json.dumps(
                    {key: value for key, value in row.items() if key not in {"selected_features", "cal_shrink_info"}},
                    ensure_ascii=False,
                    default=json_default,
                )
            )
    return rows


def run_broad_stage(
    args: argparse.Namespace,
    train_paths: list[Path],
    ranking: pd.DataFrame,
    train_min_time: int,
    train_end_time: int,
) -> list[dict]:
    all_rows: list[dict] = []
    feature_count = max_required_feature_count(args)

    for lookback in args.lookback_time_points:
        print(f"\n=== broad lookback={lookback} ===")
        frame, window_info, fit_mask, cal_mask = load_window_frame(
            train_paths,
            ranking,
            train_min_time,
            train_end_time,
            int(lookback),
            feature_count,
        )
        print(json.dumps({"window": window_info, "rows_loaded": len(frame)}, ensure_ascii=False))

        hyper_grid = itertools.product(
            args.lgbm_estimators_candidates,
            args.lgbm_learning_rate_candidates,
            args.lgbm_min_child_samples_candidates,
            args.lgbm_reg_lambda_candidates,
        )
        for estimators, learning_rate, min_child, reg_lambda in hyper_grid:
            run_args = make_run_args(
                args,
                lgbm_estimators=int(estimators),
                lgbm_learning_rate=float(learning_rate),
                lgbm_min_child_samples=int(min_child),
                lgbm_reg_lambda=float(reg_lambda),
            )
            context = {
                **window_info,
                "feature_source": str(args.fixed_features_file),
            }
            rows = evaluate_global_grid(frame, ranking, fit_mask, cal_mask, run_args, context)
            all_rows.extend(rows)

        del frame
        gc.collect()
    return all_rows


def select_promoted_candidates(args: argparse.Namespace, rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    if args.promotion_score_column not in rows[0]:
        raise ValueError(f"promotion score column not found: {args.promotion_score_column}")
    ordered = sorted(rows, key=lambda row: float(row[args.promotion_score_column]), reverse=True)

    # 去重：同一个 lookback/top_k/leaves/核心正则参数只保留一次，避免 full 阶段重复训练。
    promoted = []
    seen = set()
    for row in ordered:
        key = (
            row["lookback_time_points"],
            row["top_k"],
            row["lgbm_num_leaves"],
            row["lgbm_estimators"],
            row["lgbm_learning_rate"],
            row["lgbm_min_child_samples"],
            row["lgbm_reg_lambda"],
        )
        if key in seen:
            continue
        seen.add(key)
        promoted.append(copy.deepcopy(row))
        if len(promoted) >= int(args.promote_top_n):
            break
    return promoted


def run_full_candidate(
    args: argparse.Namespace,
    candidate_index: int,
    candidate: dict,
    train_paths: list[Path],
    feature_columns: list[str],
    ranking: pd.DataFrame,
    train_min_time: int,
    train_end_time: int,
) -> dict:
    run_dir = args.results_dir / f"full_candidate_{candidate_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_count = max(int(candidate["top_k"]), max(args.per_asset_top_k_candidates))
    frame, window_info, fit_mask, cal_mask = load_window_frame(
        train_paths,
        ranking,
        train_min_time,
        train_end_time,
        int(candidate["lookback_time_points"]),
        feature_count,
    )
    run_args = candidate_run_args(args, candidate)

    print(f"\n=== full candidate {candidate_index}: {run_dir.name} ===")
    print(json.dumps({key: candidate[key] for key in candidate if key not in {"selected_features", "cal_shrink_info"}}, ensure_ascii=False, default=json_default))

    global_cal = predict_global_for_frame(frame, fit_mask, cal_mask, candidate, run_args, include_target=True)
    pd.DataFrame({"feature_name": candidate["selected_features"]}).to_csv(
        run_dir / "global_selected_features.csv", index=False
    )

    assets = sorted(frame["asset_id"].unique().astype(int).tolist())
    if args.per_asset_feature_mode == "screen":
        aux_ranking = screen_features_by_asset_partitioned(
            train_paths,
            feature_columns,
            frame[BASE_COLUMNS_TRAIN],
            fit_mask,
            cal_mask,
            train_end_time,
        )
    else:
        aux_ranking = fixed_per_asset_ranking(ranking.head(max(args.per_asset_top_k_candidates)), assets)
    aux_ranking.to_csv(run_dir / "per_asset_feature_ranking.csv", index=False)

    aux_candidates, aux_candidate_frame, aux_cal = choose_per_asset_candidates(
        frame,
        aux_ranking,
        fit_mask,
        cal_mask,
        run_args,
    )
    aux_candidate_frame.to_csv(run_dir / "per_asset_candidate_metrics.csv", index=False)

    per_asset_selected_rows = []
    for asset_name, asset_candidate in aux_candidates.items():
        for rank, feature_name in enumerate(asset_candidate["selected_features"], start=1):
            per_asset_selected_rows.append(
                {"asset_id": int(asset_name), "rank": rank, "feature_name": str(feature_name)}
            )
    pd.DataFrame(per_asset_selected_rows).to_csv(run_dir / "per_asset_selected_features.csv", index=False)

    cal_components = merge_prediction_components(global_cal, aux_cal, include_target=True)
    final_blend = search_final_blend(cal_components, run_args)
    cal_components["prediction"] = (
        final_blend["global_weight"] * cal_components["global_prediction"]
        + final_blend["per_asset_weight"] * cal_components["per_asset_prediction"]
    )
    cal_components["error"] = cal_components["prediction"] - cal_components["target"]
    if args.save_full_calibration_predictions:
        cal_components.to_csv(run_dir / "calibration_predictions.csv", index=False)

    y_cal = cal_components["target"].to_numpy(dtype=np.float64)
    w_cal = cal_components["weight"].to_numpy(dtype=np.float64)
    time_cal = cal_components["time_id"].to_numpy(dtype=np.int64)
    final_pred = cal_components["prediction"].to_numpy(dtype=np.float64)
    global_pred = cal_components["global_prediction"].to_numpy(dtype=np.float64)
    per_asset_pred = cal_components["per_asset_prediction"].to_numpy(dtype=np.float64)

    metrics = {
        "leakage_safe": True,
        "official_test_used": False,
        "stage": "full_candidate",
        "candidate_index": int(candidate_index),
        "window": window_info,
        "global_candidate": {
            key: value for key, value in candidate.items() if key not in {"selected_features", "cal_shrink_info"}
        },
        "global_selected_feature_count": int(len(candidate["selected_features"])),
        "per_asset_feature_mode": args.per_asset_feature_mode,
        "per_asset_candidate_count": int(len(aux_candidates)),
        "final_blend": final_blend,
        "calibration_score": float(weighted_zero_mean_r2(y_cal, final_pred, w_cal)),
        "calibration_global_score": float(weighted_zero_mean_r2(y_cal, global_pred, w_cal)),
        "calibration_per_asset_score": float(weighted_zero_mean_r2(y_cal, per_asset_pred, w_cal)),
        "prediction_std": float(np.std(final_pred)),
        "prediction_abs_mean": float(np.mean(np.abs(final_pred))),
    }
    metrics.update(score_time_blocks(y_cal, final_pred, w_cal, time_cal, 4))
    metrics.update(score_time_blocks(y_cal, final_pred, w_cal, time_cal, 8))

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default))

    del frame, global_cal, aux_cal, cal_components
    gc.collect()
    return {
        "candidate_index": int(candidate_index),
        "run_dir": str(run_dir),
        **window_info,
        "global_top_k": int(candidate["top_k"]),
        "global_lgbm_num_leaves": int(candidate["lgbm_num_leaves"]),
        "global_lgbm_min_child_samples": int(candidate["lgbm_min_child_samples"]),
        "global_lgbm_reg_lambda": float(candidate["lgbm_reg_lambda"]),
        "global_lgbm_learning_rate": float(candidate["lgbm_learning_rate"]),
        "final_global_weight": float(final_blend["global_weight"]),
        "final_per_asset_weight": float(final_blend["per_asset_weight"]),
        "calibration_score": metrics["calibration_score"],
        "calibration_global_score": metrics["calibration_global_score"],
        "calibration_per_asset_score": metrics["calibration_per_asset_score"],
        "block4_min_score": metrics["block4_min_score"],
        "block8_min_score": metrics["block8_min_score"],
        "block8_last_score": metrics["block8_last_score"],
    }


def main() -> None:
    args = parse_args()
    args.results_dir = make_results_dir(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    load_window_frame.cal_time_points = int(args.cal_time_points)

    train_paths = parquet_paths(args.raw_data_dir, "train")
    train_min_time, train_max_time_available = time_range(train_paths)
    train_end_time = (
        min(train_max_time_available, int(args.max_train_time_id))
        if args.max_train_time_id is not None
        else train_max_time_available
    )
    available_columns = schema_columns(train_paths)
    feature_columns = [column for column in available_columns if column.startswith("feature_")]
    ranking = load_feature_ranking(args.fixed_features_file, available_columns)
    if max_required_feature_count(args) > len(ranking):
        raise ValueError(
            f"需要至少 {max_required_feature_count(args)} 个稳定特征，但 {args.fixed_features_file} 只有 {len(ranking)} 个"
        )

    config = {
        "leakage_safe": True,
        "official_test_used": False,
        "raw_data_dir": str(args.raw_data_dir),
        "fixed_features_file": str(args.fixed_features_file),
        "train_min_time": int(train_min_time),
        "train_max_time_available": int(train_max_time_available),
        "train_end_time": int(train_end_time),
        "args": vars(args),
    }
    (args.results_dir / "tuning_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    print(f"写入目录: {args.results_dir}")
    print(f"raw train time range: {train_min_time}..{train_max_time_available}; tuning train_end={train_end_time}")

    broad_rows = run_broad_stage(args, train_paths, ranking, train_min_time, train_end_time)
    broad_frame = pd.DataFrame([sanitize_for_csv(row) for row in broad_rows])
    broad_frame = broad_frame.sort_values(args.promotion_score_column, ascending=False).reset_index(drop=True)
    broad_frame.to_csv(args.results_dir / "global_candidate_metrics.csv", index=False)
    broad_frame.head(50).to_csv(args.results_dir / "global_top50_candidates.csv", index=False)

    promoted = select_promoted_candidates(args, broad_rows)
    pd.DataFrame([sanitize_for_csv(row) for row in promoted]).to_csv(args.results_dir / "promoted_global_candidates.csv", index=False)

    full_rows: list[dict] = []
    if args.stage in {"full", "two_stage"}:
        for index, candidate in enumerate(promoted):
            full_rows.append(
                run_full_candidate(
                    args,
                    index,
                    candidate,
                    train_paths,
                    feature_columns,
                    ranking,
                    train_min_time,
                    train_end_time,
                )
            )
        if full_rows:
            full_frame = pd.DataFrame(full_rows).sort_values("calibration_score", ascending=False).reset_index(drop=True)
            full_frame.to_csv(args.results_dir / "full_candidate_summary.csv", index=False)
            best_full = full_frame.iloc[0].to_dict()
            (args.results_dir / "best_full_candidate.json").write_text(
                json.dumps(best_full, indent=2, ensure_ascii=False, default=json_default),
                encoding="utf-8",
            )

    summary = {
        "results_dir": str(args.results_dir),
        "broad_candidate_count": int(len(broad_rows)),
        "promoted_count": int(len(promoted)),
        "full_candidate_count": int(len(full_rows)),
        "best_broad": sanitize_for_csv(broad_rows[0]) if broad_rows else None,
        "best_full": full_rows[0] if full_rows else None,
        "output_files": {
            "global_candidate_metrics": str(args.results_dir / "global_candidate_metrics.csv"),
            "global_top50_candidates": str(args.results_dir / "global_top50_candidates.csv"),
            "promoted_global_candidates": str(args.results_dir / "promoted_global_candidates.csv"),
            "full_candidate_summary": str(args.results_dir / "full_candidate_summary.csv"),
        },
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
