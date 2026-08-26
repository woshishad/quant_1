from __future__ import annotations

import argparse
import gc
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from final_train_predict import (
    BASE_COLUMNS_TRAIN,
    load_feature_ranking,
    parquet_paths,
    read_partitioned_frame,
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
        description="Ridge 先预测 target，再用 LightGBM 预测 Ridge 残差的 leakage-safe 调参脚本。"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--fixed-features-file",
        type=Path,
        default=Path("results/asset_all_stable_features_100k/selected_features_stable_top128.csv"),
    )
    parser.add_argument("--max-train-time-id", type=int, default=None)
    parser.add_argument("--lookback-time-points", type=int, nargs="+", default=[120_000])
    parser.add_argument("--cal-time-points", type=int, default=20_000)

    # 因子选择：stable 最稳；inner_screen 只用 fit 内部的早/晚切分筛因子；hybrid 合并两种排名。
    parser.add_argument(
        "--feature-selection-modes",
        choices=["stable", "inner_screen", "hybrid"],
        nargs="+",
        default=["stable"],
    )
    parser.add_argument("--screen-time-points", type=int, default=20_000)
    parser.add_argument("--hybrid-stable-weight", type=float, default=0.5)
    parser.add_argument("--top-k-candidates", type=int, nargs="+", default=[64, 128])

    # Ridge 是低方差基线，残差模型只负责补它没解释掉的部分。
    parser.add_argument("--ridge-alpha-candidates", type=float, nargs="+", default=[10.0, 100.0])
    parser.add_argument("--residual-weight-min", type=float, default=0.0)
    parser.add_argument("--residual-weight-max", type=float, default=1.0)
    parser.add_argument("--residual-weight-step", type=float, default=0.05)

    # 残差模型参数。默认仍用 LightGBM CPU；如果安装了 XGBoost，可切到 CUDA。
    parser.add_argument("--residual-model", choices=["lightgbm", "xgboost", "catboost"], default="lightgbm")
    parser.add_argument("--lgbm-num-leaves-candidates", type=int, nargs="+", default=[7, 15])
    parser.add_argument("--lgbm-estimators-candidates", type=int, nargs="+", default=[300])
    parser.add_argument("--lgbm-learning-rate-candidates", type=float, nargs="+", default=[0.005])
    parser.add_argument("--lgbm-min-child-samples-candidates", type=int, nargs="+", default=[8000, 12000])
    parser.add_argument("--lgbm-reg-lambda-candidates", type=float, nargs="+", default=[500.0, 1000.0])
    parser.add_argument("--lgbm-subsample", type=float, default=0.7)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.7)
    parser.add_argument("--lgbm-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--lgbm-n-jobs", type=int, default=-1, help="-1 表示吃满 CPU；可设为 8/12 限制线程数。")
    parser.add_argument("--xgb-max-depth-candidates", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--xgb-min-child-weight-candidates", type=float, nargs="+", default=[100.0])
    parser.add_argument("--xgb-reg-lambda-candidates", type=float, nargs="+", default=[500.0])
    parser.add_argument("--xgb-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--xgb-tree-method", choices=["hist", "approx"], default="hist")
    parser.add_argument("--xgb-max-bin", type=int, default=255)
    parser.add_argument("--catboost-task-type", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--catboost-depth-candidates", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--catboost-iterations-candidates", type=int, nargs="+", default=[300])
    parser.add_argument("--catboost-learning-rate-candidates", type=float, nargs="+", default=[0.01])
    parser.add_argument("--catboost-l2-leaf-reg-candidates", type=float, nargs="+", default=[100.0, 300.0])
    parser.add_argument("--catboost-random-strength", type=float, default=1.0)

    # shrink 和分数选择。per_asset shrink 往往能降低弱标的拖累。
    parser.add_argument("--shrink-mode", choices=["global", "per_asset"], default="per_asset")
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[1.0, 1.2, 1.4])
    parser.add_argument("--candidate-score-mode", choices=["full", "mean_halves", "min_halves"], default="full")
    parser.add_argument("--save-calibration-predictions", action="store_true")
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


def make_results_dir(path: Path | None) -> Path:
    if path is not None:
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"asset_all_residual_tuning_{timestamp}"


def score_time_blocks(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    block_count: int,
) -> dict[str, float]:
    """按时间块看稳定性，避免只被一小段行情抬高总分。"""
    unique_times = np.unique(time_id)
    chunks = [chunk for chunk in np.array_split(unique_times, block_count) if len(chunk) > 0]
    scores = []
    for chunk in chunks:
        mask = (time_id >= int(chunk[0])) & (time_id <= int(chunk[-1]))
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


def format_token(value: object) -> str:
    """把参数值转成适合放进文件名的短字符串，避免小数点和特殊字符干扰后续读取。"""
    if value is None:
        return "none"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}".replace("-", "m").replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p").replace(",", "_")


def candidate_prediction_path(
    output_dir: Path,
    feature_mode: str,
    top_k: int,
    ridge_alpha: float,
    model_params: dict,
) -> Path:
    """为每个候选生成稳定的预测文件名，方便不同实验结果后续做融合。"""
    model_name = str(model_params["residual_model"])
    if model_name == "lightgbm":
        model_part = (
            f"lgbm_leaves{format_token(model_params.get('lgbm_num_leaves'))}"
            f"_reg{format_token(model_params.get('lgbm_reg_lambda'))}"
        )
    elif model_name == "catboost":
        model_part = (
            f"catboost_depth{format_token(model_params.get('catboost_depth'))}"
            f"_l2{format_token(model_params.get('catboost_l2_leaf_reg'))}"
        )
    else:
        model_part = (
            f"xgb_depth{format_token(model_params.get('xgb_max_depth'))}"
            f"_reg{format_token(model_params.get('xgb_reg_lambda'))}"
        )
    file_name = (
        f"calibration_predictions_{feature_mode}"
        f"_top{int(top_k)}"
        f"_ridge{format_token(ridge_alpha)}"
        f"_{model_part}.csv"
    )
    return output_dir / "calibration_predictions" / file_name


def save_calibration_predictions(
    path: Path,
    y_cal: np.ndarray,
    w_cal: np.ndarray,
    asset_cal: np.ndarray,
    time_cal: np.ndarray,
    ridge_cal: np.ndarray,
    residual_cal: np.ndarray,
    raw_pred: np.ndarray,
    pred: np.ndarray,
) -> None:
    """保存校准集逐行预测；融合脚本会按 time_id/asset_id 对齐这些文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time_id": time_cal.astype(np.int64, copy=False),
            "asset_id": asset_cal.astype(np.int64, copy=False),
            "target": y_cal.astype(np.float32, copy=False),
            "weight": w_cal.astype(np.float32, copy=False),
            "ridge_prediction": ridge_cal.astype(np.float32, copy=False),
            "residual_prediction": residual_cal.astype(np.float32, copy=False),
            "raw_prediction": raw_pred.astype(np.float32, copy=False),
            "prediction": pred.astype(np.float32, copy=False),
        }
    ).to_csv(path, index=False)


def load_window(
    train_paths: list[Path],
    columns: list[str],
    train_min_time: int,
    train_end_time: int,
    lookback_time_points: int,
    cal_time_points: int,
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    train_start_time = max(int(train_min_time), int(train_end_time) - int(lookback_time_points) + 1)
    fit_end_time = int(train_end_time) - int(cal_time_points)
    cal_start_time = fit_end_time + 1
    if fit_end_time < train_start_time:
        raise ValueError(f"lookback={lookback_time_points} 太短，无法容纳 calibration={cal_time_points}")

    frame = read_partitioned_frame(
        train_paths,
        columns,
        min_time=train_start_time,
        max_time=int(train_end_time),
    )
    time_values = frame["time_id"].to_numpy(dtype=np.int64)
    fit_mask = (time_values >= train_start_time) & (time_values <= fit_end_time)
    cal_mask = (time_values >= cal_start_time) & (time_values <= int(train_end_time))
    info = {
        "train_start_time": int(train_start_time),
        "fit_train_end_time": int(fit_end_time),
        "cal_start_time": int(cal_start_time),
        "train_end_time": int(train_end_time),
        "lookback_time_points": int(lookback_time_points),
        "fit_rows": int(fit_mask.sum()),
        "cal_rows": int(cal_mask.sum()),
    }
    return frame, info, fit_mask, cal_mask


def split_inner_screen(frame: pd.DataFrame, fit_mask: np.ndarray, screen_time_points: int) -> tuple[np.ndarray, np.ndarray]:
    """在 fit 窗口内部再做早/晚切分，用来筛因子，避免偷看最终 calibration。"""
    fit_times = np.unique(frame.loc[fit_mask, "time_id"].to_numpy(dtype=np.int64))
    if len(fit_times) <= int(screen_time_points):
        split_time = fit_times[max(1, len(fit_times) // 2)]
    else:
        split_time = fit_times[-int(screen_time_points)]
    time_values = frame["time_id"].to_numpy(dtype=np.int64)
    screen_fit = fit_mask & (time_values < int(split_time))
    screen_eval = fit_mask & (time_values >= int(split_time))
    if not np.any(screen_fit) or not np.any(screen_eval):
        raise ValueError("inner_screen 切分后没有足够样本")
    return screen_fit, screen_eval


def screen_feature_ranking(
    frame: pd.DataFrame,
    feature_columns: list[str],
    fit_mask: np.ndarray,
    screen_time_points: int,
) -> pd.DataFrame:
    """用单因子加权线性模型筛选特征；只使用 fit 窗口内部数据。"""
    screen_fit, screen_eval = split_inner_screen(frame, fit_mask, screen_time_points)
    y_fit = frame.loc[screen_fit, "target"].to_numpy(dtype=np.float64)
    y_eval = frame.loc[screen_eval, "target"].to_numpy(dtype=np.float64)
    w_fit = frame.loc[screen_fit, "weight"].to_numpy(dtype=np.float64)
    w_eval = frame.loc[screen_eval, "weight"].to_numpy(dtype=np.float64)
    train_weight = w_fit / max(float(np.mean(w_fit)), 1e-12)
    rows = []

    for index, feature_name in enumerate(feature_columns):
        train_values = frame.loc[screen_fit, feature_name].to_numpy(dtype=np.float64)
        eval_values = frame.loc[screen_eval, feature_name].to_numpy(dtype=np.float64)
        mean = float(np.nanmean(train_values))
        scale = float(np.nanstd(train_values))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        train_x = np.nan_to_num((train_values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
        eval_x = np.nan_to_num((eval_values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
        denominator = float(np.sum(train_weight * train_x * train_x))
        coef = 0.0 if denominator <= 1e-18 else float(np.sum(train_weight * train_x * y_fit) / denominator)
        raw_pred = coef * eval_x
        # 单因子筛选也做 shrink，避免某些偶然幅度很大的因子被高估。
        shrink_info = calibrate_shrink_info(y_eval, raw_pred, w_eval, np.zeros(len(y_eval)), "global", 1.2)
        pred = apply_shrink(raw_pred, np.zeros(len(y_eval)), shrink_info)
        rows.append(
            {
                "feature_name": feature_name,
                "screen_score": weighted_zero_mean_r2(y_eval, pred, w_eval),
                "screen_raw_score": weighted_zero_mean_r2(y_eval, raw_pred, w_eval),
                "screen_coef": coef,
                "screen_shrink": float(shrink_info["global"]),
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(feature_columns):
            print(f"inner_screen screened {index + 1}/{len(feature_columns)} features")

    ranking = pd.DataFrame(rows).sort_values("screen_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def hybrid_feature_ranking(
    stable_ranking: pd.DataFrame,
    screen_ranking: pd.DataFrame,
    stable_weight: float,
) -> pd.DataFrame:
    """把跨 fold 稳定排名和当前窗口内部筛选排名合并，兼顾稳和新。"""
    stable = stable_ranking[["feature_name", "rank"]].rename(columns={"rank": "stable_rank"})
    screen = screen_ranking[["feature_name", "rank", "screen_score"]].rename(columns={"rank": "screen_rank"})
    merged = stable.merge(screen, on="feature_name", how="outer")
    fill_rank = int(max(merged["stable_rank"].max(), merged["screen_rank"].max()) + 100)
    merged["stable_rank"] = merged["stable_rank"].fillna(fill_rank)
    merged["screen_rank"] = merged["screen_rank"].fillna(fill_rank)
    merged["hybrid_rank_score"] = (
        float(stable_weight) * merged["stable_rank"]
        + (1.0 - float(stable_weight)) * merged["screen_rank"]
    )
    merged = merged.sort_values(["hybrid_rank_score", "screen_score"], ascending=[True, False]).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)
    return merged[["rank", "feature_name", "stable_rank", "screen_rank", "hybrid_rank_score", "screen_score"]]


def fit_ridge(
    train_x: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    alpha: float,
) -> Ridge:
    sample_weight = w_train / max(float(np.mean(w_train)), 1e-12)
    model = Ridge(alpha=float(alpha), solver="lsqr", max_iter=500)
    model.fit(train_x, y_train, sample_weight=sample_weight)
    return model


def fit_predict_lgbm_residual(
    train_x: np.ndarray,
    residual_train: np.ndarray,
    w_train: np.ndarray,
    cal_x: np.ndarray,
    feature_names: list[str],
    num_leaves: int,
    estimators: int,
    learning_rate: float,
    min_child_samples: int,
    reg_lambda: float,
    seeds: list[int],
    subsample: float,
    colsample_bytree: float,
    n_jobs: int,
) -> np.ndarray:
    sample_weight = w_train / max(float(np.mean(w_train)), 1e-12)
    train_frame = pd.DataFrame(train_x, columns=feature_names)
    cal_frame = pd.DataFrame(cal_x, columns=feature_names)
    predictions = []
    for seed in seeds:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=int(estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            min_child_samples=int(min_child_samples),
            subsample=float(subsample),
            subsample_freq=1,
            colsample_bytree=float(colsample_bytree),
            reg_lambda=float(reg_lambda),
            random_state=int(seed),
            n_jobs=int(n_jobs),
            verbose=-1,
        )
        model.fit(train_frame, residual_train, sample_weight=sample_weight)
        predictions.append(model.predict(cal_frame))
    return np.mean(np.vstack(predictions), axis=0)


def fit_predict_xgb_residual(
    train_x: np.ndarray,
    residual_train: np.ndarray,
    w_train: np.ndarray,
    cal_x: np.ndarray,
    estimators: int,
    learning_rate: float,
    max_depth: int,
    min_child_weight: float,
    reg_lambda: float,
    seeds: list[int],
    subsample: float,
    colsample_bytree: float,
    device: str,
    tree_method: str,
    max_bin: int,
    n_jobs: int,
) -> np.ndarray:
    """用 XGBoost 预测 Ridge 残差；device=cuda 时主要训练计算会走 GPU。"""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "当前环境没有 xgboost。请先运行：bash scripts/setup_wsl_env.sh"
        ) from exc

    sample_weight = w_train / max(float(np.mean(w_train)), 1e-12)
    predictions = []
    for seed in seeds:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            min_child_weight=float(min_child_weight),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_lambda=float(reg_lambda),
            tree_method=str(tree_method),
            device=str(device),
            max_bin=int(max_bin),
            random_state=int(seed),
            n_jobs=int(n_jobs),
            verbosity=0,
        )
        model.fit(train_x, residual_train, sample_weight=sample_weight)
        predictions.append(model.predict(cal_x))
    return np.mean(np.vstack(predictions), axis=0)


def fit_predict_catboost_residual(
    train_x: np.ndarray,
    residual_train: np.ndarray,
    w_train: np.ndarray,
    cal_x: np.ndarray,
    iterations: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
    random_strength: float,
    seeds: list[int],
    task_type: str,
) -> np.ndarray:
    """用 CatBoost 预测 Ridge 残差；task_type=GPU 时训练走显卡。"""
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("当前环境没有 catboost。") from exc

    sample_weight = w_train / max(float(np.mean(w_train)), 1e-12)
    predictions = []
    for seed in seeds:
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=int(iterations),
            learning_rate=float(learning_rate),
            depth=int(depth),
            l2_leaf_reg=float(l2_leaf_reg),
            random_strength=float(random_strength),
            task_type=str(task_type),
            devices="0" if str(task_type).upper() == "GPU" else None,
            random_seed=int(seed),
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(train_x, residual_train, sample_weight=sample_weight)
        predictions.append(model.predict(cal_x))
    return np.mean(np.vstack(predictions), axis=0)


def residual_param_grid(args: argparse.Namespace) -> list[dict]:
    """生成残差模型参数网格；LightGBM 和 XGBoost 使用不同复杂度参数。"""
    if args.residual_model == "lightgbm":
        rows = []
        for leaves, estimators, learning_rate, min_child, reg_lambda in itertools.product(
            args.lgbm_num_leaves_candidates,
            args.lgbm_estimators_candidates,
            args.lgbm_learning_rate_candidates,
            args.lgbm_min_child_samples_candidates,
            args.lgbm_reg_lambda_candidates,
        ):
            rows.append(
                {
                    "residual_model": "lightgbm",
                    "lgbm_num_leaves": int(leaves),
                    "lgbm_estimators": int(estimators),
                    "lgbm_learning_rate": float(learning_rate),
                    "lgbm_min_child_samples": int(min_child),
                    "lgbm_reg_lambda": float(reg_lambda),
                }
            )
        return rows

    if args.residual_model == "catboost":
        rows = []
        for depth, iterations, learning_rate, l2_leaf_reg in itertools.product(
            args.catboost_depth_candidates,
            args.catboost_iterations_candidates,
            args.catboost_learning_rate_candidates,
            args.catboost_l2_leaf_reg_candidates,
        ):
            rows.append(
                {
                    "residual_model": "catboost",
                    "lgbm_num_leaves": None,
                    "lgbm_estimators": None,
                    "lgbm_learning_rate": None,
                    "lgbm_min_child_samples": None,
                    "lgbm_reg_lambda": None,
                    "catboost_task_type": args.catboost_task_type,
                    "catboost_depth": int(depth),
                    "catboost_iterations": int(iterations),
                    "catboost_learning_rate": float(learning_rate),
                    "catboost_l2_leaf_reg": float(l2_leaf_reg),
                    "catboost_random_strength": float(args.catboost_random_strength),
                }
            )
        return rows

    rows = []
    for max_depth, estimators, learning_rate, min_child_weight, reg_lambda in itertools.product(
        args.xgb_max_depth_candidates,
        args.lgbm_estimators_candidates,
        args.lgbm_learning_rate_candidates,
        args.xgb_min_child_weight_candidates,
        args.xgb_reg_lambda_candidates,
    ):
        rows.append(
            {
                "residual_model": "xgboost",
                "lgbm_num_leaves": None,
                "lgbm_estimators": int(estimators),
                "lgbm_learning_rate": float(learning_rate),
                "lgbm_min_child_samples": None,
                "lgbm_reg_lambda": None,
                "xgb_max_depth": int(max_depth),
                "xgb_min_child_weight": float(min_child_weight),
                "xgb_reg_lambda": float(reg_lambda),
                "xgb_device": args.xgb_device,
                "xgb_tree_method": args.xgb_tree_method,
                "xgb_max_bin": int(args.xgb_max_bin),
            }
        )
    return rows


def fit_predict_residual_model(
    train_x: np.ndarray,
    residual_train: np.ndarray,
    w_train: np.ndarray,
    cal_x: np.ndarray,
    feature_names: list[str],
    params: dict,
    args: argparse.Namespace,
) -> np.ndarray:
    if params["residual_model"] == "lightgbm":
        return fit_predict_lgbm_residual(
            train_x,
            residual_train,
            w_train,
            cal_x,
            feature_names,
            int(params["lgbm_num_leaves"]),
            int(params["lgbm_estimators"]),
            float(params["lgbm_learning_rate"]),
            int(params["lgbm_min_child_samples"]),
            float(params["lgbm_reg_lambda"]),
            list(args.lgbm_seeds),
            float(args.lgbm_subsample),
            float(args.lgbm_colsample_bytree),
            int(args.lgbm_n_jobs),
        )
    if params["residual_model"] == "catboost":
        return fit_predict_catboost_residual(
            train_x,
            residual_train,
            w_train,
            cal_x,
            int(params["catboost_iterations"]),
            float(params["catboost_learning_rate"]),
            int(params["catboost_depth"]),
            float(params["catboost_l2_leaf_reg"]),
            float(params["catboost_random_strength"]),
            list(args.lgbm_seeds),
            str(params["catboost_task_type"]),
        )
    return fit_predict_xgb_residual(
        train_x,
        residual_train,
        w_train,
        cal_x,
        int(params["lgbm_estimators"]),
        float(params["lgbm_learning_rate"]),
        int(params["xgb_max_depth"]),
        float(params["xgb_min_child_weight"]),
        float(params["xgb_reg_lambda"]),
        list(args.lgbm_seeds),
        float(args.lgbm_subsample),
        float(args.lgbm_colsample_bytree),
        str(params["xgb_device"]),
        str(params["xgb_tree_method"]),
        int(params["xgb_max_bin"]),
        int(args.lgbm_n_jobs),
    )


def search_residual_weight_and_shrink(
    y_cal: np.ndarray,
    w_cal: np.ndarray,
    asset_cal: np.ndarray,
    time_cal: np.ndarray,
    ridge_cal: np.ndarray,
    residual_cal: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    best = {"score": -np.inf}
    residual_weights = np.arange(
        args.residual_weight_min,
        args.residual_weight_max + 1e-12,
        args.residual_weight_step,
    )
    for residual_weight in residual_weights:
        raw_pred = ridge_cal + float(residual_weight) * residual_cal
        for cap in args.shrink_cap_candidates:
            shrink_info = calibrate_shrink_info(
                y_cal,
                raw_pred,
                w_cal,
                asset_cal,
                args.shrink_mode,
                float(cap),
            )
            pred = apply_shrink(raw_pred, asset_cal, shrink_info)
            score_info = score_candidate_on_calibration(
                y_cal,
                pred,
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
                    "prediction": pred,
                    "raw_prediction": raw_pred,
                }
    return best


def evaluate_one_ranking(
    frame: pd.DataFrame,
    ranking: pd.DataFrame,
    fit_mask: np.ndarray,
    cal_mask: np.ndarray,
    feature_mode: str,
    window_info: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict]:
    y_fit = frame.loc[fit_mask, "target"].to_numpy(dtype=np.float32)
    w_fit = frame.loc[fit_mask, "weight"].to_numpy(dtype=np.float32)
    y_cal = frame.loc[cal_mask, "target"].to_numpy(dtype=np.float32)
    w_cal = frame.loc[cal_mask, "weight"].to_numpy(dtype=np.float32)
    asset_cal = frame.loc[cal_mask, "asset_id"].to_numpy(dtype=np.int64)
    time_cal = frame.loc[cal_mask, "time_id"].to_numpy(dtype=np.int64)
    rows = []

    model_grid = residual_param_grid(args)

    for top_k in args.top_k_candidates:
        selected = ranking.head(int(top_k))["feature_name"].astype(str).tolist()
        fit_x_raw = frame.loc[fit_mask, selected].to_numpy(dtype=np.float32)
        cal_x_raw = frame.loc[cal_mask, selected].to_numpy(dtype=np.float32)
        fit_x, cal_x, _, _ = standardize(fit_x_raw, cal_x_raw)

        for ridge_alpha in args.ridge_alpha_candidates:
            ridge = fit_ridge(fit_x, y_fit, w_fit, float(ridge_alpha))
            ridge_fit = ridge.predict(fit_x)
            ridge_cal = ridge.predict(cal_x)
            residual_fit = y_fit - ridge_fit

            for model_params in model_grid:
                residual_cal = fit_predict_residual_model(
                    fit_x,
                    residual_fit,
                    w_fit,
                    cal_x,
                    selected,
                    model_params,
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
                pred = best["prediction"]
                raw_pred = best["raw_prediction"]
                score_info = best["score_info"]
                shrink_summary = best["shrink_summary"]
                prediction_file = None
                if args.save_calibration_predictions:
                    prediction_path = candidate_prediction_path(
                        output_dir,
                        feature_mode,
                        int(top_k),
                        float(ridge_alpha),
                        model_params,
                    )
                    save_calibration_predictions(
                        prediction_path,
                        y_cal,
                        w_cal,
                        asset_cal,
                        time_cal,
                        ridge_cal,
                        residual_cal,
                        raw_pred,
                        pred,
                    )
                    prediction_file = str(prediction_path)
                row = {
                    **window_info,
                    "feature_mode": feature_mode,
                    "top_k": int(top_k),
                    "ridge_alpha": float(ridge_alpha),
                    "residual_model": model_params["residual_model"],
                    "lgbm_num_leaves": model_params.get("lgbm_num_leaves"),
                    "lgbm_estimators": (
                        None if model_params.get("lgbm_estimators") is None else int(model_params["lgbm_estimators"])
                    ),
                    "lgbm_learning_rate": (
                        None
                        if model_params.get("lgbm_learning_rate") is None
                        else float(model_params["lgbm_learning_rate"])
                    ),
                    "lgbm_min_child_samples": model_params.get("lgbm_min_child_samples"),
                    "lgbm_reg_lambda": model_params.get("lgbm_reg_lambda"),
                    "lgbm_subsample": float(args.lgbm_subsample),
                    "lgbm_colsample_bytree": float(args.lgbm_colsample_bytree),
                    "lgbm_seeds": ",".join(str(seed) for seed in args.lgbm_seeds),
                    "xgb_max_depth": model_params.get("xgb_max_depth"),
                    "xgb_min_child_weight": model_params.get("xgb_min_child_weight"),
                    "xgb_reg_lambda": model_params.get("xgb_reg_lambda"),
                    "xgb_device": model_params.get("xgb_device"),
                    "xgb_tree_method": model_params.get("xgb_tree_method"),
                    "xgb_max_bin": model_params.get("xgb_max_bin"),
                    "catboost_task_type": model_params.get("catboost_task_type"),
                    "catboost_depth": model_params.get("catboost_depth"),
                    "catboost_iterations": model_params.get("catboost_iterations"),
                    "catboost_learning_rate": model_params.get("catboost_learning_rate"),
                    "catboost_l2_leaf_reg": model_params.get("catboost_l2_leaf_reg"),
                    "catboost_random_strength": model_params.get("catboost_random_strength"),
                    "residual_weight": float(best["residual_weight"]),
                    "shrink_mode": args.shrink_mode,
                    "shrink_cap": float(best["shrink_info"]["cap"]),
                    "shrink": float(shrink_summary["cal_shrink"]),
                    "shrink_min": float(shrink_summary["cal_shrink_min"]),
                    "shrink_mean": float(shrink_summary["cal_shrink_mean"]),
                    "shrink_max": float(shrink_summary["cal_shrink_max"]),
                    "cal_score": float(best["score"]),
                    "cal_full_score": float(score_info["full_score"]),
                    "cal_first_half_score": float(score_info["first_half_score"]),
                    "cal_second_half_score": float(score_info["second_half_score"]),
                    "ridge_only_raw_score": float(weighted_zero_mean_r2(y_cal, ridge_cal, w_cal)),
                    "residual_raw_score": float(weighted_zero_mean_r2(y_cal - ridge_cal, residual_cal, w_cal)),
                    "prediction_std": float(np.std(pred)),
                    "raw_prediction_std": float(np.std(raw_pred)),
                    "target_std": float(np.std(y_cal)),
                    "calibration_prediction_file": prediction_file,
                    "selected_features": json.dumps(selected, ensure_ascii=False),
                    "shrink_info": json.dumps(best["shrink_info"], ensure_ascii=False, default=json_default),
                }
                row.update(score_time_blocks(y_cal, pred, w_cal, time_cal, 4))
                row.update(score_time_blocks(y_cal, pred, w_cal, time_cal, 8))
                rows.append(row)
                print(
                    json.dumps(
                        {key: value for key, value in row.items() if key not in {"selected_features", "shrink_info"}},
                        ensure_ascii=False,
                        default=json_default,
                    )
                )
    return rows


def materialize_rankings(
    frame: pd.DataFrame,
    feature_columns: list[str],
    stable_ranking: pd.DataFrame,
    fit_mask: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    rankings: dict[str, pd.DataFrame] = {}
    if "stable" in args.feature_selection_modes:
        rankings["stable"] = stable_ranking.copy()

    needs_screen = any(mode in args.feature_selection_modes for mode in ("inner_screen", "hybrid"))
    screen_ranking = None
    if needs_screen:
        screen_ranking = screen_feature_ranking(frame, feature_columns, fit_mask, args.screen_time_points)
        screen_ranking.to_csv(output_dir / "inner_screen_feature_ranking.csv", index=False)

    if "inner_screen" in args.feature_selection_modes:
        rankings["inner_screen"] = screen_ranking.copy()
    if "hybrid" in args.feature_selection_modes:
        rankings["hybrid"] = hybrid_feature_ranking(stable_ranking, screen_ranking, args.hybrid_stable_weight)
        rankings["hybrid"].to_csv(output_dir / "hybrid_feature_ranking.csv", index=False)
    return rankings


def main() -> None:
    args = parse_args()
    args.results_dir = make_results_dir(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train_paths = parquet_paths(args.raw_data_dir, "train")
    train_min_time, train_max_time_available = time_range(train_paths)
    train_end_time = (
        min(train_max_time_available, int(args.max_train_time_id))
        if args.max_train_time_id is not None
        else train_max_time_available
    )
    available_columns = schema_columns(train_paths)
    feature_columns = [column for column in available_columns if column.startswith("feature_")]
    stable_ranking = load_feature_ranking(args.fixed_features_file, available_columns)

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
    (args.results_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    print(f"写入目录: {args.results_dir}")
    print(f"raw train time range: {train_min_time}..{train_max_time_available}; tuning train_end={train_end_time}")

    all_rows = []
    best_payload = None
    need_all_features = any(mode in args.feature_selection_modes for mode in ("inner_screen", "hybrid"))

    for lookback in args.lookback_time_points:
        window_dir = args.results_dir / f"lookback_{int(lookback)}"
        window_dir.mkdir(parents=True, exist_ok=True)

        if need_all_features:
            columns = BASE_COLUMNS_TRAIN + feature_columns
        else:
            max_top_k = max(args.top_k_candidates)
            columns = BASE_COLUMNS_TRAIN + stable_ranking.head(max_top_k)["feature_name"].astype(str).tolist()

        frame, window_info, fit_mask, cal_mask = load_window(
            train_paths,
            columns,
            train_min_time,
            train_end_time,
            int(lookback),
            int(args.cal_time_points),
        )
        print(json.dumps({"window": window_info, "rows_loaded": len(frame), "columns_loaded": len(columns)}, ensure_ascii=False))

        rankings = materialize_rankings(frame, feature_columns, stable_ranking, fit_mask, args, window_dir)
        for mode, ranking in rankings.items():
            ranking.to_csv(window_dir / f"{mode}_feature_ranking.csv", index=False)
            rows = evaluate_one_ranking(frame, ranking, fit_mask, cal_mask, mode, window_info, args, window_dir)
            all_rows.extend(rows)

            if rows:
                mode_best = max(rows, key=lambda row: float(row["cal_score"]))
                if best_payload is None or mode_best["cal_score"] > best_payload["cal_score"]:
                    best_payload = mode_best

        del frame
        gc.collect()

    if not all_rows:
        raise ValueError("没有评估任何候选")

    result_frame = pd.DataFrame(all_rows).sort_values("cal_score", ascending=False).reset_index(drop=True)
    result_frame.to_csv(args.results_dir / "residual_candidate_metrics.csv", index=False)
    result_frame.head(50).to_csv(args.results_dir / "residual_top50_candidates.csv", index=False)

    best = result_frame.iloc[0].to_dict()
    (args.results_dir / "best_candidate.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    summary = {
        "leakage_safe": True,
        "official_test_used": False,
        "candidate_count": int(len(result_frame)),
        "best_score": float(best["cal_score"]),
        "best_candidate": best,
        "output_files": {
            "candidate_metrics": str(args.results_dir / "residual_candidate_metrics.csv"),
            "top50": str(args.results_dir / "residual_top50_candidates.csv"),
            "best_candidate": str(args.results_dir / "best_candidate.json"),
        },
        "score_target_note": (
            "0.1 是非常激进的目标；如果内部验证突然接近 0.1，需要优先排查泄漏和过拟合。"
        ),
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
