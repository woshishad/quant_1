from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from walk_forward_tabular import weighted_zero_mean_r2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用连续优化替代离散网格，为每个 asset 搜索组件融合权重和 shrink。"
    )
    parser.add_argument("--prediction-files", type=Path, nargs="+", required=True)
    parser.add_argument("--prediction-names", type=str, nargs="+", required=True)
    parser.add_argument("--prediction-columns", type=str, nargs="+", default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[0.8, 1.0, 1.2, 1.4])
    parser.add_argument("--min-weight", type=str, nargs="*", default=[])
    parser.add_argument("--max-weight", type=str, nargs="*", default=[])
    parser.add_argument("--start-param-files", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--selection-mode",
        choices=[
            "full",
            "mean_halves",
            "min_halves",
            "min_blocks4",
            "min_blocks8",
            "robust_blocks4",
            "robust_blocks8",
            "min_halves_blocks8",
        ],
        default="full",
    )
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--ftol", type=float, default=1e-10)
    parser.add_argument(
        "--l2-reference-param-file",
        type=Path,
        default=None,
        help="可选：把连续优化权重轻微拉回某个已有 per-asset 参数文件，降低过拟合风险。",
    )
    parser.add_argument("--l2-penalty", type=float, default=0.0)
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
    return Path("results") / f"per_asset_continuous_blend_{timestamp}"


def parse_weight_bounds(items: list[str], names: list[str], default: float) -> dict[str, float]:
    bounds = {name: float(default) for name in names}
    for item in items:
        if "=" not in item:
            raise ValueError(f"权重约束必须是 name=value 格式: {item}")
        name, value = item.split("=", 1)
        if name not in bounds:
            raise ValueError(f"未知组件名 {name}，可选值为 {names}")
        bounds[name] = float(value)
    return bounds


def load_prediction_file(path: Path, name: str, prediction_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["time_id", "asset_id", "target", "weight", prediction_column])
    return frame.rename(columns={prediction_column: f"prediction_{name}"}).sort_values(
        ["time_id", "asset_id"],
        kind="mergesort",
    )


def merge_prediction_files(files: list[Path], names: list[str], columns: list[str]) -> pd.DataFrame:
    """对齐多个组件预测，并校验每个组件面对的是同一批 calibration 样本。"""
    merged = None
    for path, name, column in zip(files, names, columns):
        frame = load_prediction_file(path, name, column)
        if merged is None:
            merged = frame
            continue
        suffix = f"_{name}"
        merged = merged.merge(frame, on=["time_id", "asset_id"], how="inner", suffixes=("", suffix))
        target_diff = float(np.max(np.abs(merged["target"] - merged[f"target{suffix}"])))
        weight_diff = float(np.max(np.abs(merged["weight"] - merged[f"weight{suffix}"])))
        if target_diff > 1e-6 or weight_diff > 1e-6:
            raise ValueError(f"{name} 与第一个预测文件的 target/weight 不一致")
        merged = merged.drop(columns=[f"target{suffix}", f"weight{suffix}"])
    if merged is None or merged.empty:
        raise ValueError("没有读到可融合的 calibration 预测")
    return merged


def optimal_shrink(y_true: np.ndarray, raw_prediction: np.ndarray, weight: np.ndarray, cap: float) -> float:
    denominator = float(np.sum(weight * raw_prediction * raw_prediction))
    if denominator <= 1e-18:
        return 0.0
    value = float(np.sum(weight * y_true * raw_prediction) / denominator)
    return min(float(cap), max(0.0, value))


def score_halves(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
) -> tuple[float, float, float]:
    unique_times = np.unique(time_id)
    split_time = unique_times[len(unique_times) // 2]
    first_mask = time_id < int(split_time)
    second_mask = ~first_mask
    full = weighted_zero_mean_r2(y_true, prediction, weight)
    first = weighted_zero_mean_r2(y_true[first_mask], prediction[first_mask], weight[first_mask])
    second = weighted_zero_mean_r2(y_true[second_mask], prediction[second_mask], weight[second_mask])
    return float(full), float(first), float(second)


def block_values(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    block_count: int,
) -> np.ndarray:
    values = []
    for chunk in np.array_split(np.unique(time_id), block_count):
        mask = (time_id >= int(chunk[0])) & (time_id <= int(chunk[-1]))
        values.append(float(weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])))
    return np.asarray(values, dtype=np.float64)


def objective_score(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    mode: str,
) -> tuple[float, dict[str, float]]:
    full, first, second = score_halves(y_true, prediction, weight, time_id)
    blocks4 = block_values(y_true, prediction, weight, time_id, 4)
    blocks8 = block_values(y_true, prediction, weight, time_id, 8)
    if mode == "full":
        selected = full
    elif mode == "mean_halves":
        selected = 0.5 * (first + second)
    elif mode == "min_halves":
        selected = min(first, second)
    elif mode == "min_blocks4":
        selected = float(np.min(blocks4))
    elif mode == "min_blocks8":
        selected = float(np.min(blocks8))
    elif mode == "robust_blocks4":
        selected = float(np.mean(blocks4) - np.std(blocks4))
    elif mode == "robust_blocks8":
        selected = float(np.mean(blocks8) - np.std(blocks8))
    elif mode == "min_halves_blocks8":
        selected = min(first, second, float(np.min(blocks8)))
    else:
        raise ValueError(f"未知 selection-mode: {mode}")
    return float(selected), {
        "full_score": float(full),
        "first_half_score": float(first),
        "second_half_score": float(second),
        "block4_mean": float(np.mean(blocks4)),
        "block4_min": float(np.min(blocks4)),
        "block8_mean": float(np.mean(blocks8)),
        "block8_min": float(np.min(blocks8)),
    }


def repair_weights(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """把起点修正到带上下界的 simplex 上，保证 SLSQP 从可行点开始。"""
    values = np.clip(np.asarray(values, dtype=np.float64), lower, upper)
    for _ in range(100):
        diff = 1.0 - float(np.sum(values))
        if abs(diff) <= 1e-12:
            break
        if diff > 0:
            room = np.maximum(upper - values, 0.0)
            total_room = float(np.sum(room))
            if total_room <= 1e-12:
                break
            values = np.minimum(upper, values + diff * room / total_room)
        else:
            room = np.maximum(values - lower, 0.0)
            total_room = float(np.sum(room))
            if total_room <= 1e-12:
                break
            values = np.maximum(lower, values + diff * room / total_room)
    if not np.isclose(float(np.sum(values)), 1.0, atol=1e-8):
        raise ValueError("无法构造满足上下界且和为 1 的权重起点")
    return values


def default_start(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    values = lower.copy()
    remaining = 1.0 - float(np.sum(values))
    room = upper - lower
    if remaining < -1e-12 or remaining - float(np.sum(room)) > 1e-12:
        raise ValueError("权重上下界不可行：min 之和不能超过 1，max 之和不能小于 1")
    if remaining > 0:
        values += remaining * room / max(float(np.sum(room)), 1e-12)
    return repair_weights(values, lower, upper)


def starts_from_param_files(
    paths: list[Path],
    asset_id: int,
    names: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[np.ndarray]:
    starts = []
    for path in paths:
        params = pd.read_csv(path)
        row = params.loc[params["asset_id"].astype(int) == int(asset_id)]
        if len(row) != 1:
            continue
        weights = np.asarray([float(row.iloc[0][f"weight_{name}"]) for name in names], dtype=np.float64)
        starts.append(repair_weights(weights, lower, upper))
    return starts


def reference_weights_from_file(
    path: Path | None,
    asset_id: int,
    names: list[str],
) -> np.ndarray | None:
    if path is None:
        return None
    params = pd.read_csv(path)
    row = params.loc[params["asset_id"].astype(int) == int(asset_id)]
    if len(row) != 1:
        return None
    return np.asarray([float(row.iloc[0][f"weight_{name}"]) for name in names], dtype=np.float64)


def optimize_asset(
    y: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    matrix: np.ndarray,
    asset_id: int,
    names: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    base_start = default_start(lower, upper)
    starts = [base_start]
    starts.extend(starts_from_param_files(args.start_param_files, asset_id, names, lower, upper))
    starts.append(repair_weights(np.full(len(names), 1.0 / len(names), dtype=np.float64), lower, upper))

    reference = reference_weights_from_file(args.l2_reference_param_file, asset_id, names)
    best: dict | None = None

    for cap in args.shrink_cap_candidates:
        for start in starts:
            def loss(values: np.ndarray) -> float:
                raw = matrix @ values
                shrink = optimal_shrink(y, raw, weight, float(cap))
                prediction = shrink * raw
                score, _ = objective_score(y, prediction, weight, time_id, args.selection_mode)
                penalty = 0.0
                if reference is not None and args.l2_penalty > 0:
                    penalty = float(args.l2_penalty) * float(np.sum((values - reference) ** 2))
                return -float(score) + penalty

            result = minimize(
                loss,
                start,
                method="SLSQP",
                bounds=list(zip(lower, upper)),
                constraints={"type": "eq", "fun": lambda values: float(np.sum(values) - 1.0)},
                options={"maxiter": int(args.max_iter), "ftol": float(args.ftol), "disp": False},
            )
            weights = repair_weights(result.x if result.success else start, lower, upper)
            raw = matrix @ weights
            shrink = optimal_shrink(y, raw, weight, float(cap))
            prediction = shrink * raw
            score, score_info = objective_score(y, prediction, weight, time_id, args.selection_mode)
            row = {
                "asset_id": int(asset_id),
                "selection_score": float(score),
                "shrink_cap": float(cap),
                "shrink": float(shrink),
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                **score_info,
                **{f"weight_{name}": float(value) for name, value in zip(names, weights)},
                "_prediction": prediction,
            }
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row

    if best is None:
        raise ValueError(f"asset={asset_id} 没有优化结果")
    return best


def summarize_blocks(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    block_count: int,
) -> dict[str, float]:
    values = block_values(y_true, prediction, weight, time_id, block_count)
    return {
        f"block{block_count}_mean": float(np.mean(values)),
        f"block{block_count}_min": float(np.min(values)),
        f"block{block_count}_negative_count": int(np.sum(values < 0.0)),
    }


def main() -> None:
    args = parse_args()
    if len(args.prediction_files) != len(args.prediction_names):
        raise ValueError("--prediction-files 和 --prediction-names 数量必须一致")
    prediction_columns = args.prediction_columns or ["prediction"] * len(args.prediction_files)
    if len(prediction_columns) != len(args.prediction_files):
        raise ValueError("--prediction-columns 数量必须和 --prediction-files 一致")

    args.results_dir = make_results_dir(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    min_bounds = parse_weight_bounds(args.min_weight, args.prediction_names, 0.0)
    max_bounds = parse_weight_bounds(args.max_weight, args.prediction_names, 1.0)
    lower = np.asarray([min_bounds[name] for name in args.prediction_names], dtype=np.float64)
    upper = np.asarray([max_bounds[name] for name in args.prediction_names], dtype=np.float64)
    _ = default_start(lower, upper)

    merged = merge_prediction_files(args.prediction_files, args.prediction_names, prediction_columns)
    y_all = merged["target"].to_numpy(dtype=np.float64)
    w_all = merged["weight"].to_numpy(dtype=np.float64)
    t_all = merged["time_id"].to_numpy(dtype=np.int64)
    a_all = merged["asset_id"].to_numpy(dtype=np.int64)
    pred_all = np.zeros(len(merged), dtype=np.float64)

    asset_rows = []
    for asset_id in sorted(np.unique(a_all).astype(int).tolist()):
        mask = a_all == asset_id
        matrix = np.column_stack(
            [
                merged.loc[mask, f"prediction_{name}"].to_numpy(dtype=np.float64)
                for name in args.prediction_names
            ]
        )
        best = optimize_asset(
            y_all[mask],
            w_all[mask],
            t_all[mask],
            matrix,
            int(asset_id),
            args.prediction_names,
            lower,
            upper,
            args,
        )
        pred_all[mask] = best["_prediction"]
        asset_rows.append({key: value for key, value in best.items() if not key.startswith("_")})
        print(json.dumps(asset_rows[-1], ensure_ascii=False, default=json_default))

    asset_frame = pd.DataFrame(asset_rows).sort_values("asset_id").reset_index(drop=True)
    asset_frame.to_csv(args.results_dir / "per_asset_blend_params.csv", index=False)

    output = merged.copy()
    output["prediction"] = pred_all.astype(np.float32)
    output["error"] = output["prediction"] - output["target"]
    output.to_csv(args.results_dir / "per_asset_blend_calibration_predictions.csv", index=False)

    full, first, second = score_halves(y_all, pred_all, w_all, t_all)
    summary = {
        "leakage_safe": True,
        "official_test_used": False,
        "method": "continuous constrained per-asset calibration blend",
        "row_count": int(len(merged)),
        "time_min": int(np.min(t_all)),
        "time_max": int(np.max(t_all)),
        "component_names": args.prediction_names,
        "prediction_columns": prediction_columns,
        "min_weight": min_bounds,
        "max_weight": max_bounds,
        "selection_mode": args.selection_mode,
        "shrink_cap_candidates": args.shrink_cap_candidates,
        "start_param_files": [str(path) for path in args.start_param_files],
        "l2_reference_param_file": None if args.l2_reference_param_file is None else str(args.l2_reference_param_file),
        "l2_penalty": float(args.l2_penalty),
        "full_score": float(full),
        "first_half_score": float(first),
        "second_half_score": float(second),
        "min_halves_score": float(min(first, second)),
        **summarize_blocks(y_all, pred_all, w_all, t_all, 4),
        **summarize_blocks(y_all, pred_all, w_all, t_all, 8),
        "asset_min_selection_score": float(asset_frame["selection_score"].min()),
        "asset_mean_selection_score": float(asset_frame["selection_score"].mean()),
        "optimizer_success_count": int(asset_frame["optimizer_success"].sum()),
        "output_files": {
            "params": str(args.results_dir / "per_asset_blend_params.csv"),
            "calibration_predictions": str(args.results_dir / "per_asset_blend_calibration_predictions.csv"),
        },
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
