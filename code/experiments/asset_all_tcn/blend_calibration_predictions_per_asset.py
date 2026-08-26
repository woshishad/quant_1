from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from walk_forward_tabular import weighted_zero_mean_r2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 calibration 上按 asset_id 分别搜索融合权重和 shrink。"
    )
    parser.add_argument("--prediction-files", type=Path, nargs="+", required=True)
    parser.add_argument("--prediction-names", type=str, nargs="+", required=True)
    parser.add_argument("--prediction-columns", type=str, nargs="+", default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[0.8, 1.0, 1.2, 1.4])
    parser.add_argument(
        "--min-weight",
        type=str,
        nargs="*",
        default=[],
        help="按 name=value 指定某个组件的最小权重，例如 final120k=0.25。",
    )
    parser.add_argument(
        "--max-weight",
        type=str,
        nargs="*",
        default=[],
        help="按 name=value 指定某个组件的最大权重，例如 residual60k=0.60。",
    )
    parser.add_argument(
        "--selection-mode",
        choices=[
            "min_halves",
            "mean_halves",
            "full",
            "min_blocks4",
            "min_blocks8",
            "robust_blocks4",
            "robust_blocks8",
            "min_halves_blocks8",
        ],
        default="min_halves",
        help="每个 asset 选择参数时使用的分数。默认更保守，看两个半段的较小值。",
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
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def make_results_dir(path: Path | None) -> Path:
    if path is not None:
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"per_asset_calibration_blend_{timestamp}"


def parse_weight_bounds(items: list[str], names: list[str], default: float) -> dict[str, float]:
    """解析 name=value 形式的权重上下限，并给未指定组件填默认值。"""
    bounds = {name: float(default) for name in names}
    for item in items:
        if "=" not in item:
            raise ValueError(f"权重约束必须是 name=value 格式: {item}")
        name, value = item.split("=", 1)
        if name not in bounds:
            raise ValueError(f"未知组件名 {name}，可选值为 {names}")
        bounds[name] = float(value)
    return bounds


def simplex_weights(count: int, step: float) -> list[tuple[float, ...]]:
    """生成非负且和为 1 的权重网格。组件数不大时，这种穷举最透明。"""
    units = int(round(1.0 / float(step)))
    if not np.isclose(units * float(step), 1.0):
        raise ValueError("--weight-step 必须能整除 1，例如 0.1/0.05/0.025")

    def recurse(prefix: list[int], remaining: int, slots: int):
        if slots == 1:
            yield prefix + [remaining]
            return
        for value in range(remaining + 1):
            yield from recurse(prefix + [value], remaining - value, slots - 1)

    return [tuple(float(value) / float(units) for value in row) for row in recurse([], units, count)]


def load_prediction_file(path: Path, name: str, prediction_column: str) -> pd.DataFrame:
    """读取单个 calibration 预测文件，统一预测列名。"""
    frame = pd.read_csv(path, usecols=["time_id", "asset_id", "target", "weight", prediction_column])
    return frame.rename(columns={prediction_column: f"prediction_{name}"}).sort_values(
        ["time_id", "asset_id"],
        kind="mergesort",
    )


def merge_prediction_files(files: list[Path], names: list[str], columns: list[str]) -> pd.DataFrame:
    """按 time_id/asset_id 对齐多个 calibration 文件，并校验 target/weight 一致。"""
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
    """在给定 raw prediction 下求加权最优 shrink，并裁剪到 [0, cap]。"""
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
    """返回全段、前半段、后半段 weighted zero-mean R2。"""
    unique_times = np.unique(time_id)
    split_time = unique_times[len(unique_times) // 2]
    first_mask = time_id < int(split_time)
    second_mask = ~first_mask
    full = weighted_zero_mean_r2(y_true, prediction, weight)
    first = weighted_zero_mean_r2(y_true[first_mask], prediction[first_mask], weight[first_mask])
    second = weighted_zero_mean_r2(y_true[second_mask], prediction[second_mask], weight[second_mask])
    return float(full), float(first), float(second)


def time_block_score_values(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    block_count: int,
) -> np.ndarray:
    """把 calibration 切成多个连续时间块，返回每个时间块的 weighted zero-mean R2。"""
    scores = []
    for chunk in np.array_split(np.unique(time_id), block_count):
        mask = (time_id >= int(chunk[0])) & (time_id <= int(chunk[-1]))
        scores.append(float(weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])))
    return np.asarray(scores, dtype=np.float64)


def selection_score(
    full: float,
    first: float,
    second: float,
    mode: str,
    block_scores: np.ndarray | None = None,
) -> float:
    """把候选融合参数压成一个选择分数；block 模式更重视跨时间稳定性。"""
    if mode == "full":
        return float(full)
    if mode == "mean_halves":
        return float((first + second) / 2.0)
    if mode.startswith("min_blocks"):
        if block_scores is None:
            raise ValueError(f"{mode} requires block_scores")
        return float(np.min(block_scores))
    if mode.startswith("robust_blocks"):
        if block_scores is None:
            raise ValueError(f"{mode} requires block_scores")
        # 均值减标准差会压制只在个别时间块特别好的权重组合。
        return float(np.mean(block_scores) - np.std(block_scores))
    if mode == "min_halves_blocks8":
        if block_scores is None:
            raise ValueError(f"{mode} requires block_scores")
        return float(min(first, second, float(np.min(block_scores))))
    return float(min(first, second))


def required_block_count(selection_mode: str) -> int | None:
    """根据选择目标判断是否需要额外计算时间块分数。"""
    if selection_mode.endswith("blocks4"):
        return 4
    if selection_mode.endswith("blocks8"):
        return 8
    return None


def score_time_blocks(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    block_count: int,
) -> dict[str, float]:
    """按时间块统计稳定性。"""
    scores = []
    for chunk in np.array_split(np.unique(time_id), block_count):
        mask = (time_id >= int(chunk[0])) & (time_id <= int(chunk[-1]))
        scores.append(float(weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])))
    values = np.asarray(scores, dtype=np.float64)
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
    weight_grid = []
    for weights in simplex_weights(len(args.prediction_names), args.weight_step):
        if all(
            min_bounds[name] - 1e-12 <= weight <= max_bounds[name] + 1e-12
            for name, weight in zip(args.prediction_names, weights)
        ):
            weight_grid.append(weights)
    if not weight_grid:
        raise ValueError("权重约束过严，没有任何可用权重组合")

    merged = merge_prediction_files(args.prediction_files, args.prediction_names, prediction_columns)
    y_all = merged["target"].to_numpy(dtype=np.float64)
    w_all = merged["weight"].to_numpy(dtype=np.float64)
    t_all = merged["time_id"].to_numpy(dtype=np.int64)
    a_all = merged["asset_id"].to_numpy(dtype=np.int64)
    pred_all = np.zeros(len(merged), dtype=np.float64)

    asset_rows = []
    for asset_id in sorted(np.unique(a_all).astype(int).tolist()):
        mask = a_all == asset_id
        y = y_all[mask]
        w = w_all[mask]
        t = t_all[mask]
        matrix = np.column_stack(
            [
                merged.loc[mask, f"prediction_{name}"].to_numpy(dtype=np.float64)
                for name in args.prediction_names
            ]
        )
        best = {"selection_score": -np.inf}
        block_count = required_block_count(args.selection_mode)
        for weights in weight_grid:
            raw_prediction = matrix @ np.asarray(weights, dtype=np.float64)
            for cap in args.shrink_cap_candidates:
                shrink = optimal_shrink(y, raw_prediction, w, float(cap))
                prediction = shrink * raw_prediction
                full, first, second = score_halves(y, prediction, w, t)
                block_scores = None
                if block_count is not None:
                    block_scores = time_block_score_values(y, prediction, w, t, block_count)
                selected = selection_score(full, first, second, args.selection_mode, block_scores)
                if selected > best["selection_score"]:
                    best = {
                        "asset_id": int(asset_id),
                        "selection_score": float(selected),
                        "full_score": float(full),
                        "first_half_score": float(first),
                        "second_half_score": float(second),
                        "shrink_cap": float(cap),
                        "shrink": float(shrink),
                        **{
                            f"weight_{name}": float(weight)
                            for name, weight in zip(args.prediction_names, weights)
                        },
                        "_prediction": prediction,
                    }
        pred_all[mask] = best["_prediction"]
        asset_rows.append({key: value for key, value in best.items() if not key.startswith("_")})

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
        "method": "per-asset constrained calibration blend",
        "row_count": int(len(merged)),
        "time_min": int(np.min(t_all)),
        "time_max": int(np.max(t_all)),
        "component_names": args.prediction_names,
        "prediction_columns": prediction_columns,
        "weight_step": float(args.weight_step),
        "weight_grid_count": int(len(weight_grid)),
        "min_weight": min_bounds,
        "max_weight": max_bounds,
        "selection_mode": args.selection_mode,
        "full_score": float(full),
        "first_half_score": float(first),
        "second_half_score": float(second),
        "min_halves_score": float(min(first, second)),
        **score_time_blocks(y_all, pred_all, w_all, t_all, 4),
        **score_time_blocks(y_all, pred_all, w_all, t_all, 8),
        "asset_min_selection_score": float(asset_frame["selection_score"].min()),
        "asset_mean_selection_score": float(asset_frame["selection_score"].mean()),
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
