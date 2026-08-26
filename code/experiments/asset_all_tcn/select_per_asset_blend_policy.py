from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from walk_forward_tabular import weighted_zero_mean_r2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在多个 per-asset 融合策略之间，为每个 asset 单独选择更稳的策略。"
    )
    parser.add_argument("--policy-prediction-files", type=Path, nargs="+", required=True)
    parser.add_argument("--policy-param-files", type=Path, nargs="+", required=True)
    parser.add_argument("--policy-names", type=str, nargs="+", required=True)
    parser.add_argument("--results-dir", type=Path, default=None)
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
        default="min_halves",
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
    return Path("results") / f"per_asset_policy_selection_{timestamp}"


def load_policy_prediction(path: Path, name: str) -> pd.DataFrame:
    """读取单个策略在 calibration 上的预测，并把 prediction 改成策略专属列名。"""
    frame = pd.read_csv(path, usecols=["time_id", "asset_id", "target", "weight", "prediction"])
    return frame.rename(columns={"prediction": f"prediction_{name}"}).sort_values(
        ["time_id", "asset_id"],
        kind="mergesort",
    )


def merge_policy_predictions(files: list[Path], names: list[str]) -> pd.DataFrame:
    """按 time_id/asset_id 对齐多个策略预测，同时校验 target/weight 没有错位。"""
    merged = None
    for path, name in zip(files, names):
        frame = load_policy_prediction(path, name)
        if merged is None:
            merged = frame
            continue
        suffix = f"_{name}"
        merged = merged.merge(frame, on=["time_id", "asset_id"], how="inner", suffixes=("", suffix))
        target_diff = float(np.max(np.abs(merged["target"] - merged[f"target{suffix}"])))
        weight_diff = float(np.max(np.abs(merged["weight"] - merged[f"weight{suffix}"])))
        if target_diff > 1e-6 or weight_diff > 1e-6:
            raise ValueError(f"{name} 与第一个策略文件的 target/weight 不一致")
        merged = merged.drop(columns=[f"target{suffix}", f"weight{suffix}"])
    if merged is None or merged.empty:
        raise ValueError("没有读到可选择的 per-asset 策略预测")
    return merged


def score_halves(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
) -> tuple[float, float, float]:
    """返回全段、前半段、后半段的 weighted zero-mean R2。"""
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
    """按连续时间块计算分数，用来惩罚局部时间段崩掉的策略。"""
    values = []
    for chunk in np.array_split(np.unique(time_id), block_count):
        mask = (time_id >= int(chunk[0])) & (time_id <= int(chunk[-1]))
        values.append(float(weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])))
    return np.asarray(values, dtype=np.float64)


def objective(
    full: float,
    first: float,
    second: float,
    blocks4: np.ndarray,
    blocks8: np.ndarray,
    mode: str,
) -> float:
    """把每个 asset 的策略表现压成一个选择分数。"""
    if mode == "full":
        return float(full)
    if mode == "mean_halves":
        return float((first + second) / 2.0)
    if mode == "min_halves":
        return float(min(first, second))
    if mode == "min_blocks4":
        return float(np.min(blocks4))
    if mode == "min_blocks8":
        return float(np.min(blocks8))
    if mode == "robust_blocks4":
        return float(np.mean(blocks4) - np.std(blocks4))
    if mode == "robust_blocks8":
        return float(np.mean(blocks8) - np.std(blocks8))
    if mode == "min_halves_blocks8":
        return float(min(first, second, float(np.min(blocks8))))
    raise ValueError(f"未知 selection-mode: {mode}")


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
    if not (len(args.policy_prediction_files) == len(args.policy_param_files) == len(args.policy_names)):
        raise ValueError("--policy-prediction-files/--policy-param-files/--policy-names 数量必须一致")

    args.results_dir = make_results_dir(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_policy_predictions(args.policy_prediction_files, args.policy_names)
    y_all = merged["target"].to_numpy(dtype=np.float64)
    w_all = merged["weight"].to_numpy(dtype=np.float64)
    t_all = merged["time_id"].to_numpy(dtype=np.int64)
    a_all = merged["asset_id"].to_numpy(dtype=np.int64)

    param_by_policy = {
        name: pd.read_csv(path).assign(source_policy=name)
        for name, path in zip(args.policy_names, args.policy_param_files)
    }

    pred_all = np.zeros(len(merged), dtype=np.float64)
    selected_rows = []
    score_rows = []
    selected_param_rows = []

    for asset_id in sorted(np.unique(a_all).astype(int).tolist()):
        mask = a_all == asset_id
        y = y_all[mask]
        w = w_all[mask]
        t = t_all[mask]
        best: dict | None = None

        for name in args.policy_names:
            prediction = merged.loc[mask, f"prediction_{name}"].to_numpy(dtype=np.float64)
            full, first, second = score_halves(y, prediction, w, t)
            blocks4 = block_values(y, prediction, w, t, 4)
            blocks8 = block_values(y, prediction, w, t, 8)
            selected = objective(full, first, second, blocks4, blocks8, args.selection_mode)
            row = {
                "asset_id": int(asset_id),
                "policy": name,
                "selection_score": float(selected),
                "full_score": float(full),
                "first_half_score": float(first),
                "second_half_score": float(second),
                "block4_min": float(np.min(blocks4)),
                "block8_min": float(np.min(blocks8)),
                "block8_mean": float(np.mean(blocks8)),
            }
            score_rows.append(row)
            if best is None or selected > best["selection_score"]:
                best = {**row, "_prediction": prediction}

        if best is None:
            raise ValueError(f"asset={asset_id} 没有可用策略")
        pred_all[mask] = best["_prediction"]
        selected_rows.append({key: value for key, value in best.items() if not key.startswith("_")})

        policy_params = param_by_policy[str(best["policy"])]
        selected_param = policy_params.loc[policy_params["asset_id"].astype(int) == int(asset_id)]
        if len(selected_param) != 1:
            raise ValueError(f"策略 {best['policy']} 中 asset={asset_id} 的参数行数不是 1")
        selected_param_rows.append(selected_param.iloc[0].to_dict())

    pd.DataFrame(score_rows).to_csv(args.results_dir / "policy_scores_by_asset.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.results_dir / "selected_policy_by_asset.csv", index=False)
    pd.DataFrame(selected_param_rows).sort_values("asset_id").to_csv(
        args.results_dir / "per_asset_blend_params.csv",
        index=False,
    )

    output = merged[["time_id", "asset_id", "target", "weight"]].copy()
    for name in args.policy_names:
        output[f"prediction_{name}"] = merged[f"prediction_{name}"].to_numpy(dtype=np.float32)
    output["prediction"] = pred_all.astype(np.float32)
    output["error"] = output["prediction"] - output["target"]
    output.to_csv(args.results_dir / "per_asset_blend_calibration_predictions.csv", index=False)

    full, first, second = score_halves(y_all, pred_all, w_all, t_all)
    summary = {
        "leakage_safe": True,
        "official_test_used": False,
        "method": "per-asset policy selection",
        "selection_mode": args.selection_mode,
        "policy_names": args.policy_names,
        "row_count": int(len(merged)),
        "time_min": int(np.min(t_all)),
        "time_max": int(np.max(t_all)),
        "full_score": float(full),
        "first_half_score": float(first),
        "second_half_score": float(second),
        "min_halves_score": float(min(first, second)),
        **summarize_blocks(y_all, pred_all, w_all, t_all, 4),
        **summarize_blocks(y_all, pred_all, w_all, t_all, 8),
        "selected_policy_counts": pd.Series([row["policy"] for row in selected_rows]).value_counts().to_dict(),
        "output_files": {
            "params": str(args.results_dir / "per_asset_blend_params.csv"),
            "selected_policy_by_asset": str(args.results_dir / "selected_policy_by_asset.csv"),
            "policy_scores_by_asset": str(args.results_dir / "policy_scores_by_asset.csv"),
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
