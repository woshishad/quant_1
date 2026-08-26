from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blend_predictions_from_calibration import (
    DEFAULT_SAMPLE_SUBMISSION,
    make_submission,
    numeric_grid,
    prediction_stats,
    score_by_asset,
    weighted_zero_mean_r2,
    write_zip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把单模型 prediction 做全局 shrink 稳定性诊断，并生成官方 test 提交。"
    )
    parser.add_argument("--calibration", type=Path, required=True, help="包含 target/weight/prediction 的 calibration 文件")
    parser.add_argument("--test", type=Path, required=True, help="包含 row_id/prediction 的官方 test 预测文件")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION)
    parser.add_argument("--prediction-column", type=str, default="prediction")
    parser.add_argument("--min-shrink", type=float, default=0.80)
    parser.add_argument("--max-shrink", type=float, default=1.80)
    parser.add_argument("--shrink-step", type=float, default=0.01)
    parser.add_argument("--block-count", type=int, default=4, help="按时间把 calibration 等分成几个连续块")
    parser.add_argument(
        "--selection-rule",
        choices=[
            "full_calibration_best",
            "holdout_mean_best",
            "holdout_recency_weighted_best",
            "last_block_best",
            "rolling_median",
        ],
        default="holdout_mean_best",
        help=(
            "full_calibration_best 直接用完整 calibration 最优 shrink；"
            "holdout_mean_best 用多个后续时间块的平均表现选 shrink；"
            "holdout_recency_weighted_best 对越新的 holdout 时间块给越高权重；"
            "last_block_best 只看最新 calibration 时间块；"
            "rolling_median 用滚动前缀拟合 shrink 的中位数，更保守。"
        ),
    )
    return parser.parse_args()


def read_calibration(path: Path, prediction_column: str) -> pd.DataFrame:
    needed = ["row_id", "time_id", "asset_id", "target", "weight", prediction_column]
    frame = pd.read_csv(path, usecols=needed)
    frame = frame.rename(columns={prediction_column: "raw_prediction"})
    return frame.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)


def read_test(path: Path, prediction_column: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    columns = set(header.columns)
    if prediction_column not in columns and "target" in columns:
        prediction_column = "target"
    needed = ["row_id", prediction_column]
    for optional in ["time_id", "asset_id"]:
        if optional in columns:
            needed.append(optional)
    frame = pd.read_csv(path, usecols=needed)
    return frame.rename(columns={prediction_column: "raw_prediction"})


def score_for_shrink(frame: pd.DataFrame, shrink: float) -> float:
    return weighted_zero_mean_r2(
        frame["target"].to_numpy(dtype=np.float64),
        shrink * frame["raw_prediction"].to_numpy(dtype=np.float64),
        frame["weight"].to_numpy(dtype=np.float64),
    )


def best_shrink(frame: pd.DataFrame, shrink_grid: np.ndarray) -> dict[str, float]:
    best = {"shrink": float(shrink_grid[0]), "score": -np.inf}
    for shrink in shrink_grid:
        score = score_for_shrink(frame, float(shrink))
        if score > best["score"]:
            best = {"shrink": float(shrink), "score": float(score)}
    return best


def split_time_blocks(frame: pd.DataFrame, block_count: int) -> list[pd.DataFrame]:
    if block_count < 2:
        raise ValueError("block-count 至少需要为 2")
    unique_times = np.sort(frame["time_id"].unique())
    if len(unique_times) < block_count:
        raise ValueError("calibration time_id 数量少于 block-count")
    time_blocks = np.array_split(unique_times, block_count)
    blocks = []
    for block_index, times in enumerate(time_blocks):
        block = frame[frame["time_id"].isin(times)].copy()
        block["block"] = block_index
        blocks.append(block)
    return blocks


def build_block_scores(blocks: list[pd.DataFrame], shrink_grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for block_index, block in enumerate(blocks):
        optimal = best_shrink(block, shrink_grid)
        rows.append(
            {
                "block": block_index,
                "time_min": int(block["time_id"].min()),
                "time_max": int(block["time_id"].max()),
                "rows": int(len(block)),
                "raw_score": score_for_shrink(block, 1.0),
                "best_shrink": optimal["shrink"],
                "best_score": optimal["score"],
            }
        )
    return pd.DataFrame(rows)


def build_rolling_scores(blocks: list[pd.DataFrame], shrink_grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for holdout_index in range(1, len(blocks)):
        fit_frame = pd.concat(blocks[:holdout_index], ignore_index=True)
        holdout_frame = blocks[holdout_index]
        fitted = best_shrink(fit_frame, shrink_grid)
        holdout_score = score_for_shrink(holdout_frame, fitted["shrink"])
        rows.append(
            {
                "fit_blocks": f"0..{holdout_index - 1}",
                "holdout_block": holdout_index,
                "fit_time_min": int(fit_frame["time_id"].min()),
                "fit_time_max": int(fit_frame["time_id"].max()),
                "holdout_time_min": int(holdout_frame["time_id"].min()),
                "holdout_time_max": int(holdout_frame["time_id"].max()),
                "fit_best_shrink": float(fitted["shrink"]),
                "fit_score": float(fitted["score"]),
                "holdout_score": float(holdout_score),
            }
        )
    return pd.DataFrame(rows)


def build_shrink_curve(blocks: list[pd.DataFrame], shrink_grid: np.ndarray) -> pd.DataFrame:
    rows = []
    all_frame = pd.concat(blocks, ignore_index=True)
    holdout_frame = pd.concat(blocks[1:], ignore_index=True)
    for shrink in shrink_grid:
        row = {
            "shrink": float(shrink),
            "full_calibration_score": score_for_shrink(all_frame, float(shrink)),
            "holdout_blocks_mean_score": float(np.mean([score_for_shrink(block, float(shrink)) for block in blocks[1:]])),
            "holdout_blocks_weighted_score": score_for_shrink(holdout_frame, float(shrink)),
        }
        for block_index, block in enumerate(blocks):
            row[f"block_{block_index}_score"] = score_for_shrink(block, float(shrink))
        rows.append(row)
    return pd.DataFrame(rows)


def select_shrink(
    selection_rule: str,
    full_best: dict[str, float],
    shrink_curve: pd.DataFrame,
    rolling_scores: pd.DataFrame,
) -> dict[str, float | str]:
    if selection_rule == "full_calibration_best":
        return {"selection_rule": selection_rule, "selected_shrink": float(full_best["shrink"])}
    if selection_rule == "holdout_mean_best":
        best_row = shrink_curve.sort_values("holdout_blocks_mean_score", ascending=False).iloc[0]
        return {
            "selection_rule": selection_rule,
            "selected_shrink": float(best_row["shrink"]),
            "selection_score": float(best_row["holdout_blocks_mean_score"]),
        }
    if selection_rule == "holdout_recency_weighted_best":
        block_columns = sorted(
            [column for column in shrink_curve.columns if column.startswith("block_") and column.endswith("_score")],
            key=lambda column: int(column.split("_")[1]),
        )
        holdout_columns = block_columns[1:]
        weights = np.arange(1, len(holdout_columns) + 1, dtype=np.float64)
        weighted_scores = []
        for _, row in shrink_curve.iterrows():
            values = row[holdout_columns].to_numpy(dtype=np.float64)
            weighted_scores.append(float(np.average(values, weights=weights)))
        curve = shrink_curve.copy()
        curve["holdout_recency_weighted_score"] = weighted_scores
        best_row = curve.sort_values("holdout_recency_weighted_score", ascending=False).iloc[0]
        return {
            "selection_rule": selection_rule,
            "selected_shrink": float(best_row["shrink"]),
            "selection_score": float(best_row["holdout_recency_weighted_score"]),
        }
    if selection_rule == "last_block_best":
        block_columns = sorted(
            [column for column in shrink_curve.columns if column.startswith("block_") and column.endswith("_score")],
            key=lambda column: int(column.split("_")[1]),
        )
        last_column = block_columns[-1]
        best_row = shrink_curve.sort_values(last_column, ascending=False).iloc[0]
        return {
            "selection_rule": selection_rule,
            "selected_shrink": float(best_row["shrink"]),
            "selection_score": float(best_row[last_column]),
            "selected_on": last_column,
        }
    if selection_rule == "rolling_median":
        shrink = float(np.median(rolling_scores["fit_best_shrink"].to_numpy(dtype=np.float64)))
        return {"selection_rule": selection_rule, "selected_shrink": shrink}
    raise ValueError(f"unknown selection rule: {selection_rule}")


def save_plots(results_dir: Path, block_scores: pd.DataFrame, rolling_scores: pd.DataFrame, shrink_curve: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(shrink_curve["shrink"], shrink_curve["full_calibration_score"], label="full calibration")
    plt.plot(shrink_curve["shrink"], shrink_curve["holdout_blocks_mean_score"], label="holdout blocks mean")
    plt.xlabel("shrink")
    plt.ylabel("weighted R2")
    plt.title("Global shrink score curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "shrink_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(block_scores["block"].astype(str), block_scores["best_shrink"])
    plt.xlabel("time block")
    plt.ylabel("best shrink")
    plt.title("Best shrink by calibration block")
    plt.tight_layout()
    plt.savefig(results_dir / "best_shrink_by_block.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rolling_scores["holdout_block"], rolling_scores["fit_score"], marker="o", label="fit prefix")
    plt.plot(rolling_scores["holdout_block"], rolling_scores["holdout_score"], marker="o", label="next block")
    plt.xlabel("holdout block")
    plt.ylabel("weighted R2")
    plt.title("Rolling prefix shrink validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "rolling_validation.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    shrink_grid = numeric_grid(args.min_shrink, args.max_shrink, args.shrink_step)
    calibration = read_calibration(args.calibration, args.prediction_column)
    test = read_test(args.test, args.prediction_column)
    blocks = split_time_blocks(calibration, args.block_count)

    block_scores = build_block_scores(blocks, shrink_grid)
    rolling_scores = build_rolling_scores(blocks, shrink_grid)
    shrink_curve = build_shrink_curve(blocks, shrink_grid)
    full_best = best_shrink(calibration, shrink_grid)
    selected = select_shrink(args.selection_rule, full_best, shrink_curve, rolling_scores)
    selected_shrink = float(selected["selected_shrink"])

    calibration_out = calibration.copy()
    calibration_out["shrink"] = selected_shrink
    calibration_out["prediction"] = selected_shrink * calibration_out["raw_prediction"]
    calibration_out["error"] = calibration_out["prediction"] - calibration_out["target"]

    test_out = test.copy()
    test_out["shrink"] = selected_shrink
    test_out["prediction"] = selected_shrink * test_out["raw_prediction"]

    submission = make_submission(test_out, args.sample_submission)
    submission_path = args.results_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    write_zip(submission_path, args.results_dir / "submission.zip")

    calibration_out.to_csv(args.results_dir / "calibration_predictions.csv", index=False)
    test_out.to_csv(args.results_dir / "final_test_predictions.csv", index=False)
    block_scores.to_csv(args.results_dir / "block_scores.csv", index=False, encoding="utf-8-sig")
    rolling_scores.to_csv(args.results_dir / "rolling_scores.csv", index=False, encoding="utf-8-sig")
    shrink_curve.to_csv(args.results_dir / "shrink_curve.csv", index=False, encoding="utf-8-sig")
    save_plots(args.results_dir, block_scores, rolling_scores, shrink_curve)

    row_order_matches_sample = None
    if args.sample_submission.exists():
        sample = pd.read_csv(args.sample_submission, usecols=["row_id"])
        row_order_matches_sample = bool(submission["row_id"].equals(sample["row_id"]))

    metrics = {
        "leakage_safe": True,
        "strategy": "single_model_global_shrink",
        "calibration_file": str(args.calibration),
        "test_file": str(args.test),
        "selection_rule": args.selection_rule,
        "selected_shrink": selected_shrink,
        "selected_info": selected,
        "full_calibration_best": full_best,
        "raw_full_calibration_score": score_for_shrink(calibration, 1.0),
        "selected_full_calibration_score": score_for_shrink(calibration, selected_shrink),
        "selected_full_calibration_score_by_asset": score_by_asset(calibration_out, "prediction"),
        "block_count": args.block_count,
        "block_scores": block_scores.to_dict(orient="records"),
        "rolling_scores": rolling_scores.to_dict(orient="records"),
        "test_prediction_stats": prediction_stats(test_out["prediction"]),
        "submission": {
            "path": str(submission_path),
            "zip_path": str(args.results_dir / "submission.zip"),
            "rows": int(len(submission)),
            "columns": submission.columns.tolist(),
            "row_order_matches_sample": row_order_matches_sample,
            "target_stats": prediction_stats(submission["target"]),
        },
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
