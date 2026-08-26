from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all-asset continuous time panel for model transfer testing.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "data/raw/public_release_20260630/public_release_20260630/data/train/train_partition_000.parquet"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/asset_all_time50000"))
    parser.add_argument("--time-start", type=int, default=0)
    parser.add_argument("--time-count", type=int, default=50_000)
    return parser.parse_args()


def summarize_panel(frame: pd.DataFrame, time_start: int, time_count: int) -> dict:
    # 全标的面板不强行假设每个 asset 都完整；真实数据里 asset 12 在前 50000 个时间点缺少少量行。
    # 因此这里记录缺失时间点，而不是直接报错。训练脚本会在构造滑窗时跳过含缺口的窗口。
    expected_times = set(range(time_start, time_start + time_count))
    feature_columns = [col for col in frame.columns if col.startswith("feature_")]
    asset_summary = {}
    for asset_id, asset_frame in frame.groupby("asset_id", sort=True):
        actual_times = set(asset_frame["time_id"].astype(int).tolist())
        missing_times = sorted(expected_times - actual_times)
        extra_times = sorted(actual_times - expected_times)
        asset_summary[str(int(asset_id))] = {
            "rows": int(len(asset_frame)),
            "unique_time_count": int(asset_frame["time_id"].nunique()),
            "min_time_id": int(asset_frame["time_id"].min()),
            "max_time_id": int(asset_frame["time_id"].max()),
            "missing_time_count": int(len(missing_times)),
            "missing_time_preview": missing_times[:20],
            "extra_time_count": int(len(extra_times)),
            "extra_time_preview": extra_times[:20],
        }

    per_time_asset_count = frame.groupby("time_id")["asset_id"].nunique()
    return {
        "rows": int(len(frame)),
        "time_start": int(time_start),
        "time_end": int(time_start + time_count - 1),
        "time_count": int(frame["time_id"].nunique()),
        "asset_ids": [int(asset_id) for asset_id in sorted(frame["asset_id"].unique().tolist())],
        "asset_count": int(frame["asset_id"].nunique()),
        "feature_count": int(len(feature_columns)),
        "min_assets_per_time": int(per_time_asset_count.min()),
        "max_assets_per_time": int(per_time_asset_count.max()),
        "incomplete_time_count": int((per_time_asset_count != per_time_asset_count.max()).sum()),
        "asset_summary": asset_summary,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    end_time = args.time_start + args.time_count

    # 只读取建模需要的列：feature_*、target、weight 和索引列。responder_* 暂时不参与训练，避免引入辅助目标。
    schema_columns = pq.ParquetFile(args.source).schema_arrow.names
    feature_columns = [col for col in schema_columns if col.startswith("feature_")]
    required_columns = ["row_id", "time_id", "asset_id", "weight", "target"]
    columns = required_columns + feature_columns

    frame = pd.read_parquet(
        args.source,
        columns=columns,
        filters=[("time_id", ">=", args.time_start), ("time_id", "<", end_time)],
    )
    frame = frame.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)

    missing_columns = set(required_columns) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("filtered dataset is empty")

    output_path = args.output_dir / "train.parquet"
    frame.to_parquet(output_path, index=False)

    manifest = {
        "description": f"all assets, time_id {args.time_start}..{end_time - 1} panel for all-asset weighted R2 transfer test",
        "source": str(args.source),
        "file": "train.parquet",
        **summarize_panel(frame, args.time_start, args.time_count),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
