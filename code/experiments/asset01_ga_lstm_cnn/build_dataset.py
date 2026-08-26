from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    # 这个脚本只负责“切数据”，不做训练。
    # 默认从官方第一个 train 分区里抽取 asset 0/1、time_id 0..49999 的完整连续面板。
    parser = argparse.ArgumentParser(description="Build asset 0/1 continuous 50000-time dataset.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "data/raw/public_release_20260630/public_release_20260630/data/train/train_partition_000.parquet"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/asset01_time50000"))
    parser.add_argument("--time-start", type=int, default=0)
    parser.add_argument("--time-count", type=int, default=50_000)
    return parser.parse_args()


def validate_dataset(frame: pd.DataFrame, time_start: int, time_count: int) -> dict:
    # 本实验要求数据口径非常明确：
    # 1. 只有 asset 0 和 asset 1；
    # 2. 每个 asset 都必须覆盖同一段连续 time_id；
    # 3. 每个 time_id 必须同时有两个 asset，不能缺任何一边。
    # 这些校验是为了避免再次出现“前 N 行切片”那种业务含义不够清楚的数据。
    expected_times = set(range(time_start, time_start + time_count))
    feature_columns = [col for col in frame.columns if col.startswith("feature_")]
    responder_columns = [col for col in frame.columns if col.startswith("responder_")]

    summary = {
        "rows": int(len(frame)),
        "asset_counts": {},
        "time_start": time_start,
        "time_end": time_start + time_count - 1,
        "feature_count": len(feature_columns),
        "responder_count": len(responder_columns),
    }
    if len(frame) != time_count * 2:
        raise ValueError(f"expected {time_count * 2} rows, got {len(frame)}")

    # 对每个时间点统计 asset_id 的数量；正常情况下每个 time_id 都应该恰好出现 asset 0 和 asset 1。
    rows_per_time = frame.groupby("time_id")["asset_id"].nunique()
    bad_times = rows_per_time[rows_per_time != 2]
    if not bad_times.empty:
        raise ValueError(f"some time_id values do not contain both assets: {bad_times.head().to_dict()}")

    for asset_id in [0, 1]:
        asset_frame = frame.loc[frame["asset_id"] == asset_id]
        asset_times = set(asset_frame["time_id"].astype(int).tolist())
        # 每个标的都必须完整覆盖 expected_times，不能少时间点，也不能混入额外时间点。
        if asset_times != expected_times:
            missing = sorted(expected_times - asset_times)[:10]
            extra = sorted(asset_times - expected_times)[:10]
            raise ValueError(f"asset {asset_id} time range is not continuous; missing={missing}, extra={extra}")
        summary["asset_counts"][str(asset_id)] = {
            "rows": int(len(asset_frame)),
            "unique_time_count": int(asset_frame["time_id"].nunique()),
            "min_time_id": int(asset_frame["time_id"].min()),
            "max_time_id": int(asset_frame["time_id"].max()),
        }

    # 训练和验证都会用到这些核心字段；responder_* 不参与训练，但保留在切片里方便后续扩展分析。
    required_columns = {"row_id", "time_id", "asset_id", "weight", "target"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 只构建本实验需要的连续面板：两个标的、五万个连续 time_id。
    # 读取官方 parquet 后只保留本实验需要的连续面板：
    # asset_id in [0, 1] 且 time_id 属于 [time_start, time_start + time_count)。
    # 这里没有按“行数”硬切，而是按时间区间和标的集合筛选，因此样本含义稳定。
    frame = pd.read_parquet(args.source)
    end_time = args.time_start + args.time_count
    frame = frame.loc[
        frame["asset_id"].isin([0, 1])
        & (frame["time_id"] >= args.time_start)
        & (frame["time_id"] < end_time)
    ].copy()
    # 按 time_id、asset_id 稳定排序，保证后续构造序列时每个 asset 内部时间顺序正确。
    frame = frame.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)

    summary = validate_dataset(frame, args.time_start, args.time_count)
    output_path = args.output_dir / "train.parquet"
    frame.to_parquet(output_path, index=False)

    # manifest 是给人和后续脚本看的数据说明，记录来源、行数、特征数量和每个 asset 的覆盖范围。
    manifest = {
        "description": "asset_id 0/1, time_id 0..49999 continuous panel for GA + LSTM/CNN experiment",
        "source": str(args.source),
        "file": "train.parquet",
        **summary,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
