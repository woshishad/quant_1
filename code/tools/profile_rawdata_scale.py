from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


@dataclass
class AssetStats:
    row_count: int = 0
    min_time_id: int | None = None
    max_time_id: int | None = None
    time_ids: set[int] = field(default_factory=set)

    def update(self, asset_ids: np.ndarray, time_ids: np.ndarray, asset_id: int) -> None:
        # 只处理当前 asset 的两列索引信息，避免把 300 多列特征整表读进内存。
        mask = asset_ids == asset_id
        if not np.any(mask):
            return
        selected_times = time_ids[mask].astype(np.int64, copy=False)
        self.row_count += int(selected_times.size)
        local_min = int(selected_times.min())
        local_max = int(selected_times.max())
        self.min_time_id = local_min if self.min_time_id is None else min(self.min_time_id, local_min)
        self.max_time_id = local_max if self.max_time_id is None else max(self.max_time_id, local_max)
        self.time_ids.update(int(value) for value in selected_times.tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile raw competition data scale by asset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/data_profile"))
    parser.add_argument("--batch-size", type=int, default=250_000)
    return parser.parse_args()


def manifest_files(data_root: Path, split: str) -> list[Path]:
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {}).get(split, [])
        if files:
            return [data_root / str(file) for file in files]
    return sorted((data_root / split).glob("*.parquet"))


def schema_counts(files: Iterable[Path]) -> dict[str, int]:
    first_file = next(iter(files))
    schema = pq.ParquetFile(first_file).schema_arrow
    names = schema.names
    return {
        "column_count": len(names),
        "feature_count": sum(name.startswith("feature_") for name in names),
        "responder_count": sum(name.startswith("responder_") for name in names),
        "has_weight": int("weight" in names),
        "has_target": int("target" in names),
    }


def profile_split(files: list[Path], split: str, batch_size: int) -> tuple[pd.DataFrame, dict[str, int]]:
    stats: dict[int, AssetStats] = defaultdict(AssetStats)
    parquet_rows = 0
    feature_counts = schema_counts(files)

    for path in files:
        parquet_file = pq.ParquetFile(path)
        parquet_rows += parquet_file.metadata.num_rows
        for batch in parquet_file.iter_batches(columns=["asset_id", "time_id"], batch_size=batch_size):
            frame = batch.to_pandas()
            asset_values = frame["asset_id"].to_numpy(dtype=np.int64, copy=False)
            time_values = frame["time_id"].to_numpy(dtype=np.int64, copy=False)
            for asset_id in np.unique(asset_values):
                stats[int(asset_id)].update(asset_values, time_values, int(asset_id))

    rows = []
    for asset_id in sorted(stats):
        item = stats[asset_id]
        rows.append(
            {
                "split": split,
                "asset_id": asset_id,
                "feature_count": feature_counts["feature_count"],
                "responder_count": feature_counts["responder_count"],
                "row_count": item.row_count,
                "unique_time_count": len(item.time_ids),
                "min_time_id": item.min_time_id,
                "max_time_id": item.max_time_id,
                "missing_time_count_in_range": (
                    (int(item.max_time_id) - int(item.min_time_id) + 1 - len(item.time_ids))
                    if item.min_time_id is not None and item.max_time_id is not None
                    else None
                ),
                "has_weight": bool(feature_counts["has_weight"]),
                "has_target": bool(feature_counts["has_target"]),
            }
        )
    return pd.DataFrame(rows), {"parquet_rows": parquet_rows, **feature_counts}


def save_asset_time_plot(combined: pd.DataFrame, output_path: Path) -> None:
    # 画出每个标的在训练集和测试集中的时间点数量，方便肉眼检查规模是否均衡。
    x = np.arange(len(combined))
    width = 0.38
    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, combined["train_unique_time_count"], width=width, label="train")
    plt.bar(x + width / 2, combined["test_unique_time_count"], width=width, label="test")
    plt.xticks(x, combined["asset_id"].astype(str))
    plt.xlabel("asset_id")
    plt.ylabel("unique time_id count")
    plt.title("Raw Data Time Points by Asset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_frames = []
    split_summaries = {}
    for split in ["train", "test"]:
        files = manifest_files(args.data_root, split)
        if not files:
            raise FileNotFoundError(f"No parquet files found for split={split} under {args.data_root}")
        frame, summary = profile_split(files, split, args.batch_size)
        split_frames.append(frame)
        split_summaries[split] = summary

    result = pd.concat(split_frames, ignore_index=True)
    result.to_csv(args.output_dir / "rawdata_asset_scale.csv", index=False)

    combined = (
        result.groupby("asset_id", as_index=False)
        .agg(
            feature_count=("feature_count", "max"),
            train_rows=("row_count", lambda s: int(s[result.loc[s.index, "split"] == "train"].sum())),
            test_rows=("row_count", lambda s: int(s[result.loc[s.index, "split"] == "test"].sum())),
            train_unique_time_count=(
                "unique_time_count",
                lambda s: int(s[result.loc[s.index, "split"] == "train"].sum()),
            ),
            test_unique_time_count=(
                "unique_time_count",
                lambda s: int(s[result.loc[s.index, "split"] == "test"].sum()),
            ),
        )
        .sort_values("asset_id")
    )
    combined.to_csv(args.output_dir / "rawdata_asset_scale_wide.csv", index=False)
    save_asset_time_plot(combined, args.output_dir / "rawdata_asset_time_counts.png")

    summary = {
        "data_root": str(args.data_root),
        "asset_count": int(result["asset_id"].nunique()),
        "splits": split_summaries,
        "output_files": [
            "rawdata_asset_scale.csv",
            "rawdata_asset_scale_wide.csv",
            "rawdata_asset_time_counts.png",
            "rawdata_scale_summary.json",
        ],
    }
    (args.output_dir / "rawdata_scale_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
