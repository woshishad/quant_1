from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multiple calibration feature rankings into stable features.")
    parser.add_argument("--ranking-files", type=Path, nargs="+", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset_all_stable_features_100k"))
    parser.add_argument("--top-k", type=int, nargs="+", default=[32, 64, 128])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for fold_index, path in enumerate(args.ranking_files):
        ranking = pd.read_csv(path)
        required = {"feature_name", "rank", "cal_score"}
        missing = required - set(ranking.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        # 每一折 calibration 都是发生在对应 holdout 之前的过去区间；聚合这些排名用于后续未来段，不看未来 target。
        frames.append(
            ranking[["feature_name", "rank", "cal_score"]].rename(
                columns={
                    "rank": f"rank_fold{fold_index}",
                    "cal_score": f"cal_score_fold{fold_index}",
                }
            )
        )

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="feature_name", how="outer")

    rank_columns = [column for column in merged.columns if column.startswith("rank_fold")]
    score_columns = [column for column in merged.columns if column.startswith("cal_score_fold")]
    feature_count = int(merged[rank_columns].max().max())
    merged[rank_columns] = merged[rank_columns].fillna(feature_count + 1)
    merged[score_columns] = merged[score_columns].fillna(0.0)
    merged["mean_rank"] = merged[rank_columns].mean(axis=1)
    merged["worst_rank"] = merged[rank_columns].max(axis=1)
    merged["best_rank"] = merged[rank_columns].min(axis=1)
    merged["mean_cal_score"] = merged[score_columns].mean(axis=1)
    merged["positive_fold_count"] = (merged[score_columns] > 0).sum(axis=1)

    stable = merged.sort_values(
        ["mean_rank", "worst_rank", "mean_cal_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    stable.insert(0, "stable_rank", np.arange(1, len(stable) + 1))
    stable_path = args.results_dir / "stable_feature_ranking.csv"
    stable.to_csv(stable_path, index=False)

    top_files = {}
    for top_k in args.top_k:
        output_path = args.results_dir / f"selected_features_stable_top{top_k}.csv"
        stable.head(top_k)[
            ["stable_rank", "feature_name", "mean_rank", "worst_rank", "mean_cal_score", "positive_fold_count"]
        ].to_csv(output_path, index=False)
        top_files[str(top_k)] = str(output_path)

    manifest = {
        "ranking_files": [str(path) for path in args.ranking_files],
        "feature_count": int(len(stable)),
        "ranking_rule": "sort by mean_rank, then worst_rank, then mean_cal_score descending",
        "top_files": top_files,
        "top20": stable.head(20)[
            ["stable_rank", "feature_name", "mean_rank", "worst_rank", "mean_cal_score", "positive_fold_count"]
        ].to_dict(orient="records"),
    }
    (args.results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
