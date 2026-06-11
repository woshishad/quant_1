from __future__ import annotations

import argparse
from pathlib import Path

from synthetic_competition.config import SyntheticConfig
from synthetic_competition.data import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic competition data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-times", type=int, default=120)
    parser.add_argument("--n-test-times", type=int, default=30)
    parser.add_argument("--n-assets", type=int, default=40)
    parser.add_argument("--n-features", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SyntheticConfig(
        seed=args.seed,
        n_train_times=args.n_train_times,
        n_test_times=args.n_test_times,
        n_assets=args.n_assets,
        n_features=args.n_features,
    )
    manifest = generate_dataset(config, args.output_dir)
    print(manifest["train_path"])
    print(manifest["test_path"])
    print(manifest["sample_submission_path"])


if __name__ == "__main__":
    main()

