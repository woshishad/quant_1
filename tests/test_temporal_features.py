from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1] / "code" / "experiments" / "asset_all_tcn"
)
sys.path.insert(0, str(EXPERIMENT_DIR))

from final_residual_train_predict_ts_features import (  # noqa: E402
    add_time_series_and_cross_section_features,
    historical_feature_indices,
)


def feature_args() -> Namespace:
    return Namespace(
        disable_lag=False,
        disable_delta=False,
        disable_rolling=False,
        disable_cross_section=False,
        disable_market_history=False,
        lag_steps=[1],
        rolling_windows=[2],
        rolling_min_period_frac=0.5,
    )


class TemporalFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "row_id": [0, 1, 2, 3, 4],
                "time_id": [0, 0, 1, 2, 2],
                "asset_id": [0, 1, 0, 0, 1],
                "weight": 1.0,
                "target": 0.0,
                "feature_000": [1.0, 3.0, 2.0, 100.0, 5.0],
            }
        )

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        result, _ = add_time_series_and_cross_section_features(
            frame,
            ["feature_000"],
            ["feature_000"],
            ["feature_000"],
            feature_args(),
        )
        return result.set_index(["time_id", "asset_id"]).sort_index()

    def test_asset_lag_requires_exact_time_interval(self) -> None:
        result = self.build(self.frame)
        self.assertEqual(result.loc[(2, 0), "feature_000_lag1"], 2.0)
        self.assertTrue(np.isnan(result.loc[(2, 1), "feature_000_lag1"]))

    def test_market_history_uses_previous_cross_section(self) -> None:
        result = self.build(self.frame)
        self.assertEqual(result.loc[(2, 0), "feature_000_market_mean_lag1"], 2.0)
        self.assertEqual(result.loc[(2, 0), "feature_000_market_mean_delta1"], 50.5)

    def test_future_mutation_cannot_change_past_features(self) -> None:
        original = self.build(self.frame)
        mutated = self.frame.copy()
        mutated.loc[mutated["time_id"] == 2, "feature_000"] += 10_000.0
        changed = self.build(mutated)
        pd.testing.assert_frame_equal(
            original.loc[1],
            changed.loc[1],
        )

    def test_historical_selector_excludes_current_cross_section(self) -> None:
        names = [
            "feature_000",
            "feature_000_lag1",
            "feature_000_cs_rank",
            "feature_000_market_mean",
            "feature_000_market_rollmean2",
        ]
        self.assertEqual(historical_feature_indices(names, 1), [1, 4])


if __name__ == "__main__":
    unittest.main()
