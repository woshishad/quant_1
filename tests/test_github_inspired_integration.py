from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "code" / "experiments" / "asset_all_tcn"
)
sys.path.insert(0, str(SCRIPT_DIR))

from audit_frozen_signal_candidate import fit_gamma  # noqa: E402
from run_github_inspired_experiment import (  # noqa: E402
    PROFILES,
    build_training_command,
)


class GithubInspiredIntegrationTests(unittest.TestCase):
    def test_profiles_never_predict_official_test(self) -> None:
        for name in PROFILES:
            command = build_training_command(
                "python",
                Path("trainer.py"),
                name,
                Path("results/probe"),
                Path("models/probe"),
            )
            self.assertIn("--skip-test-prediction", command)
            self.assertIn("--no-save-models", command)
            self.assertNotIn("target", " ".join(command))
            self.assertNotIn("responder", " ".join(command))

    def test_fit_gamma_recovers_weighted_residual_coefficient(self) -> None:
        signal = np.array([1.0, -2.0, 3.0, -4.0])
        base = np.array([0.2, -0.1, 0.3, -0.2])
        target = base + 0.25 * signal
        weight = np.array([1.0, 2.0, 3.0, 4.0])
        gamma = fit_gamma(target, base, signal, weight, bound=2.0)
        self.assertAlmostEqual(gamma, 0.25, places=12)

    def test_profile_sources_are_explicit_github_urls(self) -> None:
        for profile in PROFILES.values():
            self.assertTrue(profile["sources"])
            self.assertTrue(
                all(url.startswith("https://github.com/") for url in profile["sources"])
            )


if __name__ == "__main__":
    unittest.main()
