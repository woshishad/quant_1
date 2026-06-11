from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    n_train_times: int = 120
    n_test_times: int = 30
    n_assets: int = 40
    n_features: int = 24
    n_responders: int = 3
    train_partition_size: int = 20
    test_partition_size: int = 10
    validation_fraction: float = 0.2
    ridge_alpha: float = 4.0
    interaction_ridge_alpha: float = 8.0
    weight_ridge_alpha: float = 3.0
    weight_driver_features: tuple[int, ...] = field(default_factory=lambda: (0, 1, 4, 7))
    target_driver_features: tuple[int, ...] = field(default_factory=lambda: (2, 3, 4, 8))
    shared_driver_features: tuple[int, ...] = field(default_factory=lambda: (4,))
    interaction_features: tuple[int, ...] = field(default_factory=lambda: (0, 1, 2, 3, 4, 7, 8, 9))

    @property
    def n_train_rows(self) -> int:
        return self.n_train_times * self.n_assets

    @property
    def n_test_rows(self) -> int:
        return self.n_test_times * self.n_assets

