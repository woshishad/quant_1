from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def weighted_mean(values: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    if sample_weight is None:
        return float(values.mean())
    sample_weight = np.asarray(sample_weight, dtype=float)
    denominator = sample_weight.sum()
    if denominator <= 0:
        return float(values.mean())
    return float(np.sum(sample_weight * values) / denominator)


def weighted_std(values: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    mean = weighted_mean(values, sample_weight=sample_weight)
    centered = values - mean
    if sample_weight is None:
        variance = np.mean(centered**2)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        denominator = sample_weight.sum()
        if denominator <= 0:
            variance = np.mean(centered**2)
        else:
            variance = np.sum(sample_weight * centered**2) / denominator
    return float(np.sqrt(max(variance, 1e-12)))


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray, sample_weight: np.ndarray | None = None) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        if sample_weight is None:
            mean = X.mean(axis=0)
            scale = X.std(axis=0)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            total_weight = sample_weight.sum()
            if total_weight <= 0:
                mean = X.mean(axis=0)
                scale = X.std(axis=0)
            else:
                mean = np.sum(X * sample_weight[:, None], axis=0) / total_weight
                centered = X - mean
                scale = np.sqrt(np.sum(sample_weight[:, None] * centered**2, axis=0) / total_weight)
        scale = np.where(scale <= 1e-12, 1.0, scale)
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, sample_weight=sample_weight).transform(X)


@dataclass
class RidgeRegressor:
    alpha: float = 1.0
    standardizer: Standardizer | None = None
    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "RidgeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.standardizer = Standardizer().fit(X, sample_weight=sample_weight)
        Xs = self.standardizer.transform(X)
        if sample_weight is None:
            y_mean = float(y.mean())
            yc = y - y_mean
            Xw = Xs
            yw = yc
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            y_mean = weighted_mean(y, sample_weight=sample_weight)
            yc = y - y_mean
            sqrt_weight = np.sqrt(np.maximum(sample_weight, 0.0))
            Xw = Xs * sqrt_weight[:, None]
            yw = yc * sqrt_weight
        xtx = Xw.T @ Xw
        regularizer = self.alpha * np.eye(xtx.shape[0], dtype=float)
        coef = np.linalg.solve(xtx + regularizer, Xw.T @ yw)
        self.coef_ = coef
        self.intercept_ = y_mean
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.standardizer is None or self.coef_ is None:
            raise RuntimeError("RidgeRegressor is not fitted")
        Xs = self.standardizer.transform(X)
        return self.intercept_ + Xs @ self.coef_


@dataclass
class FeatureInteractionBuilder:
    interaction_features: tuple[int, ...]
    include_squares: bool = True

    def fit(self, X: np.ndarray | None = None) -> "FeatureInteractionBuilder":
        return self

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
        X = np.asarray(X, dtype=float)
        features = [X]
        names: list[str] = []
        if self.include_squares:
            for idx in self.interaction_features:
                features.append((X[:, idx] ** 2)[:, None])
                names.append(f"feature_{idx}^2")
        for left_pos, left_idx in enumerate(self.interaction_features):
            for right_idx in self.interaction_features[left_pos + 1 :]:
                features.append((X[:, left_idx] * X[:, right_idx])[:, None])
                names.append(f"feature_{left_idx}*feature_{right_idx}")
        transformed = np.hstack(features)
        return transformed, names


@dataclass
class FusionRegressor:
    base_model: RidgeRegressor
    interaction_model: RidgeRegressor
    interaction_builder: FeatureInteractionBuilder
    base_weight: float = 0.65

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "FusionRegressor":
        self.base_model.fit(X, y, sample_weight=sample_weight)
        transformed, _ = self.interaction_builder.transform(X)
        self.interaction_model.fit(transformed, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        base_pred = self.base_model.predict(X)
        transformed, _ = self.interaction_builder.transform(X)
        interaction_pred = self.interaction_model.predict(transformed)
        return self.base_weight * base_pred + (1.0 - self.base_weight) * interaction_pred


def aggregated_feature_importance(
    feature_names: list[str],
    base_coef: np.ndarray,
    interaction_names: list[str],
    interaction_coef: np.ndarray,
) -> list[tuple[str, float]]:
    importance = {name: abs(float(coef)) for name, coef in zip(feature_names, base_coef)}
    for name, coef in zip(interaction_names, interaction_coef[len(feature_names) :]):
        if "^2" in name:
            base_name = name.split("^")[0]
            importance[base_name] = importance.get(base_name, 0.0) + abs(float(coef))
        elif "*" in name:
            left_name, right_name = name.split("*")
            importance[left_name] = importance.get(left_name, 0.0) + 0.5 * abs(float(coef))
            importance[right_name] = importance.get(right_name, 0.0) + 0.5 * abs(float(coef))
    ranked = sorted(importance.items(), key=lambda item: item[1], reverse=True)
    return ranked


def top_k(items: Iterable[tuple[str, float]], k: int = 10) -> list[tuple[str, float]]:
    return list(items)[:k]
