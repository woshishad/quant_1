from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import pearson_corr, weighted_r2
from .models import aggregated_feature_importance, top_k


def permutation_importance(
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    sample_weight: np.ndarray | None = None,
    metric=weighted_r2,
    random_state: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    if sample_weight is None:
        baseline = metric(y, predict_fn(X))
    else:
        baseline = metric(y, predict_fn(X), sample_weight=sample_weight)
    rows = []
    for feature_index, feature_name in enumerate(feature_names):
        permuted = X.copy()
        permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
        if sample_weight is None:
            score = metric(y, predict_fn(permuted))
        else:
            score = metric(y, predict_fn(permuted), sample_weight=sample_weight)
        rows.append(
            {
                "feature": feature_name,
                "baseline_score": baseline,
                "permuted_score": score,
                "importance": baseline - score,
            }
        )
    frame = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    return frame


def correlation_table(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for feature_index, feature_name in enumerate(feature_names):
        rows.append({"feature": feature_name, "corr": pearson_corr(X[:, feature_index], y)})
    frame = pd.DataFrame(rows)
    frame["abs_corr"] = frame["corr"].abs()
    return frame.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"]).reset_index(drop=True)


def feature_recovery_report(
    feature_names: list[str],
    base_coef: np.ndarray,
    interaction_names: list[str],
    interaction_coef: np.ndarray,
    true_drivers: list[str],
    top_k_size: int = 8,
) -> dict[str, Any]:
    ranked = aggregated_feature_importance(feature_names, base_coef, interaction_names, interaction_coef)
    recovered = top_k(ranked, top_k_size)
    recovered_names = [name for name, _ in recovered]
    hits = [name for name in true_drivers if name in recovered_names]
    precision = len(hits) / max(len(recovered_names), 1)
    recall = len(hits) / max(len(true_drivers), 1)
    return {
        "top_k": recovered,
        "true_drivers": true_drivers,
        "hits": hits,
        "precision": precision,
        "recall": recall,
    }


def serialize_dataframe(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")
