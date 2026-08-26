"""评估指标实现：加权 R²、普通 R² 和皮尔逊相关系数。"""

from __future__ import annotations

import numpy as np


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    # 和赛题公式一致：用权重计算残差平方和，再除以加权平方和。
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
    residual = np.sum(sample_weight * (y_true - y_pred) ** 2)
    total = np.sum(sample_weight * (y_true**2))
    if total <= 0:
        return 0.0
    return 1.0 - residual / total


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 传统 R² 仅用于辅助观察，不是官方主指标。
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    centered = y_true - y_true.mean()
    denominator = np.sum(centered**2)
    if denominator <= 0:
        return 0.0
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denominator


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    # 相关系数主要用来做快速线性诊断。
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denom <= 0:
        return 0.0
    return float(np.sum(x * y) / denom)
