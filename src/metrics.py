"""Evaluation metrics: RMSE, cluster-size weighted RMSE, and Adjusted Rand Index.

Definitions follow Section 4.4 and Appendix H of arXiv 2605.01907.
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from math import comb


def rmse(theta_hat: np.ndarray, theta_star: np.ndarray) -> float:
    return float(np.sqrt(np.mean((theta_hat - theta_star) ** 2)))


def weighted_rmse(theta_hat: np.ndarray, theta_star: np.ndarray, theta_true_cluster: np.ndarray,
                  n_j: np.ndarray) -> float:
    """wRMSE = sqrt( (1/B) sum_j N_{q(j)} (theta_hat_j - theta_j*)^2 ), B = sum_j N_{q(j)}.
    Here N_{q(j)} is the pooled cluster size of task j's true cluster."""
    cluster_sizes = {int(k): int(n_j[theta_true_cluster == k].sum()) for k in np.unique(theta_true_cluster)}
    Nq = np.array([cluster_sizes[int(c)] for c in theta_true_cluster], dtype=float)
    B = Nq.sum()
    return float(np.sqrt(np.sum(Nq * (theta_hat - theta_star) ** 2) / B))


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """ARI (Rand 1971; Hubert & Arabie 1985), as defined in Appendix H."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    if n < 2:
        return 1.0
    # contingency table
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    contingency = np.array(
        [[np.sum((labels_true == c) & (labels_pred == k)) for k in clusters] for c in classes],
        dtype=float,
    )
    sum_comb_c = sum(comb(int(n_ij), 2) for n_ij in contingency.sum(axis=1) if n_ij >= 2)
    sum_comb_k = sum(comb(int(n_ij), 2) for n_ij in contingency.sum(axis=0) if n_ij >= 2)
    sum_comb = sum(comb(int(n_ij), 2) for n_ij in contingency.ravel() if n_ij >= 2)
    expected = sum_comb_c * sum_comb_k / comb(n, 2)
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return float((sum_comb - expected) / denom)
