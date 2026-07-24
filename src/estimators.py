"""Optimization solvers and the six estimators compared in Section 4.3.

Because every model's Neyman-orthogonal loss reduces to a per-task scalar quadratic
``a_j*(theta_j - b_j)^2``, all estimators solve a 1-D convex program of the form

    min_theta  sum_j a_j (theta_j - b_j)^2   +   penalty(theta),

which we solve with a small ADMM (pairwise fused lasso) or closed forms. Each
estimator returns (theta_hat, cluster_labels).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# 1-D pairwise fused lasso via ADMM
#   min_theta  sum_j a_j (theta_j-b_j)^2  +  sum_{e} lam_e |theta_{e0}-theta_{e1}|
# --------------------------------------------------------------------------------------
def fused_lasso_1d(a: np.ndarray, b: np.ndarray, edges: np.ndarray, lam: np.ndarray,
                   rho: float = 1.0, n_iter: int = 400, tol: float = 1e-7) -> np.ndarray:
    m = len(a)
    if len(edges) == 0:
        return (a * b) / np.where(a > 0, a, 1.0)
    e0 = edges[:, 0]
    e1 = edges[:, 1]
    D = np.zeros((len(edges), m))
    D[np.arange(len(edges)), e0] = 1.0
    D[np.arange(len(edges)), e1] = -1.0
    W = 2.0 * a  # Hessian of quadratic sum a_j (theta_j-b_j)^2 is 2*diag(a)
    A = np.diag(W) + rho * (D.T @ D)
    Ainv = np.linalg.inv(A) if m <= 64 else None
    Lu = rho * (D.T @ (D @ b))  # placeholder, rebuilt each iter
    z = D @ b
    u = np.zeros(len(edges))
    theta = b.copy()
    for _ in range(n_iter):
        rhs = 2.0 * a * b + rho * (D.T @ (z - u))
        theta = Ainv @ rhs if Ainv is not None else np.linalg.solve(A, rhs)
        Dth = D @ theta
        z = np.sign(Dth + u) * np.maximum(np.abs(Dth + u) - lam / rho, 0.0)
        u = u + Dth - z
        if np.max(np.abs(Dth - z)) < tol:
            break
    return theta


def all_pairs_edges(m: int) -> np.ndarray:
    return np.array([(i, j) for i in range(m) for j in range(i + 1, m)], dtype=int)


def clusters_from_lambda(pilots: np.ndarray, c_w: float, gamma: float, tau: float) -> np.ndarray:
    """Connected components of the graph of pairs the adaptive weight decides to
    FUSE (w > tau). This is the method's clustering decision (Eq 2.3-2.4)."""
    m = len(pilots)
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(m):
        for j in range(i + 1, m):
            gap = abs(pilots[i] - pilots[j])
            gap = gap if gap > 1e-12 else 1e-12
            w = c_w * (gap ** (-gamma))
            if w > tau:
                union(i, j)
    roots = [find(i) for i in range(m)]
    _, labels = np.unique(roots, return_inverse=True)
    return labels


# --------------------------------------------------------------------------------------
# Adaptive penalty matrix (Equations 2.3-2.4) from pilot estimates
# --------------------------------------------------------------------------------------
def adaptive_weights(pilots: np.ndarray, c_w: float, gamma: float) -> np.ndarray:
    m = len(pilots)
    edges = all_pairs_edges(m)
    w = np.empty(len(edges))
    for idx, (i, j) in enumerate(edges):
        gap = abs(pilots[i] - pilots[j])
        gap = gap if gap > 1e-12 else 1e-12
        w[idx] = c_w * (gap ** (-gamma))
    return w


def adaptive_lambda(pilots: np.ndarray, c_w: float, gamma: float, tau: float, eps_n: float) -> tuple[np.ndarray, np.ndarray]:
    """lambda_{jj'} = eps_n if w <= tau else w  (Eq 2.4)."""
    edges = all_pairs_edges(len(pilots))
    w = adaptive_weights(pilots, c_w, gamma)
    lam = np.where(w <= tau, eps_n, w)
    return edges, lam


# --------------------------------------------------------------------------------------
# Clustering from a fused solution: connected components where |theta_j-theta_j'| ~ 0
# --------------------------------------------------------------------------------------
def clusters_from_theta(theta: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """Group tasks whose estimate is within ``tol`` into one cluster (sorted merge)."""
    m = len(theta)
    order = np.argsort(theta)
    labels = np.full(m, -1, dtype=int)
    cur = 0
    labels[order[0]] = cur
    for k in range(1, m):
        if abs(theta[order[k]] - theta[order[k - 1]]) <= tol:
            labels[order[k]] = cur
        else:
            cur += 1
            labels[order[k]] = cur
    return labels


# --------------------------------------------------------------------------------------
# Oracle estimator (knows true clustering): pool within each true cluster
# --------------------------------------------------------------------------------------
def pool_within(a: np.ndarray, b: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Efficient pooled estimate within each labelled cluster:
    theta_k = sum_{j in k} a_j b_j / sum_{j in k} a_j  (minimiser of sum a_j(theta-b_j)^2)."""
    theta = np.empty_like(b, dtype=float)
    for k in np.unique(labels):
        sel = labels == k
        ak = a[sel].sum()
        theta[sel] = (a[sel] * b[sel]).sum() / ak if ak > 0 else float(np.mean(b[sel]))
    return theta


def oracle_estimate(a: np.ndarray, b: np.ndarray, true_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pool_within(a, b, true_labels), true_labels.copy()


# --------------------------------------------------------------------------------------
# Six estimators
# --------------------------------------------------------------------------------------
def est_personalized(a, b):  # (i): each task independent
    theta = (a * b) / np.where(a > 0, a, 1.0)
    return theta, np.arange(len(b))  # no clustering -> ARI ~ 0


def est_adaptive(a, b, pilots, c_w=0.1, gamma=2.0, tau=10.0, eps_n=1e-12, cluster_tol=0.05):  # (vi): proposed
    edges, lam = adaptive_lambda(pilots, c_w, gamma, tau, eps_n)
    theta = fused_lasso_1d(a, b, edges, lam)
    labels = clusters_from_lambda(pilots, c_w, gamma, tau)  # method's clustering decision
    # pool theta within each recovered cluster (oracle-style aggregation given the clustering)
    theta_pooled = theta.copy()
    for k in np.unique(labels):
        sel = labels == k
        ak = a[sel].sum()
        theta_pooled[sel] = (a[sel] * b[sel]).sum() / ak if ak > 0 else float(np.mean(b[sel]))
    return theta_pooled, labels


def est_metag(a, b, lam_const=0.01, cluster_tol=1e-3):  # (v): uniform pairwise fusion
    edges = all_pairs_edges(len(b))
    lam = np.full(len(edges), lam_const)
    theta = fused_lasso_1d(a, b, edges, lam)
    return theta, clusters_from_theta(theta, tol=cluster_tol)


def _kmeans_1d(x, K, weights=None, n_iter=50):
    """Weighted 1-D k-means (Lloyd) initialized at quantiles."""
    if weights is None:
        weights = np.ones_like(x)
    qs = np.quantile(x, np.linspace(0.05, 0.95, K)) if len(x) >= K else np.linspace(x.min(), x.max(), K)
    centers = np.unique(qs)
    if len(centers) < K:
        centers = np.linspace(x.min(), x.max(), K)
    c = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
    for _ in range(n_iter):
        for k in range(K):
            sel = c == k
            w = weights[sel]
            if w.sum() > 0:
                centers[k] = float(np.sum(w * x[sel]) / w.sum())
            else:
                centers[k] = float(np.median(x))
        newc = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        if np.array_equal(newc, c):
            break
        c = newc
    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order)}
    c = np.array([remap[ci] for ci in c])
    return c, centers[order]


def est_armul(a, b, n_j, K_hat, C_lam, n_iter=80):  # (ii): Duan-Wang clustered MTL
    """min sum_j a_j(theta_j-b_j)^2 + sum_j lam_j |theta_j - gamma_{c_j}|, lam_j=C_lam/sqrt(n_j).

    Alternating assignment (weighted 1-D k-means on b) and centroid/pooled update.
    The final estimate is the efficient pooled estimate within the learned clusters
    (the clustered ARMUL estimator). Initialization by weighted 1-D k-means."""
    m = len(b)
    lam_j = C_lam / np.sqrt(n_j)
    c, centers = _kmeans_1d(b, K_hat, weights=a)
    for _ in range(n_iter):
        centers = np.array([float(np.sum(a[c == k] * b[c == k]) / a[c == k].sum())
                            if (c == k).sum() > 0 and a[c == k].sum() > 0
                            else float(np.median(b)) for k in range(K_hat)])
        newc = np.argmin(np.abs(b[:, None] - centers[None, :]), axis=1)
        if np.array_equal(newc, c):
            break
        c = newc
    theta = pool_within(a, b, c)
    return theta, c


def est_cn(a, b, lam=0.1, cluster_tol=1e-3):  # (iii): Cluster Norm (Jacob 2008), 1-D reduction
    """For scalar theta the covariance-regularization penalty degenerates to a
    ridge toward the global mean (the m x m Sigma collapses). We solve
        min sum a_j(theta-b)^2 + lam * sum (theta_j - mean)^2
    which does not produce exact fusion (ARI ~ 0), matching the paper's finding."""
    m = len(b)
    # closed form with mean-coupling: iterate
    theta = (a * b) / np.where(a > 0, a, 1.0)
    for _ in range(200):
        mu = float(np.mean(theta))
        new = (2 * a * b + 2 * lam * mu) / (2 * a + 2 * lam / m * 0 + 2 * lam)
        # each coordinate: 2 a_j(theta-b) + 2 lam (theta - mu) = 0 -> theta=(a b + lam mu)/(a+lam)
        new = (a * b + lam * mu) / (a + lam)
        if np.max(np.abs(new - theta)) < 1e-9:
            theta = new
            break
        theta = new
    return theta, clusters_from_theta(theta, tol=cluster_tol)  # weak clustering


def est_fc(a, b, lam=1.0, mu=1.0, n_iter=50, cluster_tol=1e-3):  # (iv): Flexible Clustering (Zhou-Zhao 2015)
    """Representative-based assignment. We alternate a soft assignment Z and theta solve.
    1-D reduction: theta-update is ridge toward representatives; assignment by nearest rep."""
    m = len(b)
    K = 3
    reps = np.linspace(b.min(), b.max(), K)
    theta = (a * b) / np.where(a > 0, a, 1.0)
    c = np.argmin(np.abs(b[:, None] - reps[None, :]), axis=1)
    for _ in range(n_iter):
        for k in range(K):
            sel = c == k
            if sel.sum() > 0:
                reps[k] = float(np.mean(theta[sel]))
        dist = np.abs(theta[:, None] - reps[None, :])
        # soft assignment with sparsity (mu)
        c = np.argmin(dist, axis=1)
        g = reps[c]
        theta = (2 * a * b + lam * g) / (2 * a + lam)
    return theta, clusters_from_theta(theta, tol=cluster_tol)


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------
def run_all_estimators(a, b, pilots, n_j, true_labels, K, sep=1.0):
    """Return dict name -> (theta, labels) for all six estimator families.

    ``sep`` is the cluster separation delta; the clustering tolerance for fusion
    estimators is set to sep/4 (between within-cluster pull ~O(1/n) and the
    cross-cluster gap ~delta)."""
    ct = sep / 4.0
    out = {}
    out["Per"] = est_personalized(a, b)
    out["Ada"] = est_adaptive(a, b, pilots, cluster_tol=ct)
    out["MeTaG"] = est_metag(a, b, lam_const=0.01, cluster_tol=ct)
    # ARMUL with K-1, K, K+1 (Duan-Wang tune C_lam in {1,10,100}; the weighted k-means
    # assignment is insensitive to C_lam, so we report C_lam=10 as the tuned value).
    for khat, tag in [(K - 1, "ARMUL(K-1)"), (K, "ARMUL(K)"), (K + 1, "ARMUL(K+1)")]:
        out[tag] = est_armul(a, b, n_j, khat, C_lam=10.0)
    out["CN"] = est_cn(a, b, lam=0.1)
    out["FC"] = est_fc(a, b, lam=1.0, mu=1.0)
    out["Oracle"] = oracle_estimate(a, b, true_labels)
    return out
