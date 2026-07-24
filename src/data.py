"""Data-generating processes for the three semiparametric models from Section 4.2 of
arXiv 2605.01907.

All simulations share the task/cluster structure of Section 4.1:
  - m = 20 tasks partitioned into K = 3 latent clusters (assignment uniform at random).
  - task j has sample size n_j = 3200 + 80*j and covariate dimension p_j = 5 + j.
  - covariates X_ji ~ N(0, I_{p_j}).
  - cluster k has scalar target beta_k* = k*delta - (K+1)*delta/2 = (k - K/2 ... )delta;
    for K=3 this is (k-2)*delta, i.e. (-delta, 0, delta).
  - separation delta in {1/3, 2/3, 1}.

A "Task" bundles the raw observations needed to build the Neyman-orthogonal loss.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def beta_centroids(K: int, delta: float) -> np.ndarray:
    """beta_k* = k*delta - (K+1)*delta/2  (k = 1..K)."""
    k = np.arange(1, K + 1)
    return k * delta - (K + 1) * delta / 2.0


# --------------------------------------------------------------------------------------
# nuisance functions (exactly as written in Section 4.2)
# --------------------------------------------------------------------------------------
def _h_plm(X: np.ndarray) -> np.ndarray:
    return 0.2 * np.tanh(X.sum(axis=1))


def _g_plm(X: np.ndarray) -> np.ndarray:
    powers = (-0.8) ** (np.arange(1, X.shape[1] + 1))
    return (powers * sigmoid(X)).sum(axis=1)


def _g_ate(X: np.ndarray) -> np.ndarray:
    powers = (-0.8) ** (np.arange(1, X.shape[1] + 1))
    return (powers * sigmoid(X)).sum(axis=1)


def _propensity(X: np.ndarray) -> np.ndarray:
    pi = sigmoid(X[:, 3] * X[:, 4] - X[:, 0] * X[:, 1])
    return np.clip(pi, 0.05, 0.95)


def _mu_did0(X: np.ndarray) -> np.ndarray:
    powers = 0.7 ** (np.arange(1, X.shape[1] + 1))
    return (powers * sigmoid(X)).sum(axis=1)


def _mu_did1(X: np.ndarray) -> np.ndarray:
    powers = (-0.7) ** (np.arange(1, X.shape[1] + 1))
    return (powers * sigmoid(X)).sum(axis=1)


# --------------------------------------------------------------------------------------
# Model types
# --------------------------------------------------------------------------------------
@dataclass
class Task:
    """Per-task data for one model.

    For PLM/ATE/DID we store the raw variables; the orthogonal loss is built in
    ``nuisance.py`` after sample splitting and nuisance estimation.
    """

    model: str
    theta_star: float
    n: int
    p: int
    X: np.ndarray          # (n, p)
    # model-specific outcome/treatment arrays
    Y: np.ndarray | None = None      # PLM, ATE, DID (post-period)
    T: np.ndarray | None = None      # PLM continuous treatment
    D: np.ndarray | None = None      # ATE/DID binary treatment
    Y0: np.ndarray | None = None     # DID pre-period outcome
    true_cluster: int = 0


def simulate_assignment(m: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform-random cluster assignment for m tasks into K clusters."""
    labels = rng.integers(0, K, size=m)
    if len(np.unique(labels)) < K:  # ensure every cluster represented
        labels[:K] = np.arange(K)
        rng.shuffle(labels)
    return labels


def make_plm_task(j: int, beta: float, rng: np.random.Generator, n: int | None = None) -> Task:
    n = 3200 + 80 * j if n is None else n
    p = 5 + j
    X = rng.standard_normal((n, p))
    nu = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    T = _h_plm(X) + nu
    Y = beta * T + _g_plm(X) + eps
    return Task(model="PLM", theta_star=beta, n=n, p=p, X=X, T=T, Y=Y)


def make_ate_task(j: int, beta: float, rng: np.random.Generator, n: int | None = None) -> Task:
    n = 3200 + 80 * j if n is None else n
    p = 5 + j
    X = rng.standard_normal((n, p))
    pi = _propensity(X)
    D = (rng.uniform(size=n) < pi).astype(float)
    eps = rng.standard_normal(n)
    Y = beta * D + _g_ate(X) + eps
    return Task(model="ATE", theta_star=beta, n=n, p=p, X=X, D=D, Y=Y)


def make_did_task(j: int, beta: float, rng: np.random.Generator, n: int | None = None) -> Task:
    n = 3200 + 80 * j if n is None else n
    p = 5 + j
    X = rng.standard_normal((n, p))
    pi = _propensity(X)
    D = (rng.uniform(size=n) < pi).astype(float)
    eps0 = rng.standard_normal(n)
    eps1 = rng.standard_normal(n)
    Y0 = _mu_did0(X) + eps0
    Y1 = beta * D + _mu_did1(X) + eps1
    return Task(model="DID", theta_star=beta, n=n, p=p, X=X, D=D, Y=Y1, Y0=Y0)


_TASK_BUILDERS = {"PLM": make_plm_task, "ATE": make_ate_task, "DID": make_did_task}


def simulate_study(model: str, m: int, K: int, delta: float, seed: int) -> tuple[list[Task], np.ndarray, np.ndarray]:
    """Build one full simulated study (m tasks) for a given model and separation.

    Returns (tasks, true_labels, beta_centroids).
    """
    rng = np.random.default_rng(seed)
    labels = simulate_assignment(m, K, rng)
    betas = beta_centroids(K, delta)
    builder = _TASK_BUILDERS[model]
    tasks = [builder(j, float(betas[labels[j]]), rng) for j in range(m)]
    for j, t in enumerate(tasks):
        t.true_cluster = int(labels[j])
    return tasks, labels, betas


def make_heterogeneous_study(
    model: str, m: int, K: int, delta: float, seed: int, xi: float
) -> tuple[list[Task], np.ndarray, np.ndarray]:
    """Like simulate_study but inject within-cluster heterogeneity: theta_j = beta_k + xi*z,
    z ~ N(0,1), so ||theta_j - beta_k|| = O(xi).  Used for Claim 4 with xi = c/sqrt(n_min)."""
    rng = np.random.default_rng(seed)
    labels = simulate_assignment(m, K, rng)
    betas = beta_centroids(K, delta)
    builder = _TASK_BUILDERS[model]
    tasks = []
    for j in range(m):
        theta_j = float(betas[labels[j]]) + xi * rng.standard_normal()
        tasks.append(builder(j, theta_j, rng))
    for j, t in enumerate(tasks):
        t.true_cluster = int(labels[j])
    return tasks, labels, betas
