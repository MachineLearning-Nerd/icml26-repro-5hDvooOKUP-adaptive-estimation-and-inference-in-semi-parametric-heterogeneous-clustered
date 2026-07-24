"""Nuisance estimation (LightGBM) and Neyman-orthogonal loss construction.

For each of the three models the orthogonal loss reduces to a per-task scalar
quadratic

        f_j^dagger(theta) = a_j * (theta - b_j)^2 + const,

where (a_j, b_j) are computed on the second half D_{j,2} using nuisances fit on
the first half D_{j,1} (single-split; Appendix J describes R-fold cross-fitting
as an option). The Stage-1 PILOT theta_hat^init (Algorithm 1) is a separate
full-data personalized estimate used only to build the adaptive fusion weights.
We also retain per-observation orthogonal-score pieces to estimate the
asymptotic (oracle) variance sandwich Psi^{-1} Omega Psi^{-1}.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import lightgbm as lgb  # type: ignore
    _HAS_LGB = True
except Exception:  # pragma: no cover
    _HAS_LGB = False

from .data import Task


LGB_PARAMS_REG = dict(
    objective="regression", n_estimators=100, num_leaves=31, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20, verbose=-1, n_jobs=1,
)
LGB_PARAMS_CLF = dict(
    objective="binary", n_estimators=100, num_leaves=31, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20, verbose=-1, n_jobs=1,
)


def _fit_reg(Xtr, ytr, Xte):
    m = lgb.LGBMRegressor(**LGB_PARAMS_REG)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def _fit_clf_proba(Xtr, ytr, Xte):
    m = lgb.LGBMClassifier(**LGB_PARAMS_CLF)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


@dataclass
class OrthoLoss:
    """Scalar quadratic orthogonal loss a_j*(theta - b_j)^2 and influence pieces."""

    a: float
    b: float
    score_num: np.ndarray   # per-obs orthogonal-score numerator at theta=b
    curv_den: np.ndarray    # per-obs curvature (Hessian of loss /2)
    n2: int

    def asymptotic_var(self) -> float:
        """Neyman sandwich variance of sqrt(n2)*(b - theta*): E[u^2]/(E[v])^2."""
        eu2 = float(np.mean(self.score_num ** 2))
        ev = float(np.mean(self.curv_den))
        if ev <= 0:
            return float("nan")
        return eu2 / (ev ** 2)


def split_half(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(n)
    h = n // 2
    return idx[:h], idx[h:]


# ---- per-model "sufficient statistics" from a nuisance-fit on train, eval on test --------

def _plm_quad(Xtr, Ttr, Ytr, Xte, Tte, Yte):
    hhat = _fit_reg(Xtr, Ttr, Xte)
    mhat = _fit_reg(Xtr, Ytr, Xte)
    resT = Tte - hhat
    resY = Yte - mhat
    a = float(np.sum(resT ** 2))
    b = float(np.sum(resY * resT) / a) if a > 0 else float(np.mean(resY))
    return a, b, resT, resY


def _ate_quad(Xtr, Dtr, Ytr, Xte, Dte, Yte):
    pihat = np.clip(_fit_clf_proba(Xtr, Dtr, Xte), 0.05, 0.95)
    tr = Dtr == 1
    co = Dtr == 0
    m1 = _fit_reg(Xtr[tr], Ytr[tr], Xte) if tr.sum() > 5 else np.full(len(Xte), float(np.mean(Ytr[tr])) if tr.sum() else 0.0)
    m0 = _fit_reg(Xtr[co], Ytr[co], Xte) if co.sum() > 5 else np.full(len(Xte), float(np.mean(Ytr[co])) if co.sum() else 0.0)
    Yhat = Dte * (Yte - m1) / pihat - (1 - Dte) * (Yte - m0) / (1 - pihat) + m1 - m0
    a = float(len(Xte))
    b = float(np.mean(Yhat))
    return a, b, Yhat


def _did_quad(Xtr, Dtr, dYtr, Xte, Dte, dYte):
    pihat = np.clip(_fit_clf_proba(Xtr, Dtr, Xte), 0.05, 0.95)
    co = Dtr == 0
    mhat = _fit_reg(Xtr[co], dYtr[co], Xte) if co.sum() > 5 else np.full(len(Xte), float(np.mean(dYtr[co])) if co.sum() else 0.0)
    D1bar = float(np.mean(Dtr))
    vbar = float(np.mean(pihat * (1 - Dtr) / (1.0 - pihat)))
    w1 = Dte / D1bar
    w0 = (pihat * (1 - Dte) / (1.0 - pihat)) / vbar
    A = (w1 - w0) * (dYte - mhat)
    sw1 = float(np.sum(w1))
    a = float(np.mean(w1))
    b = float(np.sum(A) / sw1) if sw1 != 0 else float(np.mean(A))
    return a, b, A, sw1, w1


# ---- Stage-1 full-data pilot (Algorithm 1 step 4) -------------------------------------

def build_pilot(task: Task) -> float:
    """Full-data personalized orthogonal estimate theta_hat^init (Stage 1).

    Uses the full sample D_j with its own nuisance fit; only used to construct
    the adaptive fusion weights (it is never used for inference)."""
    X = task.X
    n = task.n
    if task.model == "PLM":
        a, b, _, _ = _plm_quad(X, task.T, task.Y, X, task.T, task.Y)
        # in-sample; a,b from full data. Use a holdout-free closed form with fitted nuisances.
        return b
    if task.model == "ATE":
        a, b, _ = _ate_quad(X, task.D, task.Y, X, task.D, task.Y)
        return b
    if task.model == "DID":
        dY = task.Y - task.Y0
        a, b, A, sw1, w1 = _did_quad(X, task.D, dY, X, task.D, dY)
        return b
    raise ValueError(task.model)


# ---- Stage-2 split orthogonal losses (Algorithm 1 steps 5-8) ---------------------------

def build_plm_loss(task: Task, rng: np.random.Generator) -> OrthoLoss:
    i1, i2 = split_half(task.n, rng)
    a, b, resT, resY = _plm_quad(task.X[i1], task.T[i1], task.Y[i1],
                                 task.X[i2], task.T[i2], task.Y[i2])
    score_num = resT * (resY - b * resT)
    curv_den = resT ** 2
    return OrthoLoss(a=a, b=b, score_num=score_num, curv_den=curv_den, n2=len(i2))


def build_ate_loss(task: Task, rng: np.random.Generator) -> OrthoLoss:
    i1, i2 = split_half(task.n, rng)
    a, b, Yhat = _ate_quad(task.X[i1], task.D[i1], task.Y[i1],
                           task.X[i2], task.D[i2], task.Y[i2])
    score_num = Yhat - b
    curv_den = np.ones(len(i2))
    return OrthoLoss(a=a, b=b, score_num=score_num, curv_den=curv_den, n2=len(i2))


def build_did_loss(task: Task, rng: np.random.Generator) -> OrthoLoss:
    i1, i2 = split_half(task.n, rng)
    dY = task.Y - task.Y0
    a, b, A, sw1, w1 = _did_quad(task.X[i1], task.D[i1], dY[i1],
                                 task.X[i2], task.D[i2], dY[i2])
    score_num = A / sw1
    curv_den = w1 / sw1
    return OrthoLoss(a=a, b=b, score_num=score_num, curv_den=curv_den, n2=len(i2))


_BUILDERS = {"PLM": build_plm_loss, "ATE": build_ate_loss, "DID": build_did_loss}


def build_losses(tasks: list[Task], rng: np.random.Generator) -> list[OrthoLoss]:
    model = tasks[0].model
    return [_BUILDERS[model](t, rng) for t in tasks]


def build_pilots(tasks: list[Task]) -> np.ndarray:
    return np.array([build_pilot(t) for t in tasks])
