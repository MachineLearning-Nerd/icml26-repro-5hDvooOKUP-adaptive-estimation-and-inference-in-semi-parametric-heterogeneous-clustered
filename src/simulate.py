"""Full Monte-Carlo simulation of Section 4.

Repeats each (model, delta) combination over many random studies, builds the
Neyman-orthogonal losses with LightGBM nuisances, runs all six estimators, and
records RMSE, weighted RMSE and ARI. Also keeps per-study adaptive/oracle
estimates for the normality analysis (Claim 3) and the heterogeneity sweep
(Claim 4).
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from .data import simulate_study, make_heterogeneous_study, simulate_assignment, beta_centroids, _TASK_BUILDERS
from .nuisance import build_losses, build_pilots
from .estimators import run_all_estimators, est_adaptive, oracle_estimate
from .metrics import rmse, weighted_rmse, adjusted_rand_index


def _metrics_for(theta, labels, theta_star, true_labels, n_j):
    return dict(
        rmse=rmse(theta, theta_star),
        wrmse=weighted_rmse(theta, theta_star, true_labels, n_j),
        ari=adjusted_rand_index(true_labels, labels),
    )


def run_one_mc(model: str, delta: float, seed: int, m: int = 20, K: int = 3) -> dict:
    tasks, true_labels, betas = simulate_study(model, m, K, delta, seed)
    n_j = np.array([t.n for t in tasks])
    theta_star = np.array([t.theta_star for t in tasks])
    rng = np.random.default_rng(100000 + seed)
    losses = build_losses(tasks, rng)
    a = np.array([l.a for l in losses])
    b = np.array([l.b for l in losses])
    pilots = build_pilots(tasks)  # Stage-1 full-data personalized estimate (Algorithm 1)
    ests = run_all_estimators(a, b, pilots, n_j, true_labels, K, sep=delta)
    rec = dict(model=model, delta=delta, seed=seed, theta_star=theta_star.tolist(),
               true_labels=true_labels.tolist(), n_j=n_j.tolist(), betas=betas.tolist())
    rec["methods"] = {}
    for name, (theta, labels) in ests.items():
        rec["methods"][name] = dict(theta=theta.tolist(), labels=labels.tolist(),
                                    **_metrics_for(theta, labels, theta_star, true_labels, n_j))
    return rec


def run_heterogeneity_mc(model: str, delta: float, seed: int, xi: float,
                         m: int = 20, K: int = 3) -> dict:
    tasks, true_labels, betas = make_heterogeneous_study(model, m, K, delta, seed, xi)
    n_j = np.array([t.n for t in tasks])
    theta_star = np.array([t.theta_star for t in tasks])
    rng = np.random.default_rng(100000 + seed)
    losses = build_losses(tasks, rng)
    a = np.array([l.a for l in losses])
    b = np.array([l.b for l in losses])
    pilots = build_pilots(tasks)
    th_ada, lab_ada = est_adaptive(a, b, pilots)
    th_ora, lab_ora = oracle_estimate(a, b, true_labels)
    rec = dict(model=model, delta=delta, seed=seed, xi=xi,
               theta_star=theta_star.tolist(), true_labels=true_labels.tolist(), n_j=n_j.tolist())
    rec["ada"] = dict(theta=th_ada.tolist(), rmse=rmse(th_ada, theta_star),
                      ari=adjusted_rand_index(true_labels, lab_ada))
    rec["ora"] = dict(theta=th_ora.tolist(), rmse=rmse(th_ora, theta_star))
    return rec


def run_simulation(n_mc: int = 100, models=("PLM", "ATE", "DID"),
                   deltas=(1 / 3, 2 / 3, 1.0), n_jobs: int = 4, seed0: int = 0) -> dict:
    jobs = [(mo, de, seed0 + i) for mo in models for de in deltas for i in range(n_mc)]
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_one_mc)(mo, de, s) for (mo, de, s) in jobs
    )
    return _aggregate(results, models, deltas, n_mc)


def _aggregate(results, models, deltas, n_mc) -> dict:
    agg = {}
    for mo in models:
        for de in deltas:
            sub = [r for r in results if r["model"] == mo and abs(r["delta"] - de) < 1e-9]
            methods = list(sub[0]["methods"].keys())
            row = dict(n=len(sub))
            for me in methods:
                row[me] = dict(
                    rmse=float(np.mean([r["methods"][me]["rmse"] for r in sub])),
                    wrmse=float(np.mean([r["methods"][me]["wrmse"] for r in sub])),
                    ari=float(np.mean([r["methods"][me]["ari"] for r in sub])),
                )
            agg[f"{mo}_d{de:.4f}"] = row
    return dict(summary=agg, raw=results)


def run_normality_study(model: str, delta: float, n_mc: int, n_jobs: int = 4,
                        seed0: int = 0, m: int = 20, K: int = 3) -> dict:
    """Collect adaptive & oracle per-cluster pooled estimates across MC runs for Claim 3."""
    raw = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_one_mc)(model, delta, seed0 + i, m=m, K=K) for i in range(n_mc)
    )
    return dict(model=model, delta=delta, raw=raw)


def aggregate_normality(norm_raw: dict) -> dict:
    """Build centered/scaled adaptive & oracle series per (model,delta) for Claim 3.

    ``norm_raw[key]`` is a list of run records (each with theta_star, true_labels,
    n_j, methods.Ada.theta, methods.Oracle.theta)."""
    out = {}
    for key, runs in norm_raw.items():
        ada_c, ora_c = [], []
        for r in runs:
            theta_star = np.array(r["theta_star"])
            true_labels = np.array(r["true_labels"])
            n_j = np.array(r["n_j"])
            ada = np.array(r["methods"]["Ada"]["theta"])
            ora = np.array(r["methods"]["Oracle"]["theta"])
            for k in np.unique(true_labels):
                sel = true_labels == k
                Nk = float(n_j[sel].sum())
                beta_k = float(np.mean(theta_star[sel]))
                ada_c.append(float(np.sqrt(Nk) * (float(ada[sel][0]) - beta_k)))
                ora_c.append(float(np.sqrt(Nk) * (float(ora[sel][0]) - beta_k)))
        out[key] = dict(ada_centered=np.array(ada_c), ora_centered=np.array(ora_c))
    return out


def _study_equal_n(model, m, K, delta, seed, n_per_task):
    """Build a study with equal per-task sample size n_per_task (for rate sweeps)."""
    rng = np.random.default_rng(seed)
    labels = simulate_assignment(m, K, rng)
    betas = beta_centroids(K, delta)
    builder = _TASK_BUILDERS[model]
    tasks = [builder(j, float(betas[labels[j]]), rng, n=n_per_task) for j in range(m)]
    for j, t in enumerate(tasks):
        t.true_cluster = int(labels[j])
    return tasks, labels, betas

def run_rate_sweep(model: str, n_bases: list[int], delta: float, n_mc: int,
                   n_jobs: int = 4, seed0: int = 0, m: int = 20, K: int = 3) -> list[dict]:
    """Claim 2: vary the per-task sample size and record Ada RMSE vs pooled cluster size N_k."""
    from .estimators import est_adaptive, oracle_estimate
    from .metrics import rmse
    points = []
    for nb in n_bases:
        errs = []
        Nks = []
        for i in range(n_mc):
            tasks, labels, betas = _study_equal_n(model, m, K, delta, seed0 + i, nb)
            n_j = np.array([t.n for t in tasks])
            theta_star = np.array([t.theta_star for t in tasks])
            rng = np.random.default_rng(100000 + seed0 + i)
            losses = build_losses(tasks, rng)
            a = np.array([l.a for l in losses]); b = np.array([l.b for l in losses])
            pilots = build_pilots(tasks)
            th, lab = est_adaptive(a, b, pilots)
            errs.append(rmse(th, theta_star))
            Nks.append(float(np.mean([n_j[labels == k].sum() for k in range(K)])))
        points.append(dict(Nk=float(np.mean(Nks)), rmse=float(np.mean(errs)), n_per_task=nb))
    return points


def run_heterogeneity_sweep(model: str, n_bases: list[int], delta: float, n_mc: int,
                            n_jobs: int = 4, seed0: int = 0, m: int = 20, K: int = 3) -> list[dict]:
    """Claim 4: inject xi = c/sqrt(N_k) within-cluster heterogeneity; check rate preserved."""
    from .estimators import est_adaptive
    from .metrics import rmse
    from .metrics import adjusted_rand_index as ari_f
    from .data import simulate_assignment, beta_centroids, _TASK_BUILDERS
    points = []
    for nb in n_bases:
        errs = []; aris = []
        for i in range(n_mc):
            rng = np.random.default_rng(seed0 + i)
            labels = simulate_assignment(m, K, rng)
            betas = beta_centroids(K, delta)
            Nk = nb * float(np.mean([(labels == k).sum() for k in range(K)]))
            xi = 1.0 / np.sqrt(Nk)  # xi_k = O(N_k^{-1/2})
            builder = _TASK_BUILDERS[model]
            tasks = []
            for j in range(m):
                theta_j = float(betas[labels[j]]) + xi * rng.standard_normal()
                tasks.append(builder(j, theta_j, rng, n=nb))
            for j, t in enumerate(tasks):
                t.true_cluster = int(labels[j])
            n_j = np.array([t.n for t in tasks])
            theta_star = np.array([t.theta_star for t in tasks])
            rng2 = np.random.default_rng(100000 + seed0 + i)
            losses = build_losses(tasks, rng2)
            a = np.array([l.a for l in losses]); b = np.array([l.b for l in losses])
            pilots = build_pilots(tasks)
            th, lab = est_adaptive(a, b, pilots)
            errs.append(rmse(th, theta_star)); aris.append(ari_f(labels, lab))
        points.append(dict(Nk=Nk, rmse=float(np.mean(errs)), ari=float(np.mean(aris)), n_per_task=nb))
    return points
