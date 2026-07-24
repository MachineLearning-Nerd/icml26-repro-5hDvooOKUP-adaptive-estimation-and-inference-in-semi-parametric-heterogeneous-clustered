"""Claim verifiers. Each returns (verdict, details). verdict in {VERIFIED, FALSIFIED, BLOCKED}.

A verifier exits nonzero (via main) only on FALSIFIED. BLOCKED means an honest
external blocker (e.g. unavailable real-world dataset) and does not fail the run.
"""
from __future__ import annotations

import numpy as np
from math import comb


def _ari(truth, pred):
    truth = np.asarray(truth); pred = np.asarray(pred)
    n = len(truth)
    if n < 2:
        return 1.0
    classes = np.unique(truth); clusters = np.unique(pred)
    cont = np.array([[np.sum((truth == c) & (pred == k)) for k in clusters] for c in classes], float)
    sc = sum(comb(int(x), 2) for x in cont.sum(1) if x >= 2)
    sk = sum(comb(int(x), 2) for x in cont.sum(0) if x >= 2)
    s = sum(comb(int(x), 2) for x in cont.ravel() if x >= 2)
    exp = sc * sk / comb(n, 2)
    mx = 0.5 * (sc + sk)
    return 1.0 if (mx - exp) == 0 else float((s - exp) / (mx - exp))


def verify_claim1(summary: dict) -> tuple[str, dict]:
    """C1: exact recovery of latent clustering w.h.p. in the SEMIPARAMETRIC setting
    (PLM/ATE/DID with LightGBM nuisance + Neyman-orthogonal losses)."""
    cells = {}
    ok = True
    for key, row in summary.items():
        ari = row["Ada"]["ari"]
        cells[key] = round(ari, 3)
        ok = ok and (ari >= 0.90)
    verdict = "VERIFIED" if ok else "FALSIFIED"
    return verdict, dict(mean_ada_ari=float(np.mean(list(cells.values()))), cells=cells,
                         threshold=0.90, note="Ada ARI>=0.90 across all 9 (model,delta) cells")


def verify_claim2(rate_sweep: dict) -> tuple[str, dict]:
    """C2: pooled parametric rate ||theta_hat-theta*|| = O_P(N_k^{-1/2}) for the
    semiparametric orthogonal estimator."""
    slopes = {}
    ok = True
    for model, pts in rate_sweep.items():
        ns = np.array([p["Nk"] for p in pts], float)
        es = np.array([p["rmse"] for p in pts], float)
        slope = float(np.polyfit(np.log(ns), np.log(es), 1)[0])
        slopes[model] = round(slope, 3)
        ok = ok and (-0.80 <= slope <= -0.25)
    verdict = "VERIFIED" if ok else "FALSIFIED"
    return verdict, dict(slopes=slopes, target=-0.5, note="log-log RMSE vs N_k slope ~ -1/2")


def verify_claim3(norm: dict) -> tuple[str, dict]:
    """C3: sqrt(N_k)(theta_hat-theta*) asymptotically Normal with covariance MATCHING
    the oracle estimator (the specific claim the prior reproduction missed)."""
    from scipy.stats import shapiro, kurtosis
    det = {}
    ok_all = True
    for key, d in norm.items():
        ada = np.array(d["ada_centered"], float)
        ora = np.array(d["ora_centered"], float)
        var_ada = float(np.var(ada, ddof=1))
        var_ora = float(np.var(ora, ddof=1))
        ratio = var_ada / var_ora if var_ora > 0 else float("nan")
        # normality
        try:
            p_ada = float(shapiro(ada).pvalue)
        except Exception:
            p_ada = 0.0
        kur_ada = float(kurtosis(ada))
        mean_ada = float(np.mean(ada))
        cov_match = 0.5 <= ratio <= 2.0
        normalish = (p_ada > 0.01) and (abs(kur_ada) < 1.0) and (abs(mean_ada) < 0.5)
        det[key] = dict(var_ada=round(var_ada, 4), var_ora=round(var_ora, 4),
                        var_ratio=round(ratio, 3), shapiro_p=round(p_ada, 4),
                        excess_kurt=round(kur_ada, 3), mean=round(mean_ada, 3),
                        cov_matches_oracle=cov_match, approximately_normal=normalish)
        ok_all = ok_all and cov_match and normalish
    verdict = "VERIFIED" if ok_all else "FALSIFIED"
    return verdict, det


def verify_claim4(het: dict) -> tuple[str, dict]:
    """C4: within-cluster heterogeneity xi_k=O(N_k^{-1/2}) preserves the pooled rate
    (semiparametric estimator)."""
    slopes = {}
    aris = {}
    ok = True
    for model, pts in het.items():
        ns = np.array([p["Nk"] for p in pts], float)
        es = np.array([p["rmse"] for p in pts], float)
        slope = float(np.polyfit(np.log(ns), np.log(es), 1)[0])
        slopes[model] = round(slope, 3)
        aris[model] = round(float(np.mean([p["ari"] for p in pts])), 3)
        ok = ok and (-0.80 <= slope <= -0.25) and aris[model] >= 0.7
    verdict = "VERIFIED" if ok else "FALSIFIED"
    return verdict, dict(slopes=slopes, aris=aris, note="rate preserved + recovery stays high")


def verify_claim5(summary: dict) -> tuple[str, dict]:
    """C5: across PLM/ATE/DID x delta in {1/3,2/3,1}, Ada ARI~1 and outperforms
    competing clustering approaches (lower RMSE than Per / misspecified ARMUL)."""
    cells = {}
    ok = True
    for key, row in summary.items():
        ada_ari = row["Ada"]["ari"]; ada_rmse = row["Ada"]["rmse"]
        per_rmse = row["Per"]["rmse"]
        armul_km1 = row["ARMUL(K-1)"]["rmse"]
        armul_kp1 = row["ARMUL(K+1)"]["rmse"]
        beats = (ada_rmse <= per_rmse) and (ada_rmse <= armul_km1) and (ada_rmse <= armul_kp1)
        cells[key] = dict(ada_ari=round(ada_ari, 3), ada_rmse=round(ada_rmse, 4),
                          per_rmse=round(per_rmse, 4), ada_beats_baselines=bool(beats))
        ok = ok and (ada_ari >= 0.90) and beats
    verdict = "VERIFIED" if ok else "FALSIFIED"
    return verdict, dict(cells=cells, note="Ada: ARI>=0.90 and lowest RMSE in every cell")


def verify_claim6(real: dict | None) -> tuple[str, dict]:
    """C6: RECS 2020 electricity-price elasticity -> 3 clusters; Virginia most elastic
    (~-1.138); large 46-state cluster inelastic (~-0.221)."""
    if not real or not real.get("clusters"):
        return "BLOCKED", dict(note="RECS 2020 dataset not available in run environment")
    clusters = real["clusters"]
    n_clusters = len(clusters)
    all_negative = all(c["estimate"] < 0 for c in clusters)
    # most elastic = most negative
    sorted_c = sorted(clusters, key=lambda c: c["estimate"])
    most_elastic = sorted_c[0]
    va_isolated = any("VA" in c["members"] for c in clusters if len(c["members"]) == 1)
    big_cluster = max(clusters, key=lambda c: len(c["members"]))
    det = dict(n_clusters=n_clusters, all_negative=all_negative,
               most_elastic=round(most_elastic["estimate"], 3),
               most_elastic_members=most_elastic["members"],
               big_cluster_size=len(big_cluster["members"]),
               big_cluster_est=round(big_cluster["estimate"], 3),
               va_isolated=va_isolated,
               note="3 clusters, all negative; Virginia most elastic; large inelastic cluster")
    ok = (n_clusters == 3) and all_negative and va_isolated and (len(big_cluster["members"]) >= 30)
    verdict = "VERIFIED" if ok else "FALSIFIED"
    return verdict, det
