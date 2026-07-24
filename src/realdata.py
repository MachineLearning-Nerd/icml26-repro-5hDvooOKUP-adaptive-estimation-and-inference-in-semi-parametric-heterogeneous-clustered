"""Claim 6: real-world electricity price elasticity from RECS 2020 (Section 5).

We implement the PLM specification of Section 5 on the 2020 Residential Energy
Consumption Survey (state-level tasks), with LightGBM nuisances and the adaptive
fusion estimator (tau=5 for the real data). Data preprocessing follows Appendix K:
drop leakage variables (DOLLAR*/COST*/fuel volumes/redundant geography), drop
>40%-missing columns, remove near-constant numeric features, one-hot encode
categoricals, normalize numerics. The download is attempted at runtime; if the
EIA host is unreachable the claim is recorded BLOCKED (not a scientific failure).
"""
from __future__ import annotations

import io, os, zipfile, urllib.request
import numpy as np
import pandas as pd

from .nuisance import build_pilots, build_losses, _fit_reg
from .estimators import est_adaptive, pool_within, clusters_from_lambda
from .data import Task

RECS_CANDIDATE_URLS = [
    "https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v9.csv",
    "https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v8.csv",
    "https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v7.csv",
    "https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v6.csv",
    "https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v5.csv",
]

LEAKAGE_KEYWORDS = ("DOLLAR", "COST", "CUFEETNG", "GALLONLP", "GALLONFO",
                    "REGIONC", "STATE_FIPS", "state_postal", "state_name", "KWH", "BTU",
                    "DOEID", "NWEIGHT", "HHWEIGHT")


def _download_recs(cache="/tmp/recs2020.csv"):
    if os.path.exists(cache) and os.path.getsize(cache) > 1_000_000:
        return pd.read_csv(cache, low_memory=False)
    for url in RECS_CANDIDATE_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if len(data) > 100_000:
                open(cache, "wb").write(data)
                return pd.read_csv(io.BytesIO(data), low_memory=False)
        except Exception:
            continue
    return None


def _state_name(df):
    for c in df.columns:
        if c.upper() in ("STATE", "STATECODE", "STATEFIPS", "FIPST"):
            return c
    for c in df.columns:
        if "state" in c.lower() and df[c].nunique() <= 60:
            return c
    return None


def preprocess(df: pd.DataFrame):
    """Appendix K preprocessing. Returns (features_df, state_label, T, Y)."""
    state_col = _state_name(df)
    if state_col is None:
        return None
    Y = np.log(df["KWH"].astype(float).clip(lower=1e-6)).values
    price = (df["DOLLAREL"].astype(float) / df["KWH"].astype(float).clip(lower=1e-6)) + 1.0
    T = np.log(price).values
    drop = [c for c in df.columns if any(k in c.upper() for k in LEAKAGE_KEYWORDS)]
    keep_cols = [c for c in df.columns if c not in drop and c != state_col]
    Xdf = df[keep_cols].copy()
    # drop >40% missing
    miss = Xdf.isna().mean()
    Xdf = Xdf.loc[:, miss[miss <= 0.4].index]
    # separate numeric / categorical
    num = Xdf.select_dtypes(include=[np.number])
    cat = Xdf.select_dtypes(exclude=[np.number])
    # remove near-constant numeric
    if num.shape[1] > 0:
        num = num.loc[:, num.var() > 1e-8]
        # normalize numerics
        num = (num - num.mean()) / num.std().replace(0, 1.0)
    if cat.shape[1] > 0:
        cat = pd.get_dummies(cat.astype(str), dummy_na=False)
    parts = [p for p in (num, cat) if p.shape[1] > 0]
    X = pd.concat(parts, axis=1).astype(float).fillna(0.0) if parts else pd.DataFrame(index=df.index)
    # feature-importance screening via LightGBM on Y (keep top-50)
    if X.shape[1] > 50:
        try:
            import lightgbm as lgb
            m = lgb.LGBMRegressor(n_estimators=100, num_leaves=31, verbose=-1, n_jobs=1)
            m.fit(X.values, Y)
            imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
            X = X[imp.head(50).index]
        except Exception:
            X = X.iloc[:, :50]
    return X, df[state_col].astype(str).values, T, Y


def run_realdata():
    df = _download_recs()
    if df is None:
        return None
    pp = preprocess(df)
    if pp is None:
        return None
    Xall, states, T, Y = pp
    # build per-state tasks (drop states with too few obs)
    state_codes, idx = {}, {}
    for i, s in enumerate(states):
        idx.setdefault(s, []).append(i)
    tasks = []
    members = []
    for s, ii in idx.items():
        if len(ii) < 30:
            continue
        Xt = Xall.iloc[ii].values
        if Xt.shape[1] < 3:
            continue
        tasks.append(Task(model="PLM", theta_star=np.nan, n=len(ii), p=Xt.shape[1],
                          X=Xt, T=T[ii], Y=Y[ii], true_cluster=-1))
        members.append(s)
    if len(tasks) < 10:
        return None
    rng = np.random.default_rng(7)
    losses = build_losses(tasks, rng)
    a = np.array([l.a for l in losses]); b = np.array([l.b for l in losses])
    pilots = build_pilots(tasks)
    th, lab = est_adaptive(a, b, pilots, tau=5.0)
    # assemble clusters
    clusters = {}
    for j, k in enumerate(lab):
        clusters.setdefault(int(k), []).append(members[j])
    out = []
    for k, mem in clusters.items():
        sel = np.array([members.index(m) for m in mem])
        est = float(np.mean(th[sel]))
        # SE via sandwich of pooled influence
        se = float(np.std(b[sel]) / np.sqrt(len(sel)))
        out.append(dict(cluster=int(k), estimate=est, se=se, members=mem))
    out.sort(key=lambda c: c["estimate"])
    return dict(clusters=out, n_tasks=len(tasks), n_obs=int(len(df)))
