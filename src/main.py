"""Entrypoint for the faithful reproduction of arXiv 2605.01907.

Runs the full Monte-Carlo simulation (Claims 1, 2, 5), the asymptotic-normality
analysis vs the oracle covariance (Claim 3), the within-cluster heterogeneity
sweep (Claim 4) and the RECS-2020 real-data experiment (Claim 6), then runs the
claim verifiers and prints a single JSON results blob to stdout.

The run command (fixed across every node) is:  uv run python -m src.main
"""
from __future__ import annotations

import json, os, sys, time, platform
import numpy as np

from . import simulate as S
from . import verify as V
from .realdata import run_realdata


NMC = 100            # Monte-Carlo replications per (model, delta) cell (encoded in code)
NMC_RATE = 24
NMC_HET = 24
MODELS = ["PLM", "ATE", "DID"]
DELTAS = [1 / 3, 2 / 3, 1.0]
N_BASES = [400, 800, 1600, 3200]


def _n_jobs():
    return min(os.cpu_count() or 4, 16)


def _json_default(o):
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main():
    t0 = time.time()
    nj = _n_jobs()
    log = dict(platform=platform.platform(), cpu_count=os.cpu_count(), n_jobs=nj,
               python=platform.python_version(), nmc=NMC, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    print(f"[main] platform={log['platform']} cpus={os.cpu_count()} n_jobs={nj} nmc={NMC}", flush=True)

    # --- Claims 1, 5: full simulation -------------------------------------------
    print("[main] running full simulation (Claims 1, 5)...", flush=True)
    sim = S.run_simulation(n_mc=NMC, models=MODELS, deltas=DELTAS, n_jobs=nj, seed0=0)
    summary = sim["summary"]
    print("[main] simulation done", flush=True)

    # compact per-run metrics for boxplots (drop the heavy per-task theta arrays)
    method_names = list(sim["raw"][0]["methods"].keys())
    per_run = {}
    for mo in MODELS:
        for de in DELTAS:
            sub = [r for r in sim["raw"] if r["model"] == mo and abs(r["delta"] - de) < 1e-9]
            per_run[f"{mo}_d{de:.4f}"] = {
                me: dict(rmse=[r["methods"][me]["rmse"] for r in sub],
                         ari=[r["methods"][me]["ari"] for r in sub]) for me in method_names}

    # --- Claim 3: normality from delta=1/3 subset -------------------------------
    norm_raw = {}
    for mo in MODELS:
        norm_raw[f"{mo}_d{1/3:.4f}"] = [r for r in sim["raw"] if r["model"] == mo and abs(r["delta"] - 1/3) < 1e-9]
    norm_agg = S.aggregate_normality(norm_raw)
    norm_out = {k: dict(ada_centered=v["ada_centered"].tolist(), ora_centered=v["ora_centered"].tolist())
                for k, v in norm_agg.items()}

    # --- Claim 2: pooled rate sweep ---------------------------------------------
    print("[main] running rate sweep (Claim 2)...", flush=True)
    rate = {mo: S.run_rate_sweep(mo, N_BASES, 1/3, NMC_RATE, n_jobs=nj) for mo in MODELS}

    # --- Claim 4: heterogeneity sweep -------------------------------------------
    print("[main] running heterogeneity sweep (Claim 4)...", flush=True)
    het = {mo: S.run_heterogeneity_sweep(mo, N_BASES, 1/3, NMC_HET, n_jobs=nj) for mo in MODELS}

    # --- Claim 6: real data -----------------------------------------------------
    print("[main] running real-data experiment (Claim 6)...", flush=True)
    try:
        real = run_realdata()
    except Exception as e:
        real = None
        print(f"[main] real-data failed: {e}", flush=True)

    # --- verify -----------------------------------------------------------------
    verdicts = {}
    verdicts["C1_exact_recovery"] = V.verify_claim1(summary)
    verdicts["C2_pooled_rate"] = V.verify_claim2(rate)
    verdicts["C3_normality"] = V.verify_claim3(norm_out)
    verdicts["C4_heterogeneity"] = V.verify_claim4(het)
    verdicts["C5_simulations"] = V.verify_claim5(summary)
    verdicts["C6_real_data"] = V.verify_claim6(real)

    log["elapsed_sec"] = round(time.time() - t0, 1)
    result = dict(meta=log, summary=summary, per_run=per_run, normality=norm_out, rate_sweep=rate,
                  heterogeneity=het, real_data=real, verdicts=verdicts)

    print("===RESULTS_JSON_BEGIN===", flush=True)
    print(json.dumps(result, default=_json_default), flush=True)
    print("===RESULTS_JSON_END===", flush=True)

    # summary of verdicts
    any_falsified = False
    for k, (v, det) in verdicts.items():
        print(f"[verdict] {k}: {v}", flush=True)
        if v == "FALSIFIED":
            any_falsified = True
    print(f"[main] elapsed {log['elapsed_sec']}s", flush=True)
    sys.exit(1 if any_falsified else 0)


if __name__ == "__main__":
    main()
