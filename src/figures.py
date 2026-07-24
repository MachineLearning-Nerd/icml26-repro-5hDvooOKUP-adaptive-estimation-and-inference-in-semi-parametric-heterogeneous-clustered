"""Generate report figures from a results JSON blob.

Usage:  uv run python -m src.figures path/to/results.json outdir/
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import probplot


METHODS = ["Per", "ARMUL(K-1)", "ARMUL(K)", "ARMUL(K+1)", "CN", "FC", "MeTaG", "Ada"]
COLORS = {"Ada": "#d62728", "ARMUL(K)": "#2ca02c", "Per": "#7f7f7f",
          "ARMUL(K-1)": "#98df8a", "ARMUL(K+1)": "#1f77b4",
          "CN": "#ff9896", "FC": "#aec7e8", "MeTaG": "#e377c2", "Oracle": "#9467bd"}


def fig_ari_rmse(per_run, summary, outdir):
    models = sorted({k.split("_d")[0] for k in summary})
    deltas = sorted({float(k.split("_d")[1]) for k in summary})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, mo in enumerate(models):
        # RMSE boxplots
        ax = axes[0, col]
        data = [per_run[f"{mo}_d{deltas[0]:.4f}"][me]["rmse"] for me in METHODS]
        bp = ax.boxplot(data, labels=METHODS, showfliers=False, patch_artist=True)
        for patch, me in zip(bp["boxes"], METHODS):
            patch.set_facecolor(COLORS.get(me, "#ccc"))
        ax.set_title(f"{mo}: RMSE (delta=1/3)")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=45)
        # ARI across deltas
        ax = axes[1, col]
        x = np.arange(len(deltas))
        w = 0.09
        for i, me in enumerate(METHODS):
            aris = [summary[f"{mo}_d{d:.4f}"][me]["ari"] for d in deltas]
            ax.bar(x + i * w, aris, w, label=me, color=COLORS.get(me, "#ccc"))
        ax.set_xticks(x + w * 4)
        ax.set_xticklabels([f"d={d:.2f}" for d in deltas])
        ax.set_title(f"{mo}: mean ARI by separation")
        ax.set_ylabel("ARI")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(1.0, color="k", lw=0.5, ls="--")
    axes[1, 0].legend(fontsize=7, ncol=2, loc="lower right")
    plt.tight_layout()
    p = os.path.join(outdir, "fig_ari_rmse.png")
    plt.savefig(p, dpi=130)
    plt.close()
    return p


def fig_normality(norm_out, outdir):
    keys = list(norm_out.keys())
    fig, axes = plt.subplots(2, len(keys), figsize=(5 * len(keys), 9))
    if len(keys) == 1:
        axes = axes.reshape(2, 1)
    for col, k in enumerate(keys):
        d = norm_out[k]
        ada = np.array(d["ada_centered"]); ora = np.array(d["ora_centered"])
        ax = axes[0, col]
        bins = np.linspace(min(ada.min(), ora.min()), max(ada.max(), ora.max()), 25)
        ax.hist(ada, bins=bins, alpha=0.6, label=f"Adaptive (var={np.var(ada):.2f})", color="#d62728", density=True)
        ax.hist(ora, bins=bins, alpha=0.5, label=f"Oracle (var={np.var(ora):.2f})", color="#2ca02c", density=True)
        xs = np.linspace(bins[0], bins[-1], 100)
        ax.plot(xs, np.exp(-xs ** 2 / (2 * np.var(ora))) / np.sqrt(2 * np.pi * np.var(ora)), "k--", label="N(0,oracle var)")
        ax.set_title(f"{k.split('_d')[0]}: sqrt(N_k)(theta_hat-theta*)")
        ax.legend(fontsize=8)
        ax = axes[1, col]
        probplot((ada - ada.mean()) / ada.std(), dist="norm", plot=ax)
        ax.set_title(f"{k.split('_d')[0]}: QQ (Adaptive, standardized)")
    plt.tight_layout()
    p = os.path.join(outdir, "fig_normality.png")
    plt.savefig(p, dpi=130)
    plt.close()
    return p


def fig_rate(rate_sweep, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for mo, pts in rate_sweep.items():
        ns = [p["Nk"] for p in pts]; es = [p["rmse"] for p in pts]
        ax.loglog(ns, es, "o-", label=f"{mo} (slope={np.polyfit(np.log(ns),np.log(es),1)[0]:.2f})")
    ns = np.array(ns)
    ax.loglog(ns, ns ** (-0.5) * es[1] / (ns[1] ** (-0.5)), "k--", alpha=0.5, label="N_k^{-1/2} reference")
    ax.set_xlabel("pooled cluster size N_k")
    ax.set_ylabel("RMSE")
    ax.set_title("Claim 2: pooled parametric rate (semiparametric orthogonal estimator)")
    ax.legend()
    plt.tight_layout()
    p = os.path.join(outdir, "fig_rate.png")
    plt.savefig(p, dpi=130)
    plt.close()
    return p


def fig_heterogeneity(het, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mo, pts in het.items():
        ns = [p["Nk"] for p in pts]; es = [p["rmse"] for p in pts]; aris = [p["ari"] for p in pts]
        axes[0].loglog(ns, es, "o-", label=f"{mo} (slope={np.polyfit(np.log(ns),np.log(es),1)[0]:.2f})")
        axes[1].semilogx(ns, aris, "o-", label=mo)
    axes[0].set_xlabel("N_k"); axes[0].set_ylabel("RMSE")
    axes[0].set_title("Claim 4: rate under xi_k=O(N_k^{-1/2}) heterogeneity")
    axes[0].legend()
    axes[1].set_xlabel("N_k"); axes[1].set_ylabel("ARI"); axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Claim 4: cluster recovery under heterogeneity")
    axes[1].legend()
    plt.tight_layout()
    p = os.path.join(outdir, "fig_heterogeneity.png")
    plt.savefig(p, dpi=130)
    plt.close()
    return p


def fig_realdata(real, outdir):
    if not real or not real.get("clusters"):
        return None
    cl = real["clusters"]
    fig, ax = plt.subplots(figsize=(9, 5))
    estimates = [c["estimate"] for c in cl]
    ses = [c["se"] for c in cl]
    labels = [f"Cluster {i}\n({len(c['members'])} states)" for i, c in enumerate(cl)]
    ax.barh(range(len(cl)), estimates, xerr=ses, color=["#d62728", "#ff7f0e", "#1f77b4"][:len(cl)])
    ax.set_yticks(range(len(cl)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("electricity price elasticity (theta_hat)")
    ax.set_title(f"Claim 6: RECS 2020 state-level elasticity clusters ({real.get('n_tasks')} state-tasks)")
    plt.tight_layout()
    p = os.path.join(outdir, "fig_realdata.png")
    plt.savefig(p, dpi=130)
    plt.close()
    return p


def main(results_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    res = json.load(open(results_path))
    paths = []
    paths.append(fig_ari_rmse(res["per_run"], res["summary"], outdir))
    paths.append(fig_normality(res["normality"], outdir))
    paths.append(fig_rate(res["rate_sweep"], outdir))
    paths.append(fig_heterogeneity(res["heterogeneity"], outdir))
    p = fig_realdata(res.get("real_data"), outdir)
    if p:
        paths.append(p)
    print("wrote figures:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
