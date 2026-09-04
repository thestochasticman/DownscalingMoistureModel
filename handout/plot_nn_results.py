"""nn-track 4-panel results figure, from cached CV predictions.

(a) the recommended station ensemble's leave-site-out fit; (b) per-station
NSE, ensemble vs model8 (the strongest single model); (c) blocked per-block
NSE from the RF-era reference through to the recommended blocked ensemble;
(d) preprocessing attribution on the hybrid (the quantile result). Reads only
data/*_predictions.csv; nothing retrains.

Run from repo root::  PYTHONPATH=. python handout/plot_nn_results.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emt.evaluation import metrics
from emt.nn.cv import block_of

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "nn_track_results.png"
TARGET = "sm_rootzone_pct"
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728",
              "MURRUMBIDGEE": "#9467bd"}
SITES = ["ADELONG", "KYEAMBA", "MURRUMBIDGEE", "YANCO"]

tab = pd.read_csv(REPO / "data" / "process_target_2006_2010.csv")
site_of = tab.drop_duplicates("station").set_index("station")["site"]


def load(name):
    return pd.read_csv(REPO / "data" / name, parse_dates=["time"])


def per_site(p):
    return p.groupby("station").apply(lambda g: pd.Series(metrics(g[TARGET], g["pred"])),
                                      include_groups=False)


def per_block(p):
    return (p.assign(b=p["station"].map(block_of)).groupby("b")
             .apply(lambda g: pd.Series(metrics(g[TARGET], g["pred"])), include_groups=False))


ens_s = load("nn_ens_stationcv_predictions.csv")     # mean(hybrid+anchor, model8, seq-big)
m8_s = load("model8_losocv_capacity_aridity_weighted_predictions.csv")
ens_s["site"] = ens_s["station"].map(site_of)
ps_h, ps_8 = per_site(ens_s), per_site(m8_s)
pooled = metrics(ens_s[TARGET], ens_s["pred"])

fig, ax = plt.subplots(2, 2, figsize=(15, 11))

# (a) LOSO fit
for s in SITES:
    d = ens_s[ens_s.site == s]
    ax[0, 0].scatter(d[TARGET], d["pred"], s=6, alpha=.25, color=SITE_COLOR[s], label=s)
lim = [0, 55]
ax[0, 0].plot(lim, lim, "k--", lw=1)
ax[0, 0].set(xlim=lim, ylim=lim, xlabel="observed root-zone (%)", ylabel="held-out predicted (%)",
             title=f"(a) recommended ensemble, leave-site-out fit  (pooled NSE {pooled['nse']:+.2f}, r {pooled['r']:.2f})")
leg = ax[0, 0].legend(fontsize=9, markerscale=2)
for lh in leg.legend_handles:
    lh.set_alpha(1)
ax[0, 0].grid(alpha=.3)

# (b) per-station NSE, hybrid vs model8 (paired dots, sorted by model8)
order = ps_8["nse"].sort_values().index
y = np.arange(len(order))
ax[0, 1].scatter(ps_8.loc[order, "nse"].clip(-3), y, s=22, color="0.45", label="model8 (grey)", zorder=3)
ax[0, 1].scatter(ps_h.loc[order, "nse"].clip(-3), y, s=22,
                 color=[SITE_COLOR[site_of[s]] for s in order], zorder=4,
                 edgecolors="k", linewidths=0.4, label="ensemble (site colour)")
for i, s in enumerate(order):
    ax[0, 1].plot([ps_8.loc[s, "nse"].clip(-3), ps_h.loc[s, "nse"].clip(-3)], [i, i],
                  color="0.8", lw=1, zorder=2)
ax[0, 1].axvline(0, color="k", lw=0.8)
ax[0, 1].set_yticks(y, order, fontsize=7)
ax[0, 1].set(xlabel="held-out per-station NSE (clipped at -3)",
             title="(b) per-station NSE: recommended ensemble (site colour) vs model8 (grey)")
ax[0, 1].legend(fontsize=9, loc="lower right")
ax[0, 1].grid(alpha=.3)

# (c) blocked per-block NSE across candidates
cands = {
    "model6": "model6_blockcv_predictions.csv",
    "model8 (weighted)": "model8_blockcv_capacity_aridity_weighted_predictions.csv",
    "nn-hybrid (quantile)": "nn_hybrid_q_blockcv_predictions.csv",
    "ensemble (median of 5)": "nn_ens_blockcv_predictions.csv",
}
blocks = None
rows = {}
for name, f in cands.items():
    pb = per_block(load(f))["nse"]
    blocks = pb.index if blocks is None else blocks
    rows[name] = pb
x = np.arange(len(blocks))
w = 0.2
colors = ["0.7", "0.45", "#d62728", "#1f77b4"]
for i, (name, pb) in enumerate(rows.items()):
    ax[1, 0].bar(x + (i - 1.5) * w, pb.loc[blocks].clip(-1.5), w, color=colors[i], label=name)
ax[1, 0].axhline(0, color="k", lw=0.8)
ax[1, 0].set_xticks(x, blocks, fontsize=8, rotation=20)
ax[1, 0].set(ylabel="held-out block NSE (clipped at -1.5)",
             title="(c) blocked validation: per-block NSE, the honest transfer test")
ax[1, 0].legend(fontsize=8)
ax[1, 0].grid(alpha=.3, axis="y")

# (d) preprocessing attribution: blocked pooled / block-median by treatment
att = {
    "z-score": "nn_hybrid_nse_blockcv_predictions.csv",
    "quantile\n+ weights": "nn_hybrid_wq_blockcv_predictions.csv",
    "quantile": "nn_hybrid_q_blockcv_predictions.csv",
}
names = list(att)
pooled_b = [metrics(load(f)[TARGET], load(f)["pred"])["nse"] for f in att.values()]
med_b = [per_block(load(f))["nse"].median() for f in att.values()]
x = np.arange(len(names))
ax[1, 1].bar(x - 0.18, pooled_b, 0.36, color="0.45", label="pooled NSE")
ax[1, 1].bar(x + 0.18, med_b, 0.36, color="#d62728", label="block-median NSE")
for xv, v in zip(x - 0.18, pooled_b):
    ax[1, 1].text(xv, v + .006, f"{v:+.2f}", ha="center", fontsize=9)
for xv, v in zip(x + 0.18, med_b):
    ax[1, 1].text(xv, v + .006, f"{v:+.2f}", ha="center", fontsize=9)
ax[1, 1].axhline(metrics(load(cands["model8 (weighted)"])[TARGET],
                         load(cands["model8 (weighted)"])["pred"])["nse"],
                 color="k", ls="--", lw=1, label="model8 pooled (+0.32)")
ax[1, 1].set_xticks(x, names, fontsize=9)
ax[1, 1].set(title="(d) blocked skill by statics preprocessing (the hybrid)")
ax[1, 1].legend(fontsize=9)
ax[1, 1].grid(alpha=.3, axis="y")

fig.tight_layout()
fig.savefig(FIG, dpi=110)
plt.close(fig)
print(f"wrote {FIG.relative_to(REPO)}")
