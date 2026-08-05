"""Full-stack model8 station-out 4-panel results figure.

The candidate configuration from the blocked-validation experiment --
model8 + AWC capacity + aridity static + stratified weights -- under classic
leave-one-STATION-out folds, against the published model8 as baseline:
(a) LOSO fit; (b) the median-r station's held-out series; (c) per-station
NSE, published model8 -> full stack; (d) per-station bias.

Reads data/model8_loso_predictions.csv (published) and
data/model8_losocv_capacity_aridity_weighted_predictions.csv (written by
``run_blocked_cv.py m8capaw@station``); nothing re-fits.

Run from repo root::  PYTHONPATH=. python handout/plot_model8_fullstack_results.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emt.evaluation import metrics

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "model8_fullstack_results.png"
TARGET = "sm_rootzone_pct"
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728",
              "MURRUMBIDGEE": "#9467bd"}
SITES = ["ADELONG", "KYEAMBA", "MURRUMBIDGEE", "YANCO"]

tab = pd.read_csv(REPO / "data" / "process_target_2006_2010.csv")
site_of = tab.drop_duplicates("station").set_index("station")["site"]
p0 = pd.read_csv(REPO / "data" / "model8_loso_predictions.csv")
pf = pd.read_csv(REPO / "data" /
                 "model8_losocv_capacity_aridity_weighted_predictions.csv")
pf["site"] = pf["station"].map(site_of)


def per_site(p):
    return p.groupby("station").apply(lambda g: pd.Series(metrics(g[TARGET], g["pred"])),
                                      include_groups=False)


ps0, psf = per_site(p0), per_site(pf)
pooled = metrics(pf[TARGET], pf["pred"])

fig, ax = plt.subplots(2, 2, figsize=(15, 11))

# (a) LOSO fit
for s in SITES:
    d = pf[pf.site == s]
    ax[0, 0].scatter(d[TARGET], d["pred"], s=6, alpha=.25, color=SITE_COLOR[s], label=s)
lim = [0, 55]
ax[0, 0].plot(lim, lim, "k--", lw=1)
ax[0, 0].set(xlim=lim, ylim=lim, xlabel="observed root-zone (%)",
             ylabel="LOSO predicted (%)",
             title=f"(a) full-stack leave-site-out fit  "
                   f"(pooled NSE {pooled['nse']:+.2f}, r {pooled['r']:.2f})")
leg = ax[0, 0].legend(fontsize=9, markerscale=2)
for lh in leg.legend_handles:
    lh.set_alpha(1)
ax[0, 0].grid(alpha=.3)

# (b) example held-out series: the median-r station
med_stn = psf["r"].sort_values().index[len(psf) // 2]
d = pf[pf.station == med_stn].copy()
d["time"] = pd.to_datetime(d["time"])
d = d.sort_values("time")
ax[0, 1].plot(d["time"], d[TARGET], color="k", lw=.9, label="observed")
ax[0, 1].plot(d["time"], d["pred"], color=SITE_COLOR[site_of[med_stn]], lw=.9,
              label="full stack (held out)")
m = metrics(d[TARGET], d["pred"])
ax[0, 1].set(ylabel="root-zone soil moisture (%)",
             title=f"(b) Held-out {med_stn} — the median-r station "
                   f"(r {m['r']:.2f}, NSE {m['nse']:+.2f})")
ax[0, 1].legend(fontsize=9)
ax[0, 1].grid(alpha=.3)

# (c) per-station NSE, published model8 -> full stack
common = [s for s in ps0.index if s in psf.index]
comp = pd.DataFrame({"m8": ps0.loc[common, "nse"],
                     "fs": psf.loc[common, "nse"]}).clip(-3, 1).sort_values("fs")
x = np.arange(len(comp))
ax[1, 0].vlines(x, comp["m8"], comp["fs"], color="#bbb", lw=1.2, zorder=1)
ax[1, 0].scatter(x, comp["m8"], s=22, color="#bbb", label="model8 (published)", zorder=2)
ax[1, 0].scatter(x, comp["fs"], s=26,
                 color=[SITE_COLOR[site_of[s]] for s in comp.index],
                 edgecolor="k", linewidth=.3, label="full stack", zorder=3)
ax[1, 0].axhline(0, color="k", lw=.8)
ax[1, 0].set_xticks(x)
ax[1, 0].set_xticklabels(comp.index, rotation=90, fontsize=7)
ax[1, 0].set(ylim=[-3.2, 1.05], ylabel="per-station NSE",
             title=f"(c) Per-station NSE: published model8 → full stack  "
                   f"({int((ps0['nse']>0).sum())} → {int((psf['nse']>0).sum())} positive)")
ax[1, 0].legend(fontsize=8, loc="lower right")
ax[1, 0].grid(alpha=.3, axis="y")

# (d) per-station bias
b = psf.loc[common, "bias"].sort_values()
ax[1, 1].bar(range(len(b)), b.values, color=[SITE_COLOR[site_of[s]] for s in b.index])
ax[1, 1].axhline(0, color="k", lw=.8)
ax[1, 1].set_xticks(range(len(b)))
ax[1, 1].set_xticklabels(b.index, rotation=90, fontsize=7)
ax[1, 1].set(ylabel="per-station bias (pred−obs, %)",
             title=f"(d) Residual per-station bias "
                   f"(median |bias| {psf['bias'].abs().median():.1f}%)")
ax[1, 1].grid(alpha=.3, axis="y")

fig.suptitle("model8 full stack (+ AWC capacity + aridity + weights), "
             "leave-site-out over 37 stations, 2006–2010",
             y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG, dpi=130)
plt.close(fig)
print(f"wrote {FIG.relative_to(REPO)}")
