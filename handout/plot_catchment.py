"""Figures for the expanded Yanco+Kyeamba+Adelong 2006-2010 training set.

Run from repo root::  PYTHONPATH=. python handout/plot_catchment.py

Produces: between-site SMIPS distributions, the leave-site-out fit, the
feature-importance comparison against the single-cluster (Kyeamba) run, and the
residual per-station bias.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emt.features import SMIPS_COL
from emt.model import leave_site_out_cv, fit, feature_importance, TARGET, FEATURES

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "catchment_results.png"
TABLE = REPO / "data" / "train_catchment_2006_2010.csv"

SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728"}
SITES = ["ADELONG", "KYEAMBA", "YANCO"]

# Kyeamba-only (4-station) importances, for the before/after comparison.
KYEAMBA_IMP = {"elevation": .257, "northness": .243, "slope": .212, "eastness": .209,
               "doy_sin": .056, "smips_totalbucket": .006, "twi": .006,
               "accumulation": .005, "doy_cos": .005, "hli": .001}

t = pd.read_csv(TABLE)
site_of = t.drop_duplicates("station").set_index("station")["site"]

cv = leave_site_out_cv(t)
pred = cv["predictions"].copy()
pred["site"] = pred["station"].map(site_of)
per_site = cv["per_site"].copy()
per_site["site"] = per_site["station"].map(site_of)
pooled = cv["pooled"]
imp = feature_importance(fit(t))

fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# (a) SMIPS now varies across sites
data = [t.loc[t.site == s, SMIPS_COL].values for s in SITES]
bp = ax[0,0].boxplot(data, labels=SITES, patch_artist=True, showfliers=False)
for patch, s in zip(bp["boxes"], SITES):
    patch.set_facecolor(SITE_COLOR[s]); patch.set_alpha(.6)
for s, d in zip(SITES, data):
    ax[0,0].text(SITES.index(s)+1, np.median(d)+1, f"μ={d.mean():.0f}", ha="center", fontsize=9)
ax[0,0].set(ylabel="SMIPS TotalBucket (mm)",
            title="(a) SMIPS distribution by site")
ax[0,0].grid(alpha=.3, axis="y")

# (b) leave-site-out predicted vs observed
for s in SITES:
    d = pred[pred.site == s]
    ax[0,1].scatter(d[TARGET], d["pred"], s=6, color=SITE_COLOR[s], alpha=.25, label=s)
lim = [0, 55]
ax[0,1].plot(lim, lim, "k--", lw=1)
ax[0,1].set(xlim=lim, ylim=lim, xlabel="observed root-zone (%)", ylabel="LOSO predicted (%)",
            title=f"(b) Leave-site-out fit  (pooled r={pooled['r']:.2f}, r²={pooled['r2']:.2f}, "
                  f"RMSE={pooled['rmse']:.1f})")
leg = ax[0,1].legend(fontsize=9, markerscale=2);
for lh in leg.legend_handles: lh.set_alpha(1)
ax[0,1].grid(alpha=.3)

# (c) feature importance: catchment vs Kyeamba-only
order = imp.index.tolist()
y = np.arange(len(order)); h = 0.4
ax[1,0].barh(y+h/2, [imp[f] for f in order], h, color="#55a868", label="catchment (30 stn)")
ax[1,0].barh(y-h/2, [KYEAMBA_IMP.get(f,0) for f in order], h, color="#bbbbbb", label="Kyeamba-only (4 stn)")
ax[1,0].set_yticks(y); ax[1,0].set_yticklabels(order); ax[1,0].invert_yaxis()
ax[1,0].set(xlabel="importance",
            title="(c) Feature importance: catchment vs single-cluster")
ax[1,0].legend(fontsize=9); ax[1,0].grid(alpha=.3, axis="x")

# (d) per-station bias (remaining limitation)
ps = per_site.sort_values("bias")
colors = [SITE_COLOR[s] for s in ps["site"]]
ax[1,1].bar(range(len(ps)), ps["bias"], color=colors)
ax[1,1].axhline(0, color="k", lw=.8)
ax[1,1].set_xticks(range(len(ps))); ax[1,1].set_xticklabels(ps["station"], rotation=90, fontsize=7)
ax[1,1].set(ylabel="per-station bias (pred−obs, %)",
            title="(d) Residual per-station bias")
ax[1,1].grid(alpha=.3, axis="y")
handles = [plt.Rectangle((0,0),1,1,color=SITE_COLOR[s]) for s in SITES]
ax[1,1].legend(handles, SITES, fontsize=8)

fig.suptitle("Downscaling across Yanco + Kyeamba + Adelong, 2006–2010 "
             f"(n={len(t):,} station-days, 30 stations)", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG, dpi=130)
print("wrote", FIG.relative_to(REPO))
print("pooled:", {k: round(v,3) if isinstance(v,float) else v for k,v in pooled.items()})
