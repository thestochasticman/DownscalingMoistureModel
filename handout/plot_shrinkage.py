"""Diagnose the per-station bias as shrinkage toward the training mean.

A Random Forest predicts by averaging training samples, so it cannot extrapolate
and compresses predictions toward the central tendency. If that is the cause of
the per-station bias, then per-station bias (pred-obs) should be negatively
correlated with each station's own mean moisture (dry stations over-predicted,
wet under-predicted) and the predicted station means should span a narrower range
than observed. Sample-size dependence is also checked (sampling imbalance).

Run from repo root::  PYTHONPATH=. python handout/plot_shrinkage.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emt.model1.model import leave_site_out_cv, TARGET

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "shrinkage_diagnostic.png"
TABLE = REPO / "data" / "train_catchment_2006_2010.csv"
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728"}

tab = pd.read_csv(TABLE)
site_of = tab.drop_duplicates("station").set_index("station")["site"]
gmean = tab[TARGET].mean()

pred = leave_site_out_cv(tab)["predictions"]
g = (pred.groupby("station")
     .apply(lambda d: pd.Series({"bias": (d["pred"] - d[TARGET]).mean(),
                                 "obs_mean": d[TARGET].mean(),
                                 "pred_mean": d["pred"].mean(), "n": len(d)}),
            include_groups=False)
     .reset_index())
g["site"] = g["station"].map(site_of)

slope = float(np.polyfit(g.obs_mean, g.bias, 1)[0])
print(f"global mean target          = {gmean:.2f}%")
print(f"observed station-mean range = {g.obs_mean.max()-g.obs_mean.min():.1f}%")
print(f"predicted station-mean range= {g.pred_mean.max()-g.pred_mean.min():.1f}% (shrunk if smaller)")
print(f"corr(bias, station mean)    = {g.bias.corr(g.obs_mean):+.2f}")
print(f"corr(bias, record length)   = {g.bias.corr(g.n):+.2f}")
print(f"shrinkage slope             = {slope:+.2f} (-1 = full collapse to global mean)")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for s, c in SITE_COLOR.items():
    d = g[g.site == s]
    ax[0].scatter(d.obs_mean, d.bias, s=40 + d.n / 15, color=c, label=s, alpha=.8,
                  edgecolor="k", lw=.5)
xs = np.linspace(g.obs_mean.min(), g.obs_mean.max(), 50)
b1, b0 = np.polyfit(g.obs_mean, g.bias, 1)
ax[0].plot(xs, b1 * xs + b0, "k--", lw=1, label=f"slope = {b1:.2f}")
ax[0].axhline(0, color="k", lw=.8); ax[0].axvline(gmean, color="grey", ls=":", lw=1)
ax[0].set(xlabel="station mean observed (%)", ylabel="station bias (pred − obs, %)",
          title="(a) Bias vs station wetness (downward slope = shrinkage)")
ax[0].legend(fontsize=8, title="marker size = record length"); ax[0].grid(alpha=.3)

ax[1].scatter(g.obs_mean, g.pred_mean, s=45, c=[SITE_COLOR[s] for s in g.site],
              edgecolor="k", lw=.5)
lim = [g.obs_mean.min() - 1, g.obs_mean.max() + 1]
ax[1].plot(lim, lim, "k--", lw=1)
ax[1].set(xlim=lim, ylim=lim, xlabel="station mean observed (%)",
          ylabel="station mean predicted (%)",
          title="(b) Station means: flatter than 1:1 = shrinkage")
ax[1].grid(alpha=.3)
fig.suptitle("Per-station bias is shrinkage toward the training mean "
             f"(slope {slope:.2f}, predicted spread {g.pred_mean.max()-g.pred_mean.min():.0f}% "
             f"vs observed {g.obs_mean.max()-g.obs_mean.min():.0f}%)", y=1.0, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG, dpi=130)
print("wrote", FIG.relative_to(REPO))
