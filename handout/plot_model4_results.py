"""Figures for model4 (the improved model) on the catchment training set.

Produces:
  figures/model4_results.png       -- LOSO fit, permutation importance, and the
                                      paired model1 -> model4 per-station NSE and
                                      bias comparisons
  figures/model4_per_station.png   -- 30-panel held-out time series

Run from repo root::  PYTHONPATH=. python handout/plot_model4_results.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from emt.model4 import model as m4
from emt.model1 import model as m1

REPO = Path(__file__).resolve().parent.parent
FIGDIR = REPO / "handout" / "figures"
TABLE = REPO / "data" / "train_catchment_2006_2010.csv"
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728"}
SITES = ["ADELONG", "KYEAMBA", "YANCO"]
TARGET = m4.TARGET

tab = pd.read_csv(TABLE)
site_of = tab.drop_duplicates("station").set_index("station")["site"]

print("model4 LOSO ...", flush=True)
cv4 = m4.leave_site_out_cv(tab)
print("model1 LOSO (reference) ...", flush=True)
cv1 = m1.leave_site_out_cv(tab)

pred = cv4["predictions"].copy()
pred["site"] = pred["station"].map(site_of)
pred["time"] = pd.to_datetime(pred["time"])
ps4 = cv4["per_site"].set_index("station")
ps1 = cv1["per_site"].set_index("station")
pooled4, pooled1 = cv4["pooled"], cv1["pooled"]

print("permutation importance ...", flush=True)
imp = m4.feature_importance(m4.fit(tab), tab)

# ---------------------------------------------------------------- results fig
fig, ax = plt.subplots(2, 2, figsize=(15, 11))

# (a) leave-site-out predicted vs observed
for s in SITES:
    d = pred[pred.site == s]
    ax[0,0].scatter(d[TARGET], d["pred"], s=6, color=SITE_COLOR[s], alpha=.25, label=s)
lim = [0, 55]
ax[0,0].plot(lim, lim, "k--", lw=1)
ax[0,0].set(xlim=lim, ylim=lim, xlabel="observed root-zone (%)",
            ylabel="LOSO predicted (%)",
            title=f"(a) model4 leave-site-out fit  (pooled r={pooled4['r']:.2f}, "
                  f"NSE={pooled4['nse']:.2f}; model1: r={pooled1['r']:.2f}, "
                  f"NSE={pooled1['nse']:.2f})")
leg = ax[0,0].legend(fontsize=9, markerscale=2)
for lh in leg.legend_handles: lh.set_alpha(1)
ax[0,0].grid(alpha=.3)

# (b) permutation feature importance, climatology + soil highlighted
GROUP_COLOR = {}
for f in imp.index:
    if f.startswith("smips_") and f != "smips_totalbucket":
        GROUP_COLOR[f] = "#d62728"      # climatology (new)
    elif f.startswith("soil_"):
        GROUP_COLOR[f] = "#9467bd"      # soil (new)
    else:
        GROUP_COLOR[f] = "#55a868"      # base
y = np.arange(len(imp))
ax[0,1].barh(y, imp.values, color=[GROUP_COLOR[f] for f in imp.index])
ax[0,1].set_yticks(y); ax[0,1].set_yticklabels(imp.index, fontsize=8)
ax[0,1].invert_yaxis()
handles = [plt.Rectangle((0,0),1,1,color=c) for c in ["#55a868", "#d62728", "#9467bd"]]
ax[0,1].legend(handles, ["base (models 1-3)", "SMIPS climatology (new)", "SLGA soil (new)"],
               fontsize=8)
ax[0,1].set(xlabel="permutation importance",
            title="(b) model4 feature importance")
ax[0,1].grid(alpha=.3, axis="x")

# (c) per-station NSE: model1 -> model4 (dumbbell)
common = [s for s in ps1.index if s in ps4.index]
comp = pd.DataFrame({"m1": ps1.loc[common, "nse"], "m4": ps4.loc[common, "nse"]},
                    index=common).sort_values("m4")
CLIP = -3.0
x = np.arange(len(comp))
m1c, m4c = comp["m1"].clip(lower=CLIP), comp["m4"].clip(lower=CLIP)
ax[1,0].vlines(x, m1c, m4c, color="#bbbbbb", lw=1.5, zorder=1)
ax[1,0].scatter(x, m1c, s=25, color="#bbbbbb", label="model1", zorder=2)
ax[1,0].scatter(x, m4c, s=30,
                color=[SITE_COLOR[site_of[s]] for s in comp.index],
                edgecolor="k", linewidth=.4, label="model4", zorder=3)
for xi, (s, row) in zip(x, comp.iterrows()):    # mark clipped model1 values
    if row["m1"] < CLIP:
        ax[1,0].annotate(f"{row['m1']:.1f}", (xi, CLIP), fontsize=6,
                         ha="center", va="top", xytext=(0, -8),
                         textcoords="offset points", color="#888888")
ax[1,0].axhline(0, color="k", lw=.8)
ax[1,0].set_xticks(x); ax[1,0].set_xticklabels(comp.index, rotation=90, fontsize=7)
n4, n1 = int((comp["m4"] > 0).sum()), int((comp["m1"] > 0).sum())
ax[1,0].set(ylim=[CLIP - 0.35, 1.05], ylabel="per-station NSE",
            title=f"(c) Per-station NSE: model1 → model4  "
                  f"({n1}/30 → {n4}/30 positive; "
                  f"median {ps1['nse'].median():.2f} → {ps4['nse'].median():.2f})")
ax[1,0].legend(fontsize=8, loc="lower right")
ax[1,0].grid(alpha=.3, axis="y")

# (d) per-station bias: model1 vs model4
bcomp = pd.DataFrame({"m1": ps1.loc[common, "bias"], "m4": ps4.loc[common, "bias"]},
                     index=common).sort_values("m1")
xb = np.arange(len(bcomp)); h = 0.4
ax[1,1].bar(xb - h/2, bcomp["m1"], h, color="#bbbbbb", label="model1")
ax[1,1].bar(xb + h/2, bcomp["m4"], h,
            color=[SITE_COLOR[site_of[s]] for s in bcomp.index], label="model4")
ax[1,1].axhline(0, color="k", lw=.8)
ax[1,1].set_xticks(xb); ax[1,1].set_xticklabels(bcomp.index, rotation=90, fontsize=7)
ax[1,1].set(ylabel="per-station bias (pred−obs, %)",
            title=f"(d) Residual per-station bias  (mean |bias| "
                  f"{ps1['bias'].abs().mean():.1f}% → {ps4['bias'].abs().mean():.1f}%)")
ax[1,1].legend(fontsize=8)
ax[1,1].grid(alpha=.3, axis="y")

fig.suptitle("model4 (regularised boosting + SMIPS climatology + soil), "
             f"leave-site-out over 30 stations, 2006–2010 (n={len(tab):,})",
             y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIGDIR / "model4_results.png", dpi=130)
plt.close(fig)
print("wrote", (FIGDIR / "model4_results.png").relative_to(REPO), flush=True)

# ---------------------------------------------------------- per-station grid
stations = sorted(tab.station.unique(),
                  key=lambda s: (SITES.index(site_of[s]), s))
ncols, nrows = 6, (len(stations) + 5) // 6
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.4), sharex=True)
axes = axes.ravel()
for axp, stn in zip(axes, stations):
    g = pred[pred.station == stn].sort_values("time")
    c = SITE_COLOR[site_of[stn]]
    axp.plot(g["time"], g[TARGET], color="k", lw=0.7, label="observed")
    axp.plot(g["time"], g["pred"], color=c, lw=0.9, label="LOSO prediction")
    m = ps4.loc[stn]
    axp.set_title(f"{stn}   r={m.r:.2f}  NSE={m.nse:.2f}", fontsize=9)
    axp.tick_params(labelsize=7)
    axp.xaxis.set_major_locator(mdates.YearLocator())
    axp.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axp.grid(alpha=.25)
for axp in axes[len(stations):]:
    axp.set_visible(False)
axes[0].legend(fontsize=7, loc="upper left")
fig.suptitle(f"model4 leave-site-out prediction vs observation, all {len(stations)} "
             f"stations (2006-2010).  Pooled NSE = {pooled4['nse']:+.2f}  "
             f"(model1: {pooled1['nse']:+.2f})", fontsize=13, y=0.995)
fig.supylabel("root-zone soil moisture (%)", fontsize=10)
fig.tight_layout(rect=[0.01, 0, 1, 0.985])
fig.savefig(FIGDIR / "model4_per_station.png", dpi=110)
plt.close(fig)
print("wrote", (FIGDIR / "model4_per_station.png").relative_to(REPO), flush=True)

nse = ps4["nse"].sort_values()
print(f"model4 per-station NSE: min={nse.min():.2f} median={nse.median():.2f} "
      f"max={nse.max():.2f}; positive {int((nse>0).sum())}/{len(nse)}")
