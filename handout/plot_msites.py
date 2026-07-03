"""Figure for the regional-site extension (30 -> 36 stations).

Adds the six scattered Murrumbidgee M-sites to the training set and shows the
leave-site-out effect. Three panels:
  (a) catchment map of all 36 stations, coloured by held-out per-station NSE
      (the spatial skill map), M-sites drawn as stars;
  (b) the original 30 stations' per-station NSE, original vs M-augmented
      training (dumbbell);
  (c) predicted vs observed for the six held-out M-sites.
Also writes msites_timeseries.png: the held-out temporal series for each M-site.

Run from repo root::  PYTHONPATH=. python handout/plot_msites.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emt.model4 import model as m4

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "msites_extension.png"
CATCH = REPO / "data" / "train_catchment_2006_2010.csv"
ALL = REPO / "data" / "train_catchment_plus_m_2006_2010.csv"
TARGET = m4.TARGET
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728",
              "MURRUMBIDGEE": "#9467bd"}

allt = pd.read_csv(ALL)
loc = allt.drop_duplicates("station").set_index("station")[["lat", "lon", "site"]]

print("model4 LOSO: 36-station ...", flush=True)
cv36 = m4.leave_site_out_cv(allt)
print("model4 LOSO: 30-station ...", flush=True)
cv30 = m4.leave_site_out_cv(pd.read_csv(CATCH))
ps36 = cv36["per_site"].set_index("station")
ps30 = cv30["per_site"].set_index("station")
pred = cv36["predictions"].copy()
p36, p30 = cv36["pooled"], cv30["pooled"]

fig = plt.figure(figsize=(16, 6.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.15, 1.0])

# (a) spatial skill map -------------------------------------------------------
axm = fig.add_subplot(gs[0, 0])
nse = ps36["nse"].clip(-1, 1)
for stn, row in loc.iterrows():
    if stn not in ps36.index:
        continue
    star = row["site"] == "MURRUMBIDGEE"
    axm.scatter(row["lon"], row["lat"], c=[nse[stn]], cmap="RdYlGn",
                vmin=-1, vmax=1, s=190 if star else 80,
                marker="*" if star else "o",
                edgecolor="k", linewidth=.6, zorder=3)
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(-1, 1))
fig.colorbar(sm, ax=axm, shrink=.8, label="held-out per-station NSE")
axm.set(xlabel="longitude", ylabel="latitude",
        title=f"(a) Spatial skill, 36 stations\n(stars = regional M-sites; "
              f"pooled NSE {p36['nse']:.2f})")
axm.grid(alpha=.3)

# (b) original-30 dumbbell: original vs augmented training ---------------------
axd = fig.add_subplot(gs[0, 1])
common = [s for s in ps30.index if s in ps36.index]
comp = pd.DataFrame({"orig": ps30.loc[common, "nse"],
                     "aug": ps36.loc[common, "nse"]}).clip(-3, 1).sort_values("aug")
y = np.arange(len(comp))
axd.hlines(y, comp["orig"], comp["aug"], color="#bbb", lw=1.2, zorder=1)
axd.scatter(comp["orig"], y, s=22, color="#bbb", label="trained on 30", zorder=2)
axd.scatter(comp["aug"], y, s=26, color="#9467bd", edgecolor="k", linewidth=.3,
            label="trained on 36 (+M-sites)", zorder=3)
axd.axvline(0, color="k", lw=.8)
axd.set_yticks(y); axd.set_yticklabels(comp.index, fontsize=6)
axd.set(xlabel="per-station NSE",
        title=f"(b) Original 30 stations\nmedian {ps30.loc[common,'nse'].median():.2f} "
              f"→ {ps36.loc[common,'nse'].median():.2f}")
axd.legend(fontsize=8, loc="lower right"); axd.grid(alpha=.3, axis="x")

# (c) M-sites predicted vs observed -------------------------------------------
axs = fig.add_subplot(gs[0, 2])
msite = sorted([s for s in ps36.index if s.startswith("M")])
for stn in msite:
    g = pred[pred.station == stn]
    axs.scatter(g[TARGET], g["pred"], s=8, alpha=.35, color="#9467bd")
    m = ps36.loc[stn]
    xm, ym = g[TARGET].mean(), g["pred"].mean()
    axs.annotate(f"{stn} (NSE {m['nse']:.2f})", (xm, ym), fontsize=7,
                 xytext=(4, 4), textcoords="offset points")
lim = [5, 45]
axs.plot(lim, lim, "k--", lw=1)
axs.set(xlim=lim, ylim=lim, xlabel="observed (%)", ylabel="held-out predicted (%)",
        title=f"(c) The 6 new M-sites (held out)\n"
              f"{int((ps36.loc[msite,'nse']>0).sum())}/6 positive NSE")
axs.grid(alpha=.3)

fig.suptitle("Extending coverage: +6 regional Murrumbidgee sites (30 → 36 stations)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(FIG, dpi=130, bbox_inches="tight")
print("wrote", FIG.relative_to(REPO), flush=True)

# ---- temporal grid: held-out series for each M-site --------------------------
import matplotlib.dates as mdates
FIG_TS = REPO / "handout" / "figures" / "msites_timeseries.png"
pred_ts = pred.copy()
pred_ts["time"] = pd.to_datetime(pred_ts["time"])
figt, axes = plt.subplots(2, 3, figsize=(15, 6.2), sharex=True)
axes = axes.ravel()
for ax, stn in zip(axes, msite):
    g = pred_ts[pred_ts.station == stn].sort_values("time")
    ax.plot(g["time"], g[TARGET], color="k", lw=0.7, label="observed")
    ax.plot(g["time"], g["pred"], color="#9467bd", lw=0.9, label="held-out prediction")
    m = ps36.loc[stn]
    ax.set_title(f"{stn}   r={m.r:.2f}  NSE={m.nse:.2f}  bias={m.bias:+.1f}", fontsize=10)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=.25)
for ax in axes[len(msite):]:
    ax.set_visible(False)
axes[0].legend(fontsize=8, loc="upper left")
figt.suptitle("Regional M-sites: held-out prediction vs observation, 2006–2010 "
              "(each site trained on the other 35, never on itself)", fontsize=12, y=1.0)
figt.supylabel("root-zone soil moisture (%)", fontsize=10)
figt.tight_layout(rect=[0.01, 0, 1, 0.97])
figt.savefig(FIG_TS, dpi=120)
print("wrote", FIG_TS.relative_to(REPO), flush=True)
print(f"30-station pooled NSE={p30['nse']:.3f} ({int((ps30['nse']>0).sum())}/30>0) | "
      f"36-station pooled NSE={p36['nse']:.3f} ({int((ps36['nse']>0).sum())}/36>0)", flush=True)
