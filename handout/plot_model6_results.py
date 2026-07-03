"""model6 held-out temporal series for all 36 stations.

Mirrors plot_model4_results.py's per-station grid, for the recommended model on
the 36-station table (catchment + regional M-sites). Each panel is one held-out
station's observed vs leave-site-out prediction over 2006-2010, titled with its
r, NSE and bias.

Run from repo root::  PYTHONPATH=. python handout/plot_model6_results.py
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from emt.model6 import model as m6

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "model6_per_station.png"
TABLE = REPO / "data" / "train_catchment_plus_m_2006_2010.csv"
TARGET = m6.TARGET
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728",
              "MURRUMBIDGEE": "#9467bd"}
SITES = ["ADELONG", "KYEAMBA", "MURRUMBIDGEE", "YANCO"]

tab = pd.read_csv(TABLE)
site_of = tab.drop_duplicates("station").set_index("station")["site"]

print("model6 LOSO (36 stations) ...", flush=True)
cv = m6.leave_site_out_cv(tab)
pred = cv["predictions"].copy()
pred["time"] = pd.to_datetime(pred["time"])
ps = cv["per_site"].set_index("station")
pooled = cv["pooled"]

stations = sorted(tab.station.unique(), key=lambda s: (SITES.index(site_of[s]), s))
ncols = 6
nrows = (len(stations) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.4), sharex=True)
axes = axes.ravel()
for ax, stn in zip(axes, stations):
    g = pred[pred.station == stn].sort_values("time")
    c = SITE_COLOR[site_of[stn]]
    ax.plot(g["time"], g[TARGET], color="k", lw=0.7, label="observed")
    ax.plot(g["time"], g["pred"], color=c, lw=0.9, label="LOSO prediction")
    m = ps.loc[stn]
    ax.set_title(f"{stn}   r={m.r:.2f}  NSE={m.nse:.2f}  bias={m.bias:+.1f}", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=.25)
for ax in axes[len(stations):]:
    ax.set_visible(False)
axes[0].legend(fontsize=7, loc="upper left")
fig.suptitle(f"model6 leave-site-out prediction vs observation, all {len(stations)} "
             f"stations (2006-2010).  Pooled NSE = {pooled['nse']:+.2f}, r = {pooled['r']:.2f}  "
             f"(purple = regional M-sites)", fontsize=13, y=0.995)
fig.supylabel("root-zone soil moisture (%)", fontsize=10)
fig.tight_layout(rect=[0.01, 0, 1, 0.985])
fig.savefig(FIG, dpi=110)
plt.close(fig)
print("wrote", FIG.relative_to(REPO), flush=True)
print(f"pooled NSE={pooled['nse']:.3f} r={pooled['r']:.3f}; "
      f"per-station >0: {int((ps['nse']>0).sum())}/{len(ps)} "
      f"median |bias| {ps['bias'].abs().median():.2f}", flush=True)
