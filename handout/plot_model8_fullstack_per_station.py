"""Full-stack model8 station-out held-out series, every station.

Each panel: one station's observed series against its leave-one-STATION-out
predictions from the published model8 (gray) and the full stack
(model8 + AWC capacity + aridity + weights; site colour). Titles carry both
NSEs. Reads data/model8_loso_predictions.csv and
data/model8_losocv_capacity_aridity_weighted_predictions.csv; nothing re-fits.

Run from repo root::  PYTHONPATH=. python handout/plot_model8_fullstack_per_station.py
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from emt.evaluation import metrics

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "model8_fullstack_per_station.png"
TARGET = "sm_rootzone_pct"
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728",
              "MURRUMBIDGEE": "#9467bd"}
SITES = ["ADELONG", "KYEAMBA", "MURRUMBIDGEE", "YANCO"]

tab = pd.read_csv(REPO / "data" / "process_target_2006_2010.csv")
site_of = tab.drop_duplicates("station").set_index("station")["site"]
p0 = pd.read_csv(REPO / "data" / "model8_loso_predictions.csv", parse_dates=["time"])
pf = pd.read_csv(REPO / "data" /
                 "model8_losocv_capacity_aridity_weighted_predictions.csv",
                 parse_dates=["time"])

pooled = metrics(pf[TARGET], pf["pred"])
ps0 = p0.groupby("station").apply(lambda g: pd.Series(metrics(g[TARGET], g["pred"])),
                                  include_groups=False)
psf = pf.groupby("station").apply(lambda g: pd.Series(metrics(g[TARGET], g["pred"])),
                                  include_groups=False)

stations = sorted(pf.station.unique(), key=lambda s: (SITES.index(site_of[s]), s))
ncols = 6
nrows = (len(stations) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.4), sharex=True)
axes = axes.ravel()
for ax, stn in zip(axes, stations):
    gf = pf[pf.station == stn].sort_values("time")
    g0 = p0[p0.station == stn].sort_values("time")
    c = SITE_COLOR[site_of[stn]]
    ax.plot(gf["time"], gf[TARGET], color="k", lw=0.7, label="observed")
    ax.plot(g0["time"], g0["pred"], color="#aaa", lw=0.8, label="model8 (published)")
    ax.plot(gf["time"], gf["pred"], color=c, lw=0.9, label="full stack")
    ax.set_title(f"{stn}   NSE {psf.loc[stn,'nse']:+.2f} "
                 f"(was {ps0.loc[stn,'nse']:+.2f})   bias {psf.loc[stn,'bias']:+.1f}",
                 fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=.25)
for ax in axes[len(stations):]:
    ax.set_visible(False)
axes[0].legend(fontsize=7, loc="upper left")
fig.suptitle("model8 full stack, leave-site-out prediction vs observation, all "
             f"{len(stations)} stations (2006–2010).  Pooled NSE {pooled['nse']:+.2f}, "
             f"r {pooled['r']:.2f}  (gray = published model8)", fontsize=13, y=0.995)
fig.supylabel("root-zone soil moisture (%)", fontsize=10)
fig.tight_layout(rect=[0.01, 0, 1, 0.985])
fig.savefig(FIG, dpi=110)
plt.close(fig)
print(f"wrote {FIG.relative_to(REPO)}")
