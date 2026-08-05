"""Blocked-CV held-out prediction vs observation, every station.

The blocked companion to plot_model8_per_station.py: each panel is one
station's observed series against its **leave-one-block-out** predictions --
the prediction made when the station's whole block (its cluster, or the
M-site itself) was held out -- for the two headline configurations:

    model8 + aridity + weights   (green, the process track's blocked best)
    model6                       (blue, the ML track)

Panels are grouped by block, blocks ordered dry -> wet by mean P/PET, so the
climate-envelope story reads left-to-right, top-to-bottom; each title carries
both models' held-out NSE. Reads data/model{6,8}_blockcv*_predictions.csv
(written by run_blocked_cv.py); nothing re-fits.

Run from repo root::  PYTHONPATH=. python handout/plot_blocked_per_station.py
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
FIG = REPO / "handout" / "figures" / "blocked_per_station.png"
TARGET = "sm_rootzone_pct"
C_M6, C_M8 = "#1f77b4", "#2ca02c"

p8 = pd.read_csv(REPO / "data" / "model8_blockcv_aridity_weighted_predictions.csv",
                 parse_dates=["time"])
p6 = pd.read_csv(REPO / "data" / "model6_blockcv_predictions.csv",
                 parse_dates=["time"])

# Blocks ordered dry -> wet (mean P/PET), stations alphabetical within a block.
clim = pd.read_csv(REPO / "data" / "process_climate_statics.csv")
clim["block"] = clim["station"].map(
    lambda s: {"Y": "YANCO", "K": "KYEAMBA", "A": "ADELONG"}.get(s[0], s))
block_ar = clim.groupby("block")["aridity"].mean()
block_of = p8.drop_duplicates("station").set_index("station")["block"]
stations = sorted(p8.station.unique(),
                  key=lambda s: (block_ar[block_of[s]], block_of[s], s))

ps8 = p8.groupby("station").apply(lambda g: pd.Series(metrics(g[TARGET], g["pred"])),
                                  include_groups=False)
ps6 = p6.groupby("station").apply(lambda g: pd.Series(metrics(g[TARGET], g["pred"])),
                                  include_groups=False)
pooled8 = metrics(p8[TARGET], p8["pred"])
pooled6 = metrics(p6[TARGET], p6["pred"])

ncols = 6
nrows = (len(stations) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.4), sharex=True)
axes = axes.ravel()
for ax, stn in zip(axes, stations):
    g8 = p8[p8.station == stn].sort_values("time")
    g6 = p6[p6.station == stn].sort_values("time")
    ax.plot(g8["time"], g8[TARGET], color="k", lw=0.7, label="observed")
    ax.plot(g6["time"], g6["pred"], color=C_M6, lw=0.8, alpha=0.9, label="model6")
    ax.plot(g8["time"], g8["pred"], color=C_M8, lw=0.9, label="model8+arid+wts")
    n8 = ps8.loc[stn, "nse"]
    n6 = ps6.loc[stn, "nse"] if stn in ps6.index else float("nan")
    ax.set_title(f"{stn} ({block_of[stn]}, P/PET {block_ar[block_of[stn]]:.2f})   "
                 f"NSE m8 {n8:+.2f} · m6 {n6:+.2f}", fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=.25)
for ax in axes[len(stations):]:
    ax.set_visible(False)
axes[0].legend(fontsize=7, loc="upper left")
fig.suptitle("Leave-one-block-out prediction vs observation, all "
             f"{len(stations)} stations, blocks ordered dry → wet.  "
             f"Pooled NSE: model8+arid+wts {pooled8['nse']:+.2f}, "
             f"model6 {pooled6['nse']:+.2f}", fontsize=13, y=0.995)
fig.supylabel("root-zone soil moisture (%)", fontsize=10)
fig.tight_layout(rect=[0.01, 0, 1, 0.985])
fig.savefig(FIG, dpi=110)
plt.close(fig)
print(f"wrote {FIG.relative_to(REPO)}")
