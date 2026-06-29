"""Per-station leave-site-out time series for the catchment model.

Each panel shows one held-out station's observed vs predicted root-zone moisture
over the full record, labelled with that station's r and NSE. The pooled NSE
(over all station-days) is the headline cross-validated skill and is stated in
the title, to distinguish it from the single-date spatial snapshot in
downscale_yanco.png.

Run from repo root::  PYTHONPATH=. python handout/plot_per_station.py
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from emt.model import leave_site_out_cv, TARGET
from emt.features import SMIPS_COL  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
FIGDIR = REPO / "handout" / "figures"
TABLE = REPO / "data" / "train_catchment_2006_2010.csv"
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728"}

tab = pd.read_csv(TABLE)
tab["time"] = pd.to_datetime(tab["time"])
site_of = tab.drop_duplicates("station").set_index("station")["site"]

cv = leave_site_out_cv(tab)
pred = cv["predictions"].copy()
pred["time"] = pd.to_datetime(pred["time"])
ps = cv["per_site"].set_index("station")
pooled = cv["pooled"]


def order(stations):
    sites = ["ADELONG", "KYEAMBA", "YANCO"]
    return sorted(stations, key=lambda s: (sites.index(site_of[s]), s))


def grid(stations, fname, suptitle, ncols):
    stations = order(stations)
    nrows = (len(stations) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.4),
                             sharex=True)
    axes = axes.ravel()
    for ax, stn in zip(axes, stations):
        g = pred[pred.station == stn].sort_values("time")
        c = SITE_COLOR[site_of[stn]]
        ax.plot(g["time"], g[TARGET], color="k", lw=0.7, label="observed")
        ax.plot(g["time"], g["pred"], color=c, lw=0.9, label="LOSO prediction")
        m = ps.loc[stn]
        ax.set_title(f"{stn}   r={m.r:.2f}  NSE={m.nse:.2f}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=.25)
    for ax in axes[len(stations):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(suptitle, fontsize=13, y=0.995)
    fig.supylabel("root-zone soil moisture (%)", fontsize=10)
    fig.tight_layout(rect=[0.01, 0, 1, 0.985])
    fig.savefig(FIGDIR / fname, dpi=110)
    plt.close(fig)
    print("wrote", (FIGDIR / fname).relative_to(REPO))


all30 = list(tab.station.unique())
grid(all30, "catchment_per_station.png", ncols=6,
     suptitle=(f"Leave-site-out prediction vs observation, all {len(all30)} stations "
               f"(2006-2010).  Pooled NSE = {pooled['nse']:+.2f}  "
               f"(headline cross-validated skill; panel titles are per-station)"))

kyeamba = [s for s in all30 if site_of[s] == "KYEAMBA"]
grid(kyeamba, "kyeamba_per_station.png", ncols=4,
     suptitle=(f"Leave-site-out prediction vs observation, {len(kyeamba)} Kyeamba "
               f"stations (2006-2010).  Pooled NSE = {pooled['nse']:+.2f}"))

# Per-station NSE distribution, for reference.
nse = ps["nse"].sort_values()
print(f"per-station NSE: min={nse.min():.2f}, median={nse.median():.2f}, "
      f"max={nse.max():.2f}; positive at {int((nse>0).sum())}/{len(nse)} stations")
print(f"per-station r:   median={ps['r'].median():.2f}; "
      f"ubRMSE median={ps['ubrmse'].median():.2f}%")
print(f"pooled: NSE={pooled['nse']:+.3f}, r={pooled['r']:.3f}, "
      f"ubRMSE={pooled['ubrmse']:.2f}%")
