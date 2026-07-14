"""model6 held-out per-station series, with cached predictions.

Runs the leave-site-out CV **once**, printing per-station progress and saving the
out-of-fold predictions to data/model6_loso_predictions.csv; if that cache exists
it is reused (no refit). This avoids re-running the ~10-min LOSO every time the
figure is drawn.

Run from repo root::  PYTHONPATH=. python handout/plot_model6_results.py
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from emt.model6 import model as m6
from emt.evaluation import metrics

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "model6_per_station.png"
TABLE = REPO / "data" / "train_catchment_plus_m_2006_2010.csv"
CACHE = REPO / "data" / "model6_loso_predictions.csv"
TARGET = m6.TARGET
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728",
              "MURRUMBIDGEE": "#9467bd"}
SITES = ["ADELONG", "KYEAMBA", "MURRUMBIDGEE", "YANCO"]
t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

tab = pd.read_csv(TABLE)
site_of = tab.drop_duplicates("station").set_index("station")["site"]

if CACHE.exists():
    stamp(f"loading cached LOSO predictions from {CACHE.name}")
    pred = pd.read_csv(CACHE, parse_dates=["time"])
else:
    stamp("computing model6 leave-site-out (once) ...")
    sub = m6.ensure_features(tab).dropna(subset=m6.FEATURES + [TARGET]).reset_index(drop=True)
    sub["time"] = pd.to_datetime(sub["time"])
    stations = list(sub["station"].unique())
    out = sub[["station", "time", TARGET]].copy()
    out["pred"] = np.nan
    for i, stn in enumerate(stations, 1):
        te = (sub["station"] == stn).values
        est = m6.build_estimator()
        est.fit(sub.loc[~te, m6.FEATURES], sub.loc[~te, TARGET])
        out.loc[te, "pred"] = est.predict(sub.loc[te, m6.FEATURES])
        m = metrics(out.loc[te, TARGET], out.loc[te, "pred"])
        stamp(f"  [{i}/{len(stations)}] {stn}: NSE={m['nse']:+.2f} r={m['r']:.2f}")
    pred = out
    pred.to_csv(CACHE, index=False)
    stamp(f"saved predictions -> {CACHE.name}")

pooled = metrics(pred[TARGET], pred["pred"])
ps = (pred.groupby("station").apply(lambda g: pd.Series(metrics(g[TARGET], g["pred"])),
                                    include_groups=False))
stamp(f"pooled NSE={pooled['nse']:+.3f} r={pooled['r']:.3f}; "
      f"per-station >0 {int((ps['nse']>0).sum())}/{len(ps)} median {ps['nse'].median():+.2f}")

stations = sorted(pred.station.unique(), key=lambda s: (SITES.index(site_of[s]), s))
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
             f"(honest, leak-free; purple = regional M-sites)", fontsize=13, y=0.995)
fig.supylabel("root-zone soil moisture (%)", fontsize=10)
fig.tight_layout(rect=[0.01, 0, 1, 0.985])
fig.savefig(FIG, dpi=110)
plt.close(fig)
stamp(f"wrote {FIG.relative_to(REPO)}")
