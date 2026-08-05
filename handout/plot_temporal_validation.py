"""Temporal-validation figure: which kind of generalisation is actually hard.

(a) Skill of each held-out year against that year's rainfall. Four of the five
years score alike; 2010 — the drought-breaking wet year, at nearly four times
2006's rainfall — is the one that breaks, and it breaks by running too dry.
(b) The same models under four fold designs of increasing strictness (year /
station / block / block x year), showing that transfer across *time* is easier
than transfer across *space*, and that the strict double hold-out is the floor.

Reads data/model{6,8,9}_yearcv*_predictions.csv plus the blockcv/losocv tables
written by run_blocked_cv.py, and the forcing store for annual rainfall.
Nothing re-fits.

Run from repo root::  PYTHONPATH=. python handout/plot_temporal_validation.py
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
FIG = REPO / "handout" / "figures" / "temporal_validation.png"
TARGET = "sm_rootzone_pct"
C = {"model6": "#1f77b4", "model8": "#2ca02c", "model9": "#B07A1E"}

YEAR = {"model6": "model6_yearcv_predictions.csv",
        "model8": "model8_yearcv_capacity_aridity_weighted_predictions.csv",
        "model9": "model9_yearcv_predictions.csv"}
LADDER = {
    "model6": {"year": YEAR["model6"], "station": None,
               "block": "model6_blockcv_predictions.csv",
               "blockyear": "model6_blockyearcv_predictions.csv"},
    "model8": {"year": YEAR["model8"],
               "station": "model8_losocv_capacity_aridity_weighted_predictions.csv",
               "block": "model8_blockcv_capacity_aridity_weighted_predictions.csv",
               "blockyear": "model8_blockyearcv_capacity_aridity_weighted_predictions.csv"},
    "model9": {"year": YEAR["model9"], "station": "model9_losocv_predictions.csv",
               "block": "model9_blockcv_predictions.csv",
               "blockyear": "model9_blockyearcv_predictions.csv"},
}


def load(name):
    p = REPO / "data" / name
    return pd.read_csv(p) if p.exists() else None


f = pd.read_csv(REPO / "data" / "process_forcing_2005_2010.csv", parse_dates=["time"])
f["year"] = f["time"].dt.year
rain = (f.groupby("year")["daily_rain"].sum()
        / f.groupby("year")["station"].nunique())

fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), width_ratios=[1.25, 1])

# (a) per-year skill vs that year's rainfall
for m, fn in YEAR.items():
    d = load(fn)
    if d is None:
        continue
    per = {y: metrics(g[TARGET], g["pred"])["nse"] for y, g in d.groupby("year")}
    ys = sorted(per, key=lambda y: rain[y])      # order along the x-axis, not by year
    nse = [per[y] for y in ys]
    ax[0].plot([rain[y] for y in ys], nse, "o-", color=C[m], lw=1.6, ms=7,
               label=m, zorder=3)
    for y, n in zip(ys, nse):
        if y in (2010, 2006):
            ax[0].annotate(str(y), (rain[y], n), textcoords="offset points",
                           xytext=(0, -16), ha="center", fontsize=9,
                           color=C[m], fontweight="bold")
ax[0].axhline(0, color="k", lw=.8)
ax[0].set(xlabel="rainfall in the held-out year (mm, per-station mean)",
          ylabel="NSE of the held-out year",
          title="(a) Only the wet extreme breaks — 2010, the drought-breaking year")
ax[0].legend(fontsize=9, loc="lower left")
ax[0].grid(alpha=.3)

# (b) the harness ladder
designs = ["year", "station", "block", "blockyear"]
labels = {"year": "YEAR\n(new weather,\nknown sites)",
          "station": "STATION\n(new pixel,\nneighbours known)",
          "block": "BLOCK\n(new district)",
          "blockyear": "BLOCK × YEAR\n(new district,\nnew regime)"}
w, xs = 0.26, np.arange(len(designs))
for i, (m, srcs) in enumerate(LADDER.items()):
    vals, pos = [], []
    for j, dsg in enumerate(designs):
        d = load(srcs[dsg]) if srcs[dsg] else None
        if d is None:
            # model6 has no station-out run on the 37-station table; mark the
            # gap rather than letting the bars silently close up.
            ax[1].annotate("not run", (j + (i - 1) * w, 0.015), ha="center",
                           fontsize=7.5, rotation=90, color="#8AA0A7")
            continue
        vals.append(metrics(d[TARGET], d["pred"])["nse"])
        pos.append(j + (i - 1) * w)
    ax[1].bar(pos, vals, width=w - .03, color=C[m], label=m, zorder=2)
    for p, v in zip(pos, vals):
        ax[1].annotate(f"{v:+.2f}", (p, v), textcoords="offset points",
                       xytext=(0, 3), ha="center", fontsize=8)
ax[1].set_xticks(xs)
ax[1].set_xticklabels([labels[d] for d in designs], fontsize=7.8)
ax[1].set(ylabel="pooled NSE", ylim=[0, 0.88],
          title="(b) Each rung removes more of what the model could lean on")
ax[1].legend(fontsize=9, loc="upper right")
ax[1].grid(alpha=.3, axis="y", zorder=0)

fig.suptitle("Temporal validation: holding out whole years, 2006–2010",
             y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG, dpi=130)
plt.close(fig)
print(f"wrote {FIG.relative_to(REPO)}")
