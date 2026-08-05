"""Single-message figures for presentation, extracted from the cached results.

The handout's four-panel result figures are records, not slides. These are one
claim each, sized to be read from the back of a room:

  slide_rhythm_vs_level.png   one held-out station: the shape is right, the
                              level is not -- the whole problem in one panel
  slide_r_vs_nse.png          every station's correlation against its NSE:
                              dynamics are solved, levels are not
  slide_two_harnesses.png     the same model scored two ways -- what a held-out
                              station's neighbours are worth

Reads cached prediction tables only; nothing re-fits.

Run from repo root::  PYTHONPATH=. python handout/plot_slide_figures.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from emt.evaluation import metrics

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "handout" / "figures"
T = "sm_rootzone_pct"
INK, WATER, ALARM, MUTE = "#12222A", "#10707F", "#A6392F", "#8AA0A7"

loso = pd.read_csv(REPO / "data" / "model8_loso_predictions.csv", parse_dates=["time"])
blocked = pd.read_csv(REPO / "data" /
                      "model8_blockcv_capacity_aridity_weighted_predictions.csv",
                      parse_dates=["time"])

# ---------------------------------------------------------------- 1. rhythm
# Y7: high correlation, badly offset -- the failure mode, not a cherry-picked win.
stn = "Y7"
g = blocked[blocked.station == stn].sort_values("time")
m = metrics(g[T], g["pred"])
fig, ax = plt.subplots(figsize=(13, 5.2))
ax.plot(g["time"], g[T], color=INK, lw=1.5, label="observed")
ax.plot(g["time"], g["pred"], color=WATER, lw=1.5, label="predicted, station withheld")
ax.fill_between(g["time"], g[T], g["pred"], color=ALARM, alpha=.13, lw=0)
ax.set_ylabel("root-zone soil moisture (%)", fontsize=12)
ax.set_title(f"Station {stn}, withheld from fitting:  phase and amplitude reproduced, "
             f"level displaced\n"
             f"r = {m['r']:.2f}      NSE = {m['nse']:+.2f}      "
             f"mean bias = {m['bias']:+.1f} percentage points",
             fontsize=13, pad=12, loc="left")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(fontsize=11, loc="upper left", frameon=False)
ax.grid(alpha=.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIGS / "slide_rhythm_vs_level.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------- 2. r against NSE
per = blocked.groupby("station").apply(
    lambda d: pd.Series(metrics(d[T], d["pred"])), include_groups=False)
fig, ax = plt.subplots(figsize=(9.5, 6.4))
ax.axhspan(-1, 0, color=ALARM, alpha=.06, lw=0)
ax.scatter(per["r"], per["nse"].clip(lower=-1), s=64, color=WATER,
           edgecolor="white", linewidth=1.1, zorder=3)
ax.axhline(0, color=INK, lw=1)
ax.set(xlim=[0, 1], ylim=[-1.05, 1],
       xlabel="Pearson correlation with observations",
       ylabel="Nash–Sutcliffe efficiency")
ax.set_title("Correlation against efficiency, all 37 stations, district withheld\n"
             "(efficiency clipped below at −1)", fontsize=13, pad=12, loc="left")
ax.annotate("below zero: no improvement upon\nthe station's own long-run mean",
            (0.03, -0.55), fontsize=10.5, color=ALARM, va="center")
ax.grid(alpha=.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIGS / "slide_r_vs_nse.png", dpi=140)
plt.close(fig)

# --------------------------------------------------- 3. what neighbours buy
a = loso.groupby("station").apply(
    lambda d: metrics(d[T], d["pred"])["nse"], include_groups=False)
b = blocked.groupby("station").apply(
    lambda d: metrics(d[T], d["pred"])["nse"], include_groups=False)
common = [s for s in a.index if s in b.index]
a, b = a[common].clip(lower=-1.5), b[common].clip(lower=-1.5)
order = a.sort_values().index
fig, ax = plt.subplots(figsize=(13, 5.4))
x = np.arange(len(order))
ax.vlines(x, b[order], a[order], color=MUTE, lw=1.4, zorder=1)
ax.scatter(x, a[order], s=42, color=WATER, zorder=3,
           label="single station withheld (neighbours retained)")
ax.scatter(x, b[order], s=42, color=ALARM, zorder=3,
           label="entire district withheld")
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=90, fontsize=8)
ax.set_ylabel("Nash–Sutcliffe efficiency", fontsize=12)
ax.set_title("Per-station efficiency under two cross-validation designs\n"
             "(clipped below at −1.5)", fontsize=13, pad=12, loc="left")
ax.legend(fontsize=11, loc="lower right", frameon=False)
ax.grid(alpha=.25, axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIGS / "slide_two_harnesses.png", dpi=140)
plt.close(fig)

print("wrote slide_rhythm_vs_level.png, slide_r_vs_nse.png, slide_two_harnesses.png")
print(f"  {stn}: r {m['r']:.2f}  NSE {m['nse']:+.2f}  bias {m['bias']:+.1f}")
print(f"  stations with r>0.7: {(per['r']>0.7).sum()}/{len(per)}; "
      f"with NSE>0: {(per['nse']>0).sum()}/{len(per)}")
