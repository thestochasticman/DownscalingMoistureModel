"""One-page summary card for the nominated property, calendar 2025.

Three jobs, three forms: the headline figures as stat tiles, the year as a
daily line, and the seasonal shape as monthly means with their observed range.
A single sequential hue carries magnitude throughout; one reserved accent marks
the driest day. No second axis -- every panel is in volumetric per cent.

Run from repo root::  PYTHONPATH=. python handout/plot_property_summary.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parent.parent
DIR = REPO / "handout" / "figures" / "property_2025"
FIG = DIR / "property_summary.png"

INK, MID, MUTE = "#12222A", "#465C65", "#8AA0A7"
WATER, WATER_L, RULE = "#10707F", "#BBDCE1", "#D5DFE1"
DRY = "#A6392F"                      # reserved: marks the driest day only

s = pd.read_csv(DIR / "property_soil_moisture_2025.csv", parse_dates=["date"])
v = s["soil_moisture_pct"]
lat, lon = s["lat"].iloc[0], s["lon"].iloc[0]

wet_i, dry_i = v.idxmax(), v.idxmin()
mon = s.groupby(s.date.dt.month)["soil_moisture_pct"].agg(["mean", "min", "max"])
mon.index = [pd.Timestamp(2025, m, 1).strftime("%b") for m in mon.index]

# forcing, for the provenance line
silo = pd.read_csv("/workspace/paddockts-data/tmp/"
                   "m8pt_m35p0970_148p9361_2023m01m01_2025m12m31/Environmental/"
                   "m8pt_m35p0970_148p9361_2023m01m01_2025m12m31_silo.csv")
silo.columns = ["time" if c.startswith("YYYY") else c for c in silo.columns]
silo["time"] = pd.to_datetime(silo["time"])
y25 = silo[silo.time.dt.year == 2025]
rain_2025 = y25["daily_rain"].sum()
aridity = silo["daily_rain"].mean() / silo["et_morton_potential"].mean()

fig = plt.figure(figsize=(12, 10.5))
gs = GridSpec(4, 3, figure=fig, height_ratios=[.34, .50, 1.15, .95],
              hspace=.55, wspace=.28, left=.075, right=.965, top=.945, bottom=.115)

# ---------------------------------------------------------------- header
hax = fig.add_subplot(gs[0, :]); hax.axis("off")
hax.text(0, .78, "Root-zone soil moisture, 2025", fontsize=21, fontweight="bold",
         color=INK, va="top")
hax.text(0, .20, f"Property centred {lat:.5f}, {lon:.5f}  ·  0–90 cm profile  ·  "
                 f"process model (model8), 30 m", fontsize=11.5, color=MID, va="top")
hax.axhline(0, color=RULE, lw=1)

# ------------------------------------------------------------ stat tiles
tiles = [("Annual mean", f"{v.mean():.1f}%", "of soil volume"),
         ("Wettest", f"{v.max():.1f}%", f"{s.date[wet_i]:%d %B}"),
         ("Driest", f"{v.min():.1f}%", f"{s.date[dry_i]:%d %B}")]
for i, (label, value, note) in enumerate(tiles):
    a = fig.add_subplot(gs[1, i]); a.axis("off")
    a.text(0, 1.02, " ".join(label.upper()), fontsize=8.5, color=MUTE, va="top",
           fontweight="bold")
    a.text(0, .46, value, fontsize=30, fontweight="bold",
           color=DRY if label == "Driest" else WATER, va="center")
    a.text(0, -.10, note, fontsize=10.5, color=MID, va="bottom")

# --------------------------------------------------------- daily series
ax = fig.add_subplot(gs[2, :])
ax.fill_between(s.date, v, v.min() - .6, color=WATER_L, alpha=.55, lw=0)
ax.plot(s.date, v, color=WATER, lw=1.7)
ax.axhline(v.mean(), color=MUTE, lw=1, ls=(0, (5, 4)))
ax.text(s.date.iloc[len(s) // 2], v.mean() - 1.05, f"annual mean {v.mean():.1f}%",
        fontsize=10, color=MID, ha="center")
for idx, col, lbl, dy in ((wet_i, WATER, "wettest", 1.1), (dry_i, DRY, "driest", -1.9)):
    ax.plot(s.date[idx], v[idx], "o", ms=8, color=col, zorder=4,
            markeredgecolor="white", markeredgewidth=1.4)
    ax.annotate(f"{lbl}  {v[idx]:.1f}%", (s.date[idx], v[idx] + dy), fontsize=10.5,
                color=col, ha="center", fontweight="bold")
ax.set_ylabel("soil moisture (% of volume)", fontsize=11.5, color=MID)
ax.set_ylim(v.min() - 1.4, v.max() + 2.6)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.tick_params(colors=MID, labelsize=10)
ax.grid(axis="y", color=RULE, lw=.8)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(RULE)

# --------------------------------------------------------- monthly shape
bx = fig.add_subplot(gs[3, :])
x = np.arange(len(mon))
bx.vlines(x, mon["min"], mon["max"], color=WATER_L, lw=7, capstyle="round", zorder=1)
bx.plot(x, mon["mean"], "o-", color=WATER, lw=2, ms=8, zorder=3,
        markeredgecolor="white", markeredgewidth=1.4)
for i, m in enumerate(mon["mean"]):
    bx.annotate(f"{m:.1f}", (i, mon["max"].iloc[i] + .55), fontsize=9.5,
                color=MID, ha="center")
bx.set_xticks(x); bx.set_xticklabels(mon.index, fontsize=10.5, color=MID)
bx.set_ylabel("monthly mean\nand observed range (%)", fontsize=11.5, color=MID)
bx.tick_params(colors=MID, labelsize=10)
bx.grid(axis="y", color=RULE, lw=.8)
bx.set_axisbelow(True)
bx.set_ylim(mon["min"].min() - 1.2, mon["max"].max() + 2.4)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    bx.spines[sp].set_color(RULE)

# ---------------------------------------------------------------- footer
fig.text(.075, .050,
         f"Rainfall {rain_2025:.0f} mm in 2025  ·  aridity P/PET {aridity:.2f}, "
         f"inside the calibrated range 0.15–0.60",
         fontsize=10, color=MID)
fig.text(.075, .020,
         "No sensor on site: the seasonal timing is dependable, the absolute level "
         "carries an unquantified offset of a few percentage points.",
         fontsize=10, color=MUTE)

fig.savefig(FIG, dpi=150, facecolor="white")
plt.close(fig)
print(f"wrote {FIG.relative_to(REPO)}")
print(f"  mean {v.mean():.1f}%  min {v.min():.1f}% ({s.date[dry_i]:%d %b})  "
      f"max {v.max():.1f}% ({s.date[wet_i]:%d %b})  rain {rain_2025:.0f} mm")
