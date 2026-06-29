"""Illustrate the SMIPS WCS grid-alignment issue with real data at station K6.

Top row: three request windows without native-grid alignment. The server refits
an integer pixel grid to each requested box, so the cell boundaries differ
between windows and the station (star) falls in different ~1 km cells, giving
three different sampled values for the same location and date.

Bottom row: the same three windows with snap_bbox. Each request is aligned to the
native grid, the returned cells coincide, and the station yields one value.

Run from repo root::  PYTHONPATH=. python handout/plot_grid_alignment.py
"""
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import emt.smips as smips
from emt.smips import smips_day
from emt.queries import query_for_station

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "grid_alignment.png"
DAY = date(2020, 6, 1)
LON, LAT = 147.45720, -35.38978          # station K6
HALF = 0.028                              # zoom half-width (deg), common to all panels

q_stn = query_for_station("K6", LAT, LON, DAY, DAY, buffer_km=1.5)
WINDOWS = {
    "Wide request":   [147.435, -35.437, 147.626, -35.372],
    "Medium request": list(q_stn.bbox),
    "Narrow request": [LON - 0.015, LAT - 0.015, LON + 0.015, LAT + 0.015],
}


def fetch(bbox):
    return smips_day(DAY, tuple(bbox)).sortby("x").sortby("y")


def cell_edges(centres):
    c = np.asarray(centres, float)
    d = np.diff(c).mean()
    return np.concatenate([[c[0] - d / 2], c + d / 2])


def value_at(da):
    return float(da.sel(x=LON, y=LAT, method="nearest").values)


orig_snap = smips.snap_bbox
fields = {}
for name, bbox in WINDOWS.items():
    smips.snap_bbox = lambda b, pad=1: list(b)       # alignment off (raw bbox)
    off = fetch(bbox)
    smips.snap_bbox = orig_snap                       # alignment on
    on = fetch(bbox)
    fields[name] = (off, on)

crop = (fields["Wide request"][0]
        .sel(x=slice(LON - HALF, LON + HALF), y=slice(LAT - HALF, LAT + HALF)).values)
vmin, vmax = np.nanpercentile(crop, [5, 95])

fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
for col, (name, (off, on)) in enumerate(fields.items()):
    for row, da in [(0, off), (1, on)]:
        ax = axes[row, col]
        ax.pcolormesh(cell_edges(da.x.values), cell_edges(da.y.values), da.values,
                      cmap="YlGnBu", vmin=vmin, vmax=vmax,
                      edgecolors="0.5", linewidth=0.4)
        v = value_at(da)
        ax.plot(LON, LAT, marker="*", ms=22, mfc="red", mec="k", mew=1.2, zorder=5)
        ax.set_xlim(LON - HALF, LON + HALF); ax.set_ylim(LAT - HALF, LAT + HALF)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title((f"{name}\n" if row == 0 else "") +
                     ("aligned (snap_bbox): " if row == 1 else "raw request: ") +
                     f"{v:.1f} mm", fontsize=11)
        ax.text(LON, LAT - 0.006, f"{v:.1f} mm", ha="center", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", alpha=.85, ec="none"))

axes[0, 0].set_ylabel("Without grid alignment\n(server refits pixels to the box)", fontsize=12)
axes[1, 0].set_ylabel("With grid alignment\n(request snapped to native grid)", fontsize=12)
fig.suptitle("SMIPS WCS grid alignment at station K6 (2020-06-01): same location, "
             "three request windows", fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIG, dpi=130)
print("wrote", FIG.relative_to(REPO))
for name, (off, on) in fields.items():
    print(f"{name:16} raw={value_at(off):6.2f}  aligned={value_at(on):6.2f}")
