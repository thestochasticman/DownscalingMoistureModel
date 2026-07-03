"""Gallery of generated 30 m soil moisture from model6 (the recommended model).

model6 adds antecedent-meteorology features. For a downscaled map these are
needed per pixel, but SILO is a ~5 km grid and the focus AOI is ~35 km across,
so the trailing-window features are taken at the AOI centre and applied
uniformly over the field (a small-AOI approximation; a national product would
use gridded SILO). The spatial structure still comes from terrain, soil and
SMIPS climatology; the antecedent features set the per-date level.

Run from repo root::  PYTHONPATH=. python handout/plot_downscale_gallery_model6.py
"""
from __future__ import annotations
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emt.queries import query_for_focus_area, query_for_station
from emt.downscale import downscale
from emt.smips import smips_climatology
from emt.slga import soil_covariates, SOIL_VARS
from emt.antecedent import _trailing, _station_silo, ANTECEDENT_VARS
from emt.model6.model import build_estimator, ensure_features, FEATURES, TARGET

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "downscale_gallery_model6.png"
TABLE = REPO / "data" / "train_catchment_plus_m_2006_2010.csv"
AOI = "kyeamba"
CLIM_PERIOD = (date(2006, 1, 1), date(2010, 12, 31))
DATES = [date(2008, m, d) for m, d in
         [(1, 15), (2, 25), (4, 5), (5, 15), (6, 25), (8, 5), (9, 15), (10, 25), (12, 5)]]
t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# --- production model: model6 on all 36 stations ---
tab = ensure_features(pd.read_csv(TABLE)).dropna(subset=FEATURES + [TARGET])
model = build_estimator().fit(tab[FEATURES], tab[TARGET])
stamp(f"trained model6 on {tab.station.nunique()} stations, {len(tab):,} rows")

# --- static AOI rasters (climatology + soil) ---
q_clim = query_for_focus_area(AOI, *CLIM_PERIOD)
clim = smips_climatology(q_clim, step_days=5)
soil = soil_covariates(q_clim)
static = {"smips_mean_px": clim["smips_mean_px"], "smips_std_px": clim["smips_std_px"],
          **{v: soil[v] for v in SOIL_VARS}}
stamp("AOI climatology + soil")

# --- antecedent meteorology at the AOI centre (trailing windows, per date) ---
minx, miny, maxx, maxy = q_clim.bbox
clat, clon = (miny + maxy) / 2, (minx + maxx) / 2
silo = _station_silo(f"{AOI}_ctr", clat, clon, date(2006, 1, 1), date(2010, 12, 31))
ante = _trailing(silo).set_index("time")
stamp(f"antecedent series at AOI centre ({clat:.2f}, {clon:.2f})")

# --- downscale each date ---
fields = []
for d in DATES:
    ante_d = {v: float(ante.loc[pd.Timestamp(d), v]) for v in ANTECEDENT_VARS}
    ds = downscale(model, query_for_focus_area(AOI, d, d), d, FEATURES,
                   extra_layers={**static, **ante_d})
    fields.append((d, ds["sm_pred"]))
    stamp(f"downscaled {d}  (mean {float(ds['sm_pred'].mean()):.1f}%)")

allvals = np.concatenate([f.values[np.isfinite(f.values)] for _, f in fields])
vmin, vmax = np.percentile(allvals, [2, 98])
g0 = fields[0][1]
ext = [float(g0.x.min()), float(g0.x.max()), float(g0.y.min()), float(g0.y.max())]

ncol = 3
nrow = int(np.ceil(len(fields) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.3, nrow * 4.0))
axes = axes.ravel()
im = None
for ax, (d, f) in zip(axes, fields):
    im = ax.imshow(f.values, extent=ext, origin="upper", cmap="YlGnBu",
                   vmin=vmin, vmax=vmax)
    ax.set_title(d.strftime("%d %b %Y"), fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes[len(fields):]:
    ax.set_visible(False)
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cax, label="root-zone soil moisture (%)")
fig.suptitle(f"Generated 30 m soil moisture over {AOI.title()}, 2008 "
             f"(model6; antecedent meteorology at AOI centre)", fontsize=14, y=0.98)
fig.savefig(FIG, dpi=125, bbox_inches="tight")
stamp(f"wrote {FIG.relative_to(REPO)}")
