"""One date, two panels: the coarse product beside the 30 m field it becomes.

The nine-row seasonal gallery (``plot_downscale_gallery.py``) is the complete
record; this is the single-date version of the same comparison, sized for a
slide. Left: SMIPS as delivered, ~1 km, blocky. Right: the 30 m field the
downscaling model generates over the same ground on the same day.

Run from repo root::  PYTHONPATH=. python handout/plot_downscale_pair_single.py
"""
from __future__ import annotations
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

from emt.queries import query_for_focus_area
from emt.downscale import downscale
from emt.smips import smips_lookback_day, smips_day
from emt.slga import soil_covariates, SOIL_VARS
from emt.model4.model import build_estimator, ensure_features, FEATURES, TARGET

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "downscale_pair_single.png"
DAY = date(2008, 8, 5)          # deep in the austral winter wet-up
AREA = "kyeamba"

table = ensure_features(pd.read_csv(REPO / "data" / "train_catchment_plus_m_2006_2010.csv"))
sub = table.dropna(subset=FEATURES + [TARGET])
est = build_estimator().fit(sub[FEATURES], sub[TARGET])
print(f"model4 fitted on {len(sub)} rows", flush=True)

q = query_for_focus_area(AREA, DAY, DAY)
lb = smips_lookback_day(q, DAY, workers=32)
soil = soil_covariates(q)
extra = {**lb, **{v: soil[v] for v in SOIL_VARS}}
ds = downscale(est, q, DAY, FEATURES, extra_layers=extra)
print("30 m field:", dict(ds.sizes), flush=True)

coarse = smips_day(DAY, tuple(q.bbox)).rio.write_crs(4326)
grid = ds["sm_pred"]
coarse_on = coarse.rio.reproject_match(grid, resampling=Resampling.nearest)

fig, ax = plt.subplots(1, 2, figsize=(13, 6.6))
c0 = ax[0].imshow(coarse_on.values, cmap="YlGnBu", origin="upper")
ax[0].set_title("As delivered — SMIPS, ≈1 km", fontsize=13, pad=10)
fig.colorbar(c0, ax=ax[0], fraction=.046, pad=.03, label="soil water (mm)")

v = grid.values
c1 = ax[1].imshow(v, cmap="YlGnBu", origin="upper",
                  vmin=np.nanpercentile(v, 2), vmax=np.nanpercentile(v, 98))
ax[1].set_title("Generated — 30 m", fontsize=13, pad=10)
fig.colorbar(c1, ax=ax[1], fraction=.046, pad=.03, label="root-zone moisture (%)")

for a in ax:
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_edgecolor("#8AA0A7")

fig.suptitle(f"Kyeamba, {DAY:%d %B %Y}", fontsize=15, y=0.97)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG, dpi=140)
plt.close(fig)
print(f"wrote {FIG.relative_to(REPO)}")
