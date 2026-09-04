"""One date, two panels: the coarse product beside the 30 m field — from the
BEST mappable configuration, not a single early model.

Left: SMIPS as delivered, ~1 km. Right: the median of the four mappable
models' 30 m fields — nn-hybrid (per-pixel bucket), model8, model9, model6 —
the map analogue of the recommended blocked ensemble (minus the SMIPS-anchor
hybrid variant, whose training-era climatological static has no per-pixel
equivalent). Every ingredient is the same public national data.

Run from repo root::  PYTHONPATH=. python handout/plot_downscale_pair_best.py
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
from rasterio.enums import Resampling

from emt.queries import query_for_focus_area
from emt.downscale import downscale
from emt.smips import smips_lookback_day, smips_day
from emt.slga import soil_covariates, SOIL_VARS
from emt.antecedent import antecedent_grid, antecedent_day_layers
from emt.model6 import model as m6mod
from emt.model6.model import FEATURES as M6_FEATURES
from emt.persist import fit_cached
from emt.model8.predict import predict_map
from emt.nn.spatial import hybrid_map

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "downscale_pair_single.png"
DAY = date(2008, 8, 5)          # deep in the austral winter wet-up
AREA = "kyeamba"
CLIM_PERIOD = (date(2006, 1, 1), date(2010, 12, 31))
t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

q = query_for_focus_area(AREA, DAY, DAY)
bbox = tuple(q.bbox)

# --- model6 (boosting, needs SMIPS + antecedent): fit-once via persist.
# The shipped model6.joblib predates this env's scikit-learn and cannot be
# unpickled, so the fit is cached under an env-tagged name; the CACHED FEATURE
# TABLE supplies every column, so nothing is fetched to (re)fit ---
m6 = fit_cached(m6mod, pd.read_csv(REPO / "data" / "model6_features_2006_2010.csv"), "model6_sk19")
stamp("model6 ready (fit-once cache: model6_sk19)")
lb = smips_lookback_day(q, DAY, workers=32)
soil = soil_covariates(q)
ant_cube = antecedent_grid(q, CLIM_PERIOD[0], DAY)
extra = {**lb, **{v: soil[v] for v in SOIL_VARS}, **antecedent_day_layers(ant_cube, DAY)}
ds6 = downscale(m6, q, DAY, M6_FEATURES, extra_layers=extra)
grid = ds6["sm_pred"]
stamp("model6 30 m field")

# --- process track: model8 and model9 (bucket + offsets / pedotransfer) ---
maps = {"model6": grid}
for name in ("model8", "model9"):
    ds = predict_map(bbox, DAY, model_name=name, save=False, plot=False)
    maps[name] = ds["sm_pred"].rio.reproject_match(grid, resampling=Resampling.nearest)
    stamp(f"{name} 30 m field")

# --- nn-hybrid: per-pixel bucket parameters ---
dsh = hybrid_map(bbox, DAY)
maps["nn-hybrid"] = dsh["sm_pred"].rio.reproject_match(grid, resampling=Resampling.nearest)
stamp("nn-hybrid 30 m field")

stack = np.stack([m.values for m in maps.values()])
best = np.nanmedian(stack, axis=0)
stamp(f"ensemble median of {list(maps)}")

coarse = smips_day(DAY, bbox).rio.write_crs(4326)
coarse_on = coarse.rio.reproject_match(grid, resampling=Resampling.nearest)

fig, ax = plt.subplots(1, 2, figsize=(13, 6.6))
c0 = ax[0].imshow(coarse_on.values, cmap="YlGnBu", origin="upper")
ax[0].set_title("As delivered — SMIPS, ≈1 km", fontsize=13, pad=10)
fig.colorbar(c0, ax=ax[0], fraction=.046, pad=.03, label="soil water (mm)")
c1 = ax[1].imshow(best, cmap="YlGnBu", origin="upper",
                  vmin=np.nanpercentile(best, 2), vmax=np.nanpercentile(best, 98))
ax[1].set_title("Generated — 30 m (4-model ensemble median)", fontsize=13, pad=10)
fig.colorbar(c1, ax=ax[1], fraction=.046, pad=.03, label="root-zone moisture (%)")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"Kyeamba Creek, {DAY} — the coarse input and the field the ensemble generates",
             fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG, dpi=110)
print(f"wrote {FIG.relative_to(REPO)}")
