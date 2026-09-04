"""Seasonal gallery from the recommended ensemble: coarse SMIPS beside the
30 m field, nine dates through 2008.

The model4 gallery (``plot_downscale_gallery.py``) shows what downscaling
*does*; this shows it at the branch's best skill — the median of the mappable
members of the blocked pick (model6, model8, model9, nn-hybrid).

Cost note: the bucket models are sequential, so ONE spin-up simulation passes
through every date. The forcing grid, terrain and soil are fetched once, each
process model is simulated once, and the nine dates are read out as snapshots
— a ninth of the naive cost.

Run from repo root::  PYTHONPATH=. python handout/plot_downscale_gallery_best.py
"""
from __future__ import annotations
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

from emt.queries import query_for_focus_area
from emt.downscale import downscale
from emt.smips import smips_lookback_series
from emt.slga import soil_covariates, SOIL_VARS
from emt.antecedent import antecedent_grid, antecedent_day_layers
from emt.covariates import terrain_covariates, sample_points
from emt.model6 import model as m6mod
from emt.model6.model import FEATURES as M6_FEATURES
from emt.model8 import predict as m8p
from emt.model8.model import TERRAIN_STATIC_VARS
from emt.nn.spatial import hybrid_map
from emt.persist import fit_cached

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "downscale_gallery_ensemble.png"
AOI = "kyeamba"
DATES = [date(2008, m, d) for m, d in
         [(1, 15), (2, 25), (4, 5), (5, 15), (6, 25), (8, 5), (9, 15), (10, 25), (12, 5)]]
CLIM_START = date(2006, 1, 1)
t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

q = query_for_focus_area(AOI, DATES[0], DATES[-1])
bbox = tuple(q.bbox)

# ---------------------------------------------------------------- model6 (ML)
m6 = fit_cached(m6mod, pd.read_csv(REPO / "data" / "model6_features_2006_2010.csv"),
                "model6_sk19")
lb = smips_lookback_series(q, DATES)
soil = soil_covariates(q)
soil_layers = {v: soil[v] for v in SOIL_VARS}
ant = antecedent_grid(q, CLIM_START, DATES[-1])
stamp("model6 ready; AOI SMIPS lookback series, soil, antecedent grid")

fields, smips_fields = {d: [] for d in DATES}, {}
for d in DATES:
    extra = {**lb[d], **soil_layers, **antecedent_day_layers(ant, d)}
    ds = downscale(m6, query_for_focus_area(AOI, d, d), d, M6_FEATURES, extra_layers=extra)
    fields[d].append(ds["sm_pred"])
    smips_fields[d] = ds["smips_native"]
    if "grid" not in dir():
        grid = ds["sm_pred"]
stamp("model6: 9 dates")

# ------------------------------------------------- process models, ONE run each
terr = terrain_covariates(q)
sim_start = m8p._spinup_start(DATES[0])
rain, pet, aridity, lons, lats = m8p._forcing_grid(bbox, sim_start, DATES[-1],
                                                   m8p.GRID_STEP_DEG, True)
rows = {d: (d - sim_start).days for d in DATES}
stamp(f"forcing grid {rain.shape} ({sim_start} .. {DATES[-1]})")

src = {**{v: (soil[v], Resampling.nearest) for v in SOIL_VARS},
       **{v: (terr[v], Resampling.nearest) for v in TERRAIN_STATIC_VARS},
       "aridity": (xr.DataArray(aridity.reshape(len(lats), len(lons)),
                                coords={"y": lats, "x": lons}, dims=("y", "x"))
                   .rio.write_crs(4326), Resampling.bilinear)}

def on_grid(a, resamp=Resampling.nearest):
    if a.rio.crs is None:
        a = a.rio.write_crs(4326)
    return a.rio.reproject_match(grid, resampling=resamp)

for name in ("model8", "model9"):
    model = m8p._load(None, name)
    awc = np.array([float(sample_points(soil["soil_awc"], float(lo), float(la)).values)
                    for la in lats for lo in lons])
    storage = m8p._simulate(rain, pet, model, cap_ratio=m8p._cap_ratio(model, awc))
    S = np.column_stack([on_grid(*src[v]).values.ravel() for v in model._static_vars])
    lim = m8p._pedo_limits(model, on_grid(soil["soil_clay"]).values.ravel(),
                           on_grid(soil["soil_sand"]).values.ravel())
    ok = np.isfinite(S).all(axis=1)
    for d in DATES:
        coarse = xr.DataArray(storage[rows[d]].reshape(len(lats), len(lons)),
                              coords={"y": lats, "x": lons}, dims=("y", "x")).rio.write_crs(4326)
        stor = on_grid(coarse, Resampling.bilinear).values.ravel()
        pred = np.full(stor.shape, np.nan, dtype="float32")
        v = ok & np.isfinite(stor)
        lim_v = None if lim is None else (lim[0][v], lim[1][v])
        pred[v] = model.readout(stor[v], S[v], limits=lim_v).astype("float32")
        fields[d].append(xr.DataArray(pred.reshape(grid.shape), coords=grid.coords,
                                      dims=grid.dims))
    stamp(f"{name}: 9 dates from one simulation")

# ------------------------------------------------------- nn-hybrid (snapshots)
dsh = hybrid_map(bbox, DATES[-1], snapshots=DATES)
for d in DATES:
    key = "sm_pred" if d == DATES[-1] else f"sm_pred_{d}"
    fields[d].append(on_grid(dsh[key]))
stamp("nn-hybrid: 9 dates from one simulation")

# ------------------------------------------------------------------- ensemble
ens = {d: xr.DataArray(np.nanmedian(np.stack([f.values for f in fields[d]]), axis=0),
                       coords=grid.coords, dims=grid.dims) for d in DATES}
stamp("ensemble median (model6, model8, model9, nn-hybrid)")

ext = [float(grid.x.min()), float(grid.x.max()), float(grid.y.min()), float(grid.y.max())]
def rng(fs):
    v = np.concatenate([f.values[np.isfinite(f.values)] for f in fs])
    return np.percentile(v, [2, 98])
svmin, svmax = rng([smips_fields[d] for d in DATES])
pvmin, pvmax = rng([ens[d] for d in DATES])

n = len(DATES)
fig, axes = plt.subplots(n, 2, figsize=(6.4, n * 2.35))
ims = [None, None]
for i, d in enumerate(DATES):
    ims[0] = axes[i, 0].imshow(smips_fields[d].values, extent=ext, origin="upper",
                               cmap="YlGnBu", vmin=svmin, vmax=svmax)
    ims[1] = axes[i, 1].imshow(ens[d].values, extent=ext, origin="upper",
                               cmap="YlGnBu", vmin=pvmin, vmax=pvmax)
    axes[i, 0].set_ylabel(d.strftime("%d %b %Y"), fontsize=10)
    for j in (0, 1):
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
axes[0, 0].set_title("Coarse SMIPS input\n(~1 km, mm)", fontsize=11)
axes[0, 1].set_title("Generated 30 m\n(ensemble median, %)", fontsize=11)
fig.subplots_adjust(left=0.11, right=0.86, top=0.95, bottom=0.03, wspace=0.05, hspace=0.06)
fig.colorbar(ims[0], cax=fig.add_axes([0.885, 0.53, 0.02, 0.38]), label="SMIPS (mm)")
fig.colorbar(ims[1], cax=fig.add_axes([0.885, 0.08, 0.02, 0.38]), label="root-zone (%)")
fig.savefig(FIG, dpi=125, bbox_inches="tight")
plt.close(fig)
stamp(f"wrote {FIG.relative_to(REPO)}")
