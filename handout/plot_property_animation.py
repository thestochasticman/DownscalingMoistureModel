"""Weekly 30 m soil-moisture frames for the nominated property, 2025.

The water balance is integrated ONCE over the whole period on the SILO forcing
grid; each frame then applies that day's storage through the fitted readout on
the 30 m grid. This avoids re-fetching the forcing per frame, which calling
predict_map repeatedly would do (each date is a different Query stub).

Outputs an animated GIF plus the individual PNG frames.
"""
from __future__ import annotations
import os
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from rasterio.enums import Resampling
from PaddockTS.query import Query

from emt.covariates import terrain_covariates
from emt.model8.model import TERRAIN_STATIC_VARS
from emt.model8.predict import (_forcing_grid, _simulate, _cap_ratio, _pedo_limits,
                                _spinup_start, _tag)
from emt.persist import load_model
from emt.slga import soil_covariates, SOIL_VARS

SCR = Path("/tmp/claude-0/-workspace-DownscalingMoistureModel/"
           "84ec1765-7145-45bf-a9b7-1821ed763972/scratchpad")
FRAMES = SCR / "property_frames"
FRAMES.mkdir(exist_ok=True)
GIF = SCR / "property_soil_moisture_2025.gif"

BBOX = (148.91869, -35.10803, 148.95345, -35.08605)
START, END = date(2025, 1, 1), date(2025, 12, 31)
STEP_DAYS = 7

model = load_model("model8")
q = Query(bbox=list(BBOX), start=END, end=END,
          stub=_tag("propmap", *[f"{v:.4f}" for v in BBOX], END))

print("terrain (30 m) ...", flush=True)
terr = terrain_covariates(q)
grid = terr["elevation"]
print("  grid", dict(grid.sizes), flush=True)
print("SLGA soil ...", flush=True)
soil = soil_covariates(q)

sim_start = _spinup_start(START)
print(f"SILO forcing {sim_start} -> {END} ...", flush=True)
rain, pet, aridity, lons, lats = _forcing_grid(BBOX, sim_start, END, 0.1, True)
times = pd.date_range(sim_start, END, freq="D")[:rain.shape[0]]

# capacity per forcing cell, then one integration for the whole period
awc_cells = np.array([float(soil["soil_awc"].sel(x=lo, y=la, method="nearest").values)
                      for la in lats for lo in lons])
storage_all = _simulate(rain, pet, model, cap_ratio=_cap_ratio(model, awc_cells))
print("storage integrated:", storage_all.shape, flush=True)

# static layers on the 30 m grid, in the fitted model's variable order
src = {**{v: (soil[v], Resampling.nearest) for v in SOIL_VARS},
       **{v: (terr[v], Resampling.nearest) for v in TERRAIN_STATIC_VARS},
       "aridity": (xr.DataArray(aridity.reshape(len(lats), len(lons)),
                                coords={"y": lats, "x": lons}, dims=("y", "x")
                                ).rio.write_crs(4326), Resampling.bilinear)}
cols = []
for v in model._static_vars:
    a, rs = src[v]
    if a.rio.crs is None:
        a = a.rio.write_crs(4326)
    cols.append(a.rio.reproject_match(grid, resampling=rs).values.ravel())
S = np.column_stack(cols)


def _on_grid(name):
    a = soil[name]
    if a.rio.crs is None:
        a = a.rio.write_crs(4326)
    return a.rio.reproject_match(grid, resampling=Resampling.nearest).values.ravel()


lim = _pedo_limits(model, _on_grid("soil_clay"), _on_grid("soil_sand"))

days = []
d = START
while d <= END:
    days.append(d)
    d += timedelta(days=STEP_DAYS)

# first pass: compute every frame so the colour scale is common to all
fields = []
for d in days:
    i = int((pd.Timestamp(d) - times[0]).days)
    coarse = (xr.DataArray(storage_all[i].reshape(len(lats), len(lons)),
                           coords={"y": lats, "x": lons}, dims=("y", "x"))
              .rio.write_crs(4326))
    stor = coarse.rio.reproject_match(grid, resampling=Resampling.bilinear).values.ravel()
    ok = np.isfinite(S).all(axis=1) & np.isfinite(stor)
    pred = np.full(stor.shape, np.nan, dtype="float32")
    if ok.any():
        lv = None if lim is None else (lim[0][ok], lim[1][ok])
        pred[ok] = model.readout(stor[ok], S[ok], limits=lv).astype("float32")
    fields.append(pred.reshape(grid.shape))

stack = np.stack(fields)
vmin, vmax = np.nanpercentile(stack, 1), np.nanpercentile(stack, 99)
print(f"{len(days)} frames, colour scale {vmin:.1f}-{vmax:.1f}%", flush=True)

paths = []
for d, f in zip(days, fields):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(f, cmap="YlGnBu", vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{d:%d %B %Y}", fontsize=15, pad=10, family="monospace")
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    cb.set_label("root-zone soil moisture (%)", fontsize=10)
    ax.text(0.5, -0.06, f"mean {np.nanmean(f):.1f}%", transform=ax.transAxes,
            ha="center", fontsize=11, color="#465C65")
    fig.tight_layout()
    p = FRAMES / f"frame_{d:%Y%m%d}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

imageio.mimsave(GIF, [imageio.imread(p) for p in paths], duration=0.28, loop=0)
print(f"wrote {GIF} ({GIF.stat().st_size/1e6:.1f} MB, {len(paths)} frames)")
print(f"frames in {FRAMES}")
