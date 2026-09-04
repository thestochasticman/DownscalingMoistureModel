"""30 m maps from the hybrid: per-pixel bucket parameters, any date.

model8's map runs ONE water balance per ~5 km forcing cell and adds 30 m
structure only through the readout offsets. The hybrid's parameters vary with
the statics, so here every 30 m pixel gets its own bucket -- capacity, ET
stress and recession from its own soil/terrain -- forced by its nearest SILO
cell, exactly mirroring how the model was trained (station statics + station
forcing). The heavy lifting (SILO forcing grid, terrain/SLGA rasters, spin-up
convention, output writing) is reused from :mod:`emt.model8.predict`.

    PYTHONPATH=. python -m emt.nn.spatial --bbox 147.30 -35.52 147.62 -35.10 --date 2008-08-05
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from rasterio.enums import Resampling

from emt.covariates import terrain_covariates
from emt.slga import soil_covariates, SOIL_VARS
from emt.model8 import predict as m8p
from emt.nn.hybrid import HybridModel

MODEL_PATH = Path("data/models/nn_hybrid_q.pt")


def hybrid_map(bbox, day, model_path: str | Path = MODEL_PATH,
               step_deg: float = m8p.GRID_STEP_DEG, chunk: int = 500_000,
               verbose: bool = True, snapshots=None) -> xr.Dataset:
    """A 30 m root-zone soil-moisture field for any ``day`` (no SMIPS).

    ``snapshots``: extra dates to read out from the SAME simulation. The bucket
    is sequential, so one spin-up run passes through every earlier date -- a
    nine-date gallery costs one simulation, not nine. Returns those fields as
    additional data variables named ``sm_pred_<YYYY-MM-DD>``.
    """
    day = m8p._as_date(day)
    snaps = sorted({m8p._as_date(d) for d in (snapshots or [])} | {day})
    model = HybridModel.load(model_path)
    cfg = model.data
    sim_start = m8p._spinup_start(day)
    q = m8p.Query(bbox=list(bbox), start=day, end=day,
                  stub=m8p._tag("nnhybmap", *[f"{v:.3f}" for v in bbox], day))

    if verbose:
        print("  terrain (30 m) ...", flush=True)
    terr = terrain_covariates(q)
    grid = terr["elevation"]
    if verbose:
        print("  SLGA soil ...", flush=True)
    soil = soil_covariates(q)
    rain, pet, aridity, lons, lats = m8p._forcing_grid(bbox, sim_start, day,
                                                       step_deg, verbose)

    # statics per pixel, in the model's fitted order (cfg.statics)
    src = {**{v: (soil[v], Resampling.nearest) for v in SOIL_VARS},
           **{v: (terr[v], Resampling.nearest) for v in terr.data_vars},
           "aridity": (xr.DataArray(aridity.reshape(len(lats), len(lons)),
                                    coords={"y": lats, "x": lons},
                                    dims=("y", "x")).rio.write_crs(4326),
                       Resampling.bilinear)}
    cols = []
    for v in cfg.statics:
        a, resamp = src[v]
        if a.rio.crs is None:
            a = a.rio.write_crs(4326)
        cols.append(a.rio.reproject_match(grid, resampling=resamp).values.ravel())
    S = np.column_stack(cols).astype(np.float32)

    # nearest forcing cell per pixel
    gx, gy = np.meshgrid(grid.x.values, grid.y.values)
    ix = np.abs(gx.ravel()[:, None] - lons[None, :]).argmin(1)
    iy = np.abs(gy.ravel()[:, None] - lats[None, :]).argmin(1)
    cell = (iy * len(lons) + ix).astype(np.int64)

    # row index of each requested date in the forcing series (daily from sim_start)
    n_days = rain.shape[0]
    rows = {d: (d - sim_start).days for d in snaps}
    if any(r < 0 or r >= n_days for r in rows.values()):
        raise ValueError(f"snapshot dates outside the forcing window "
                         f"({sim_start} .. {day}, {n_days} days)")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rain_t = torch.as_tensor(rain, dtype=torch.float32, device=dev)   # (T, n_cells)
    pet_t = torch.as_tensor(pet, dtype=torch.float32, device=dev)
    valid = np.isfinite(S).all(1)
    Sz = torch.as_tensor(model.scaler.x(np.where(valid[:, None], S, 0.0)),
                         dtype=torch.float32, device=dev)
    cell_t = torch.as_tensor(cell, device=dev)

    preds = {d: np.full(len(S), np.nan, dtype="float32") for d in snaps}
    idx_all = np.flatnonzero(valid)
    if verbose:
        print(f"  simulating {len(idx_all):,} pixels x {n_days} days x "
              f"{len(model.nets)} members on {dev} "
              f"({len(snaps)} snapshot date(s)) ...", flush=True)
    want = {r: d for d, r in rows.items()}
    with torch.no_grad():
        for start in range(0, len(idx_all), chunk):
            sl = slice(start, start + chunk)
            idx = torch.as_tensor(idx_all[sl], device=dev)
            sz, cl = Sz[idx], cell_t[idx]
            acc = {d: torch.zeros(len(idx), device=dev) for d in snaps}
            for net in model.nets:
                bucket = net.net.to(dev)
                smax, alpha, k, theta_r, dtheta = bucket.params(sz).unbind(1)
                off = (bucket.offset(sz).squeeze(-1) if bucket.offset is not None
                       else torch.zeros(len(idx), device=dev))
                s = 0.5 * smax
                denom = alpha * smax
                for t in range(n_days):
                    s = s + rain_t[t, cl]
                    aet = pet_t[t, cl] * torch.clamp(s / denom, max=1.0)
                    s = torch.clamp(s - aet - k * s, min=0.0)
                    s = torch.minimum(s, smax)
                    if t in want:
                        acc[want[t]] += theta_r + dtheta * s / smax + off
            for d in snaps:
                preds[d][idx_all[sl]] = (acc[d] / len(model.nets)).cpu().numpy()
    if verbose:
        for d in snaps:
            print(f"  {d}: mean {np.nanmean(preds[d]):.1f}%", flush=True)
    out = {"sm_pred": (grid.dims, preds[day].reshape(grid.shape))}
    for d in snaps:
        if d != day:
            out[f"sm_pred_{d}"] = (grid.dims, preds[d].reshape(grid.shape))
    return xr.Dataset(out, coords=grid.coords).rio.write_crs(grid.rio.crs)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    ap.add_argument("--date", required=True)
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    ds = hybrid_map(tuple(a.bbox), a.date, a.model)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        ds["sm_pred"].rio.to_raster(a.out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
