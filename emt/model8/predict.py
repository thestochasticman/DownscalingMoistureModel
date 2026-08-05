"""Run model8 anywhere in Australia, for **any date** -- point series or 30 m map.

    from emt.model8.predict import predict_point, predict_map
    s  = predict_point(lat=-35.05, lon=147.5, start="2024-01-01", end="2024-12-31")
    ds = predict_map(bbox=(147.30, -35.52, 147.62, -35.10), day="2024-09-15")

or from the command line::

    python -m emt.model8.predict --lat -35.05 --lon 147.5 --start 2024-01-01 --end 2024-12-31
    python -m emt.model8.predict --bbox 147.30 -35.52 147.62 -35.10 --date 2024-09-15

**Why a process model can answer for any date.** model8 carries no lookback
*features*; it carries a *state*. To predict day D the bucket is simply run
forward from a spin-up start to D on SILO rain/PET -- so the same fitted model
serves 2006 or last week, with no retraining and no SMIPS.

**Spin-up.** The simulation starts at ``SPINUP_YEARS`` calendar years before the
requested period (see :func:`_spinup_start`) purely to wash out the arbitrary
initial condition ``S = 0.5 * smax``. It is *not* a limit on which dates can be
predicted -- any date SILO covers works; it only sets how much forcing is
fetched ahead of the target. Measured convergence at one Murrumbidgee point
(target 2025-06-01, same forcing, varying spin-up):

    spin-up   0.25 yr   0.5 yr   1 yr     2 yr     4 yr     10 yr
    storage   43.28mm   40.68    40.68    40.68    40.68    40.68
    VWC       17.724%   17.473   17.473   17.473   17.473   17.473

Everything from six months out is identical to a ten-year spin-up; only the
three-month run is off (+0.25 % VWC). (Measured with the pre-full-stack
fit, ``k = 0.0073``; the shipped full-stack fit has ``k = 0.0065/day``, a
154-day e-folding, so the default's margin is ~4x either way.) The cost is fetch time on a first run -- in map
mode every forcing cell pulls that many years -- so lowering it to 1 is safe if
you want faster cold starts.

The 30 m map is a genuine downscaling: the water balance runs on the SILO
forcing grid (~5 km, the scale at which weather actually varies), and the
fine structure comes from the per-pixel SLGA soil + terrain offsets on the
30 m Copernicus-DEM grid.

REQUIRES network access, a SILO email and a TERN API key in
``~/.config/PaddockTS.json``. Like every model here, model8 is calibrated on
the Murrumbidgee: elsewhere it produces a plausible but **unvalidated** field.
"""
from __future__ import annotations

import argparse
import os
from datetime import date as _date
from pathlib import Path

# 30 m terrain COGs live on the public Copernicus-DEM bucket -- read anonymously
# so no AWS setup is needed (see emt/predict.py for the same rationale).
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.enums import Resampling

from PaddockTS.query import Query
from PaddockTS.Environmental.SILO.download_silo import download_silo

from emt.covariates import terrain_covariates, sample_points
from emt.model7.model import _step_loop
from emt.model8.model import TERRAIN_STATIC_VARS
from emt.persist import load_model
from emt.slga import soil_covariates, SOIL_VARS

WARNING = ("NOTE: model8 is calibrated on the Murrumbidgee catchment; predictions "
           "elsewhere are plausible but UNVALIDATED. Treat as indicative.")

SPINUP_YEARS = 2          # ~4x the 6-month convergence point (see the docstring)
GRID_STEP_DEG = 0.1       # forcing-grid spacing for maps (SILO's native is 0.05)

_RAIN, _PET = "daily_rain", "et_morton_potential"


def _as_date(d) -> _date:
    return _date.fromisoformat(d) if isinstance(d, str) else d


def _spinup_start(start: _date) -> _date:
    """First simulated day: 1 January, ``SPINUP_YEARS`` calendar years back.

    Snapping to 1 January means the actual lead-in is between ``SPINUP_YEARS``
    and ``SPINUP_YEARS + 1`` years (a June start gets ~2.4), and keeps the
    Query stub -- hence the SILO cache -- shared by every request in a year.
    """
    return _date(start.year - SPINUP_YEARS, 1, 1)


def _tag(*parts) -> str:
    return "_".join(str(p) for p in parts).replace(".", "p").replace("-", "m")


def _load(model, model_name: str):
    if model is not None:
        return model
    est = load_model(model_name)
    if est is None:
        raise FileNotFoundError(
            f"no trained model at data/models/{model_name}.joblib — fit one with "
            f"`python -m emt.model7.build` then `emt.model8.model.fit`.")
    return est


def _silo_series(q: Query) -> pd.DataFrame:
    """Daily rain/PET for a Query's centre, indexed by date."""
    s = download_silo(q).rename(columns=lambda c: "time" if c.startswith("YYYY") else c)
    s["time"] = pd.to_datetime(s["time"])
    return s.set_index("time")[[_RAIN, _PET]].sort_index()


def _cap_ratio(model, awc) -> np.ndarray | None:
    """Per-location capacity ratio (smax multiplier) from SLGA AWC.

    The fitted model normalises capacity by its training-station mean AWC
    (``cap_train_mean_``); a new location's ratio is its own AWC over that
    same constant. ``None`` (capacity off, e.g. an older fit) or non-finite
    AWC (e.g. water pixels) falls back to the global ``smax``.
    """
    train_mean = getattr(model, "cap_train_mean_", None)
    if train_mean is None:
        return None
    awc = np.asarray(awc, dtype=float)
    return np.where(np.isfinite(awc), awc / train_mean, 1.0)


def _simulate(rain: np.ndarray, pet: np.ndarray, model,
              cap_ratio: np.ndarray | None = None) -> np.ndarray:
    """Bucket storage (days, n) from forcing (days, n) using the fitted params.

    ``cap_ratio`` (per column) scales the capacity: ``smax_i = smax * ratio_i``
    -- the inference counterpart of the estimator's ``capacity=`` input. The
    readout keeps the global ``smax`` denominator (see ``Forcing.vwc``).
    """
    smax, alpha, k = model.bucket_params
    rain = np.ascontiguousarray(rain, dtype=float)
    pet = np.ascontiguousarray(pet, dtype=float)
    smax_i = (np.full(rain.shape[1], smax) if cap_ratio is None
              else smax * np.asarray(cap_ratio, dtype=float))
    return _step_loop(rain, pet, smax_i, alpha, k)


def _aridity(silo: pd.DataFrame) -> float:
    """Aridity normal (mean P / mean PET) over a fetched SILO series.

    Training used the 2005-2010 forcing; at inference the normal is estimated
    over the fetched window (spin-up start -> end, >= 2 calendar years). P/PET
    is a stable ratio, so the window difference moves the static by well under
    one training standard deviation.

    One consequence worth knowing: because the window depends on the requested
    period, the *same* day predicted with a different ``--end`` gets a very
    slightly different aridity static, hence a slightly different level. The
    effect is tiny (measured ~0.001 % VWC between a 3-day and a 5-day request,
    with storage bit-identical) but it is not exactly zero.
    """
    return float(silo[_RAIN].mean() / silo[_PET].mean())


def _pedo_limits(model, clay, sand):
    """Per-location ``(theta_r, dtheta)`` for a pedotransfer-readout fit.

    Returns ``None`` for the 5-parameter fits (model7/model8), whose readout
    constants are global and already in ``params_``. For model9 the limits are
    rebuilt per pixel from the same SLGA clay/sand the statics come from, using
    the span the estimator was built with.
    """
    if getattr(model, "_n_process", 5) != 4:
        return None
    from emt.pedotransfer import wilting_point, field_capacity, saturation
    wp = wilting_point(clay, sand)
    top = (saturation(clay, sand) if getattr(model, "readout_span_", "available")
           == "saturation" else field_capacity(clay, sand))
    return (wp, top - wp)


def _statics_at_point(q: Query, lon: float, lat: float, model,
                      silo: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Statics at one point in the *fitted* model's variable order, + raw values.

    Sources per variable: SLGA rasters (soil), the 30 m DEM stack (terrain),
    and the SILO series (``aridity``). Driven by ``model._static_vars`` so an
    older fit (no aridity) keeps working unchanged. The raw dict carries the
    soil values inference needs beyond the statics themselves (AWC for the
    capacity ratio, clay/sand for a pedotransfer readout).
    """
    soil, terr = soil_covariates(q), terrain_covariates(q)
    src = {**{v: soil[v] for v in SOIL_VARS},
           **{v: terr[v] for v in TERRAIN_STATIC_VARS}}
    vals = {v: float(sample_points(src[v], lon, lat).values) for v in src}
    vals["aridity"] = _aridity(silo)
    return np.array([[vals[v] for v in model._static_vars]]), vals


# --------------------------------------------------------------------------- #
# Point time series -- any date range
# --------------------------------------------------------------------------- #
def predict_point(lat: float, lon: float, start, end=None, model=None,
                  model_name: str = "model8", buffer_km: float = 1.5,
                  verbose: bool = True) -> pd.DataFrame:
    """Daily root-zone soil moisture (%) at one location over any date range.

    Returns a DataFrame ``[time, sm_pred, storage_mm]`` covering
    ``[start, end]`` (``end`` defaults to ``start``, i.e. a single day).
    """
    start = _as_date(start)
    end = _as_date(end) if end is not None else start
    model = _load(model, model_name)
    if verbose:
        print(WARNING, flush=True)

    sim_start = _spinup_start(start)
    q = Query.from_lat_lon(lat=lat, lon=lon, buffer_km=buffer_km,
                           start=sim_start, end=end,
                           stub=_tag("m8pt", f"{lat:.4f}", f"{lon:.4f}",
                                     sim_start, end))
    if verbose:
        print(f"  SILO forcing {sim_start} → {end} "
              f"({SPINUP_YEARS}-year spin-up) ...", flush=True)
    silo = _silo_series(q)
    if verbose:
        print("  SLGA soil + terrain ...", flush=True)
    statics, vals = _statics_at_point(q, lon, lat, model, silo)

    storage = _simulate(silo[[_RAIN]].to_numpy(), silo[[_PET]].to_numpy(), model,
                        cap_ratio=_cap_ratio(model, [vals["soil_awc"]]))
    lim = _pedo_limits(model, vals["soil_clay"], vals["soil_sand"])
    vwc = model.readout(storage[:, 0], np.repeat(statics, len(silo), axis=0),
                        limits=lim)

    out = pd.DataFrame({"time": silo.index, "sm_pred": vwc,
                        "storage_mm": storage[:, 0]})
    out = out[(out["time"] >= pd.Timestamp(start)) & (out["time"] <= pd.Timestamp(end))]
    if verbose:
        print(f"  {len(out)} days, mean {out['sm_pred'].mean():.1f}%", flush=True)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 30 m map -- any day
# --------------------------------------------------------------------------- #
def _forcing_grid(bbox, sim_start: _date, day: _date, step_deg: float,
                  verbose: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Daily rain/PET on a lon/lat grid over ``bbox`` -> (rain, pet, lons, lats).

    One cached SILO point download per cell (the open gridded NetCDFs are
    chunked by time, so a small-bbox subset would pull continental grids --
    the same reasoning as :func:`emt.antecedent.antecedent_grid`).
    """
    minx, miny, maxx, maxy = bbox

    def axis(lo, hi):
        n = max(2, int(np.ceil((hi - lo) / step_deg)) + 1)
        return np.round(np.linspace(lo, hi, n), 4)

    lons, lats = axis(minx, maxx), axis(miny, maxy)
    if verbose:
        print(f"  SILO forcing grid {len(lats)}x{len(lons)} "
              f"({len(lats) * len(lons)} cells) @ {step_deg}deg ...", flush=True)

    cols, times = [], None
    for la in lats:
        for lo in lons:
            q = Query.from_lat_lon(lat=float(la), lon=float(lo), buffer_km=1.0,
                                   start=sim_start, end=day,
                                   stub=_tag("m8grid", f"{la:.4f}", f"{lo:.4f}",
                                             sim_start, day))
            s = _silo_series(q)
            times = s.index if times is None else times
            cols.append(s.reindex(times))
    rain = np.column_stack([c[_RAIN].to_numpy() for c in cols])
    pet = np.column_stack([c[_PET].to_numpy() for c in cols])
    aridity = np.array([_aridity(c) for c in cols])
    return rain, pet, aridity, lons, lats


def predict_map(bbox, day, model=None, model_name: str = "model8",
                step_deg: float = GRID_STEP_DEG, save: bool = True,
                plot: bool = True, verbose: bool = True) -> xr.Dataset:
    """A 30 m root-zone soil-moisture field over ``bbox`` for any ``day``.

    The water balance runs on the ~``step_deg`` SILO forcing grid; the 30 m
    structure comes from the per-pixel soil + terrain offsets. Returns an
    ``xr.Dataset`` with ``sm_pred`` (%) and ``storage_mm``; when saved, paths
    are recorded in ``ds.attrs['output_tif' / 'output_png']``.
    """
    day = _as_date(day)
    model = _load(model, model_name)
    if verbose:
        print(WARNING, flush=True)

    sim_start = _spinup_start(day)
    q = Query(bbox=list(bbox), start=day, end=day,
              stub=_tag("m8map", *[f"{v:.3f}" for v in bbox], day))

    if verbose:
        print("  terrain (30 m) ...", flush=True)
    terr = terrain_covariates(q)
    grid = terr["elevation"]
    if verbose:
        print("  SLGA soil ...", flush=True)
    soil = soil_covariates(q)

    rain, pet, aridity, lons, lats = _forcing_grid(bbox, sim_start, day, step_deg,
                                                   verbose)

    # Capacity per forcing cell: SLGA AWC sampled at the cell centre (the
    # water balance runs at forcing scale, matching the station-scale fit).
    awc_cells = np.array([float(sample_points(soil["soil_awc"], float(lo),
                                              float(la)).values)
                          for la in lats for lo in lons])
    storage = _simulate(rain, pet, model,
                        cap_ratio=_cap_ratio(model, awc_cells))[-1]
    storage = storage.reshape(len(lats), len(lons))
    coarse = (xr.DataArray(storage, coords={"y": lats, "x": lons}, dims=("y", "x"))
              .rio.write_crs(4326))
    fine = coarse.rio.reproject_match(grid, resampling=Resampling.bilinear)

    # Static layers per pixel, in the fitted model's variable order: soil and
    # terrain from their rasters (nearest, per-pixel identity); the aridity
    # normal from the forcing grid (bilinear -- a smooth climate field).
    src = {**{v: (soil[v], Resampling.nearest) for v in SOIL_VARS},
           **{v: (terr[v], Resampling.nearest) for v in TERRAIN_STATIC_VARS},
           "aridity": (xr.DataArray(aridity.reshape(len(lats), len(lons)),
                                    coords={"y": lats, "x": lons},
                                    dims=("y", "x")).rio.write_crs(4326),
                       Resampling.bilinear)}
    stat_cols = []
    for v in model._static_vars:
        a, resamp = src[v]
        if a.rio.crs is None:
            a = a.rio.write_crs(4326)
        stat_cols.append(a.rio.reproject_match(grid, resampling=resamp)
                         .values.ravel())
    S = np.column_stack(stat_cols)
    stor = fine.values.ravel()

    # Pedotransfer limits per pixel (model9); None for the 5-param fits.
    clay = (soil["soil_clay"].rio.reproject_match(grid, resampling=Resampling.nearest)
            .values.ravel())
    sand = (soil["soil_sand"].rio.reproject_match(grid, resampling=Resampling.nearest)
            .values.ravel())
    lim = _pedo_limits(model, clay, sand)

    valid = np.isfinite(S).all(axis=1) & np.isfinite(stor)
    pred = np.full(stor.shape, np.nan, dtype="float32")
    if valid.any():
        lim_v = None if lim is None else (lim[0][valid], lim[1][valid])
        pred[valid] = model.readout(stor[valid], S[valid],
                                    limits=lim_v).astype("float32")
    if verbose:
        print(f"  predicted {int(valid.sum()):,} pixels, mean "
              f"{np.nanmean(pred):.1f}%", flush=True)

    ds = xr.Dataset({"sm_pred": (grid.dims, pred.reshape(grid.shape)),
                     "storage_mm": (grid.dims, fine.values.astype("float32"))},
                    coords=grid.coords).rio.write_crs(grid.rio.crs)

    if save:
        outdir = Path(q.out_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        tif = outdir / f"soil_moisture_model8_{day}.tif"
        ds["sm_pred"].rio.to_raster(tif)
        ds.attrs["output_tif"] = str(tif)
        if verbose:
            print(f"  saved {tif}", flush=True)
        if plot:
            from emt.predict import plot_field
            png = outdir / f"soil_moisture_model8_{day}.png"
            plot_field(ds, png, title=f"Root-zone soil moisture, {day} (30 m, model8)")
            ds.attrs["output_png"] = str(png)
            if verbose:
                print(f"  saved {png}", flush=True)
    return ds


def main():
    ap = argparse.ArgumentParser(
        description="Run model8 (process model) for any date: point series or 30 m map.")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                    help="map mode: lon/lat bounds (EPSG:4326)")
    ap.add_argument("--date", help="map mode: YYYY-MM-DD")
    ap.add_argument("--lat", type=float, help="point mode: latitude")
    ap.add_argument("--lon", type=float, help="point mode: longitude")
    ap.add_argument("--start", help="point mode: first day (YYYY-MM-DD)")
    ap.add_argument("--end", help="point mode: last day (default: --start)")
    ap.add_argument("--model", default="model8")
    ap.add_argument("--step-deg", type=float, default=GRID_STEP_DEG,
                    help="map mode: forcing-grid spacing (default %(default)s)")
    ap.add_argument("-o", "--out", default=None,
                    help="also write the GeoTIFF (map) or CSV (point) here")
    ap.add_argument("--no-plot", action="store_true", help="map mode: skip the PNG")
    a = ap.parse_args()

    if a.bbox and a.date:
        ds = predict_map(tuple(a.bbox), a.date, model_name=a.model,
                         step_deg=a.step_deg, plot=not a.no_plot)
        print(f"wrote {ds.attrs['output_tif']}")
        if "output_png" in ds.attrs:
            print(f"wrote {ds.attrs['output_png']}")
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            ds["sm_pred"].rio.to_raster(a.out)
            print(f"also wrote {a.out}")
    elif a.lat is not None and a.lon is not None and a.start:
        s = predict_point(a.lat, a.lon, a.start, a.end, model_name=a.model)
        print(s.to_string(index=False) if len(s) <= 20 else s.head(10).to_string(index=False))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            s.to_csv(a.out, index=False)
            print(f"wrote {a.out}")
    else:
        ap.error("give either --bbox with --date (map), or --lat/--lon/--start (point)")


if __name__ == "__main__":
    main()
