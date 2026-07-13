"""Antecedent-meteorology features from SILO (last week / month / year).

Soil moisture is driven by how much water has recently arrived versus left.
These trailing-window features summarise that history at each station-day:

    rain_7, rain_30, rain_365     trailing rainfall sums (mm)
    ppet_30, ppet_365             trailing water balance, P - PET (mm)
    vpd_30                        trailing mean vapour-pressure deficit
    rain_365_anom                 rain_365 minus the pixel's mean 365 d rain
                                  (a simple drought / wet-year index)

All are *dynamic* (a per-pixel time series, not a static site value) and come
from SILO, a national ~5 km daily grid, so they are computable at every 30 m
pixel at inference and cannot memorise station identity -- the same leakage test
the other features pass.

SILO is fetched via PaddockTS (:func:`download_silo`), extended one year before
the study start so the 365-day window is complete at every training date.
"""
from __future__ import annotations

from datetime import date
import glob

import pandas as pd

from emt.queries import query_for_station
from PaddockTS.Environmental.SILO.download_silo import download_silo, get_filename

ANTECEDENT_VARS = ["rain_7", "rain_30", "rain_365",
                   "ppet_30", "ppet_365", "vpd_30", "rain_365_anom"]

# SILO columns used (rainfall, Morton potential ET, vapour-pressure deficit).
_RAIN, _PET, _VPD = "daily_rain", "et_morton_potential", "vp_deficit"


def _trailing(silo: pd.DataFrame) -> pd.DataFrame:
    """Trailing-window features from a station's daily SILO frame."""
    s = silo.copy()
    s = s.rename(columns={s.columns[0]: "time"}) if "time" not in s.columns else s
    s["time"] = pd.to_datetime(s["time"])
    s = s.set_index("time").sort_index()
    rain, ppet, vpd = s[_RAIN], s[_RAIN] - s[_PET], s[_VPD]
    out = pd.DataFrame({
        "rain_7":   rain.rolling(7,   min_periods=4).sum(),
        "rain_30":  rain.rolling(30,  min_periods=15).sum(),
        "rain_365": rain.rolling(365, min_periods=180).sum(),
        "ppet_30":  ppet.rolling(30,  min_periods=15).sum(),
        "ppet_365": ppet.rolling(365, min_periods=180).sum(),
        "vpd_30":   vpd.rolling(30,   min_periods=15).mean(),
    })
    # As-of-date drought anomaly: this year's 365-day rain minus the mean of all
    # PRIOR days' 365-day totals (expanding, shifted) -- never the full-period
    # mean, which would peek at the future.
    past_mean = out["rain_365"].shift(1).expanding(min_periods=180).mean()
    out["rain_365_anom"] = out["rain_365"] - past_mean
    return out.reset_index()


def _station_silo(stn: str, lat: float, lon: float,
                  start: date, end: date) -> pd.DataFrame | None:
    """Fetch (cached) SILO for one station over [start-1yr, end]."""
    q = query_for_station(stn, lat, lon, date(start.year - 1, 1, 1), end)
    # reuse an existing cache file if present (either extended or study-period).
    for pat in (get_filename(q),
                f"{q.tmp_dir}/Environmental/oznet_{stn}_*_silo.csv"):
        hit = glob.glob(pat)
        if hit:
            return pd.read_csv(sorted(hit)[-1])
    try:
        return download_silo(q)
    except Exception as e:                       # noqa: BLE001
        print(f"  SILO {stn}: FAIL {type(e).__name__}: {e}", flush=True)
        return None


def antecedent_grid(query, start: date, end: date, step_deg: float = 0.05,
                    reload: bool = False, verbose: bool = True) -> "object":
    """Gridded antecedent-meteorology cube over ``query.bbox`` (for inference).

    Samples SILO on a regular ``step_deg`` grid over the AOI using the fast
    point path (:func:`download_silo`, one cell per point, cached), computes the
    trailing-window features per cell, and assembles an ``xr.Dataset``
    (``time, y, x``, EPSG:4326) with ``ANTECEDENT_VARS``. ``step_deg`` defaults
    to SILO's native 0.05°. Cached to NetCDF.

    (The open gridded NetCDFs are chunked by time, so byte-range subsetting a
    small bbox would pull whole daily continent grids — the point path is far
    cheaper for an AOI. A true national run would read the grids locally.)

    Use :func:`antecedent_day_layers` to pull one day's fields for
    :func:`emt.downscale.downscale`.
    """
    import numpy as np
    import xarray as xr
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    from os import makedirs
    from os.path import exists

    fn = f"{query.tmp_dir}/Environmental/{query.stub}_antecedent_grid.nc"
    if not reload and exists(fn):
        if verbose:
            print(f"  cached: {fn}", flush=True)
        with xr.open_dataset(fn) as ds:
            return ds.load()

    minx, miny, maxx, maxy = query.bbox
    lons = np.round(np.arange(minx, maxx + step_deg / 2, step_deg), 4)
    lats = np.round(np.arange(miny, maxy + step_deg / 2, step_deg), 4)
    if verbose:
        print(f"  SILO point-grid {len(lats)}x{len(lons)} "
              f"({len(lats)*len(lons)} cells) @ {step_deg}deg", flush=True)

    times = pd.date_range(start, end, freq="D")
    grids = {v: np.full((len(times), len(lats), len(lons)), np.nan) for v in ANTECEDENT_VARS}
    for iy, lat in enumerate(lats):
        for ix, lon in enumerate(lons):
            silo = _station_silo(f"grid_{lat:.3f}_{lon:.3f}".replace("-", "m").replace(".", "p"),
                                 float(lat), float(lon), start, end)
            if silo is None:
                continue
            tr = _trailing(silo).set_index("time").reindex(times)
            for v in ANTECEDENT_VARS:
                grids[v][:, iy, ix] = tr[v].values
        if verbose:
            print(f"    row {iy+1}/{len(lats)}", flush=True)

    ante = xr.Dataset(
        {v: (("time", "y", "x"), grids[v]) for v in ANTECEDENT_VARS},
        coords={"time": times, "y": lats, "x": lons},
    ).rio.write_crs(4326)
    makedirs(f"{query.tmp_dir}/Environmental", exist_ok=True)
    ante.to_netcdf(fn)
    if verbose:
        print(f"  saved: {fn}", flush=True)
    return ante


def antecedent_day_layers(cube: "object", day) -> dict:
    """One day's antecedent fields as ``{var: DataArray}`` for ``extra_layers``."""
    import pandas as _pd
    sl = cube.sel(time=_pd.Timestamp(day), method="nearest")
    return {v: sl[v] for v in ANTECEDENT_VARS}


def add_antecedent(table: pd.DataFrame, coords: pd.DataFrame,
                   start: date, end: date, verbose: bool = True) -> pd.DataFrame:
    """Attach ``ANTECEDENT_VARS`` to ``table`` (one row per station-day)."""
    coords = coords.set_index("station") if "station" in coords.columns else coords
    tab = table.copy()
    tab["time"] = pd.to_datetime(tab["time"])
    frames = []
    for stn in tab["station"].unique():
        silo = _station_silo(stn, float(coords.loc[stn, "lat"]),
                             float(coords.loc[stn, "lon"]), start, end)
        if silo is None:
            continue
        f = _trailing(silo)
        f["station"] = stn
        frames.append(f)
        if verbose:
            print(f"  antecedent {stn}: {len(f)} days", flush=True)
    if not frames:
        return tab
    ante = pd.concat(frames, ignore_index=True)
    return tab.merge(ante, on=["station", "time"], how="left")
