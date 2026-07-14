"""EMT-local SMIPS loader (raw TotalBucket via TERN GeoServer WCS).

Why this exists instead of ``PaddockTS.Environmental.SMIPS.download_smips``:
PaddockTS points at the old ``landscapes-mapserver.tern.org.au`` WMS, which TERN
has decommissioned (404). The replacement GeoServer *WMS* only serves a styled
8-bit palette image, not raw values. The raw daily Cloud-Optimised GeoTIFFs at
``data.tern.org.au`` require a TERN login.

The GeoServer **WCS** endpoint, however, is public and returns the raw float32
``TotalBucket`` soil-water field (mm) with a queryable ``time`` axis -- so EMT
reads SMIPS from there. Everything else (the AOI/cache) still goes through the
PaddockTS :class:`Query`.

Endpoint:
    https://geoserver.tern.org.au/geoserver/landscapes/smips/ows
    coverage ``landscapes__smips_totalbucket`` (also ``landscapes__smips_smindex``)
"""
from __future__ import annotations

import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from os import makedirs
from os.path import exists

import pandas as pd
import requests
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from PaddockTS.query import Query

WCS_URL = "https://geoserver.tern.org.au/geoserver/landscapes/smips/ows"
COVERAGES = {
    "totalbucket": "landscapes__smips_totalbucket",   # total profile soil water (mm)
    "smindex": "landscapes__smips_smindex",            # 0-1 saturation index
}
TIMEOUT = 90

# SMIPS native grid (EPSG:4326), from the WCS DescribeCoverage: envelope outer
# edges + per-pixel offset vectors. GeoServer resamples every WCS GetCoverage to
# fit an INTEGER pixel count into the *requested* envelope, so a point's sampled
# value depends on the request window (a narrow per-station box and a wide
# cluster box land on differently-shifted grids and can pick different ~1 km
# cells -- observed up to ~40% disagreement). Snapping the request bbox to these
# native cell boundaries removes the resampling: each window returns the true
# native pixels, so sampling is window-independent and per-station == cluster.
NATIVE_WEST = 112.90499114990234
NATIVE_SOUTH = -43.73500061035156
NATIVE_DX = 0.009997566018978103    # Long step
NATIVE_DY = 0.009997121616580312    # Lat step (magnitude)


def snap_bbox(bbox, pad: int = 1):
    """Expand a ``(minx, miny, maxx, maxy)`` bbox outward to native SMIPS grid
    lines, plus ``pad`` cells of margin. Idempotent up to the added padding, so
    call it once per request (not inside the per-day fetch)."""
    minx, miny, maxx, maxy = bbox
    i0 = math.floor((minx - NATIVE_WEST) / NATIVE_DX) - pad
    i1 = math.ceil((maxx - NATIVE_WEST) / NATIVE_DX) + pad
    j0 = math.floor((miny - NATIVE_SOUTH) / NATIVE_DY) - pad
    j1 = math.ceil((maxy - NATIVE_SOUTH) / NATIVE_DY) + pad
    return [NATIVE_WEST + i0 * NATIVE_DX, NATIVE_SOUTH + j0 * NATIVE_DY,
            NATIVE_WEST + i1 * NATIVE_DX, NATIVE_SOUTH + j1 * NATIVE_DY]

get_filename = lambda q, var: f"{q.tmp_dir}/Environmental/{q.stub}_smips_{var}.nc"


def _fetch_geotiff(d: date, bbox, coverage: str) -> bytes:
    """WCS GetCoverage for one day over ``bbox`` -> raw GeoTIFF bytes (raises on error)."""
    minx, miny, maxx, maxy = bbox
    # The caller is expected to have snapped ``bbox`` to the native grid (see
    # snap_bbox); scaleFactor is intentionally NOT sent -- it is a no-op on this
    # GeoServer (identical output with and without it) and does not control
    # resampling. Grid alignment is what makes sampling window-independent.
    params = [
        ("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
        ("coverageId", coverage), ("format", "image/geotiff"),
        ("subset", f"Long({minx},{maxx})"),
        ("subset", f"Lat({miny},{maxy})"),
        ("subset", f'time("{d.isoformat()}T00:00:00.000Z")'),
    ]
    r = requests.get(WCS_URL, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    if not (r.content.startswith(b"II") or r.content.startswith(b"MM")):
        snippet = r.content[:400].decode("utf-8", errors="replace")
        raise RuntimeError(f"SMIPS WCS error for {d}: {snippet}")
    return r.content


def smips_day(d: date | str, bbox, var: str = "totalbucket") -> xr.DataArray:
    """Fetch the raw SMIPS field over ``bbox`` for one day as a 2D DataArray (mm)."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    tiff = _fetch_geotiff(d, snap_bbox(bbox), COVERAGES[var])
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(tiff)
        tmp_path = tmp.name
    try:
        da = rioxarray.open_rasterio(tmp_path, masked=True).squeeze("band", drop=True).load()
    finally:
        os.unlink(tmp_path)
    return da


def smips_cube(start: date | str, end: date | str, bbox,
               var: str = "totalbucket", workers: int = 8,
               skip_missing: bool = True, days: list[date] | None = None) -> xr.DataArray:
    """Fetch a ``(time, y, x)`` raw SMIPS cube over ``bbox`` (inclusive of both ends).

    ``days`` overrides the daily range with an explicit list of dates (used by
    :func:`smips_climatology` to fetch a thinned sample).
    """
    if days is None:
        days = [d.date() for d in pd.date_range(start, end, freq="D")]
    slices: dict[date, xr.DataArray] = {}

    def fetch(d):
        return d, smips_day(d, bbox, var=var)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch, d): d for d in days}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                _, da = fut.result()
                slices[d] = da
            except (requests.RequestException, RuntimeError) as e:
                if skip_missing:
                    print(f"  [{d}] {type(e).__name__}: {e}")
                else:
                    raise

    if not slices:
        raise RuntimeError("No SMIPS days returned data.")

    ordered = sorted(slices.items())
    times = pd.to_datetime([d for d, _ in ordered])
    cube = xr.concat([da for _, da in ordered], dim=pd.Index(times, name="time"))
    cube.name = f"smips_{var}"
    cube.attrs.update(
        source="TERN SMIPS via GeoServer WCS", doi="10.25901/b020-nm39",
        license="CC-BY 4.0", endpoint=WCS_URL, coverage=COVERAGES[var], units="mm",
    )
    return cube


def download_smips(query: Query, var: str = "totalbucket", workers: int = 8,
                   reload: bool = False) -> xr.DataArray:
    """Download (and cache) the raw SMIPS cube for ``query``.

    Args:
        query: PaddockTS :class:`Query` (provides bbox, dates, and cache dir).
        var: ``"totalbucket"`` (mm, the downscaling target) or ``"smindex"``.
        workers: Concurrent per-day WCS requests.
        reload: If True, ignore any cached file and refetch.

    Returns:
        ``(time, y, x)`` DataArray of raw SMIPS values, also written to
        ``{query.tmp_dir}/Environmental/{query.stub}_smips_{var}.nc``.
    """
    filename = get_filename(query, var)
    if not reload and exists(filename):
        print(f"  cached: {filename}")
        with xr.open_dataset(filename) as ds:
            name = [v for v in ds.data_vars if v != "spatial_ref"][0]
            return ds[name].load()

    makedirs(f"{query.tmp_dir}/Environmental", exist_ok=True)
    print(f"  fetching SMIPS ({var}) for bbox {query.bbox} ({query.start} → {query.end})...", flush=True)
    cube = smips_cube(query.start, query.end, tuple(query.bbox), var=var, workers=workers).compute()
    cube.to_dataset(name=cube.name).to_netcdf(filename)
    print(f"  saved: {filename} ({cube.sizes['time']} days, "
          f"{cube.sizes.get('y')}x{cube.sizes.get('x')} px)")
    return cube


def smips_lookback_day(query: Query, day, var: str = "totalbucket",
                       windows=(7, 30, 365), workers: int = 8) -> dict:
    """SMIPS lookback rasters over the AOI, as of ``day`` (for inference).

    Fetches the SMIPS cube for ``[day - max(windows), day]`` over ``query.bbox``
    and returns the trailing means ending at ``day`` — the inference-side match to
    the training features ``smips_7d/30d/365d`` — plus ``smips_totalbucket`` (the
    day's value) and ``smips_anom`` (day minus the past-year mean). Every window
    looks strictly backward from ``day``.

    Returns ``{name: DataArray}`` on the SMIPS grid (EPSG:4326), ready for
    ``downscale``/``predict`` ``extra_layers``.
    """
    day = pd.Timestamp(day)
    start = (day - pd.Timedelta(days=max(windows))).date()
    cube = smips_cube(start, day.date(), tuple(query.bbox), var=var,
                      workers=workers).sortby("time")
    out = {}
    for w in windows:
        out[f"smips_{w}d"] = cube.isel(time=slice(-w, None)).mean("time")
    today = cube.isel(time=-1)                     # nearest available day <= day
    out["smips_totalbucket"] = today
    out["smips_anom"] = today - out["smips_365d"]
    return {k: v.rio.write_crs(4326) for k, v in out.items()}


def smips_climatology(query: Query, var: str = "totalbucket", step_days: int = 5,
                      workers: int = 8, reload: bool = False) -> xr.Dataset:
    """Per-pixel SMIPS mean/std over the query period (the model4 level features).

    Fetches every ``step_days``-th day over ``[query.start, query.end]`` and
    reduces over time. Thinning keeps the request count tractable for multi-year
    AOI climatologies (a 5-year period at ``step_days=5`` is ~365 samples per
    pixel -- ample for a stable mean/std). Cached like the daily cubes.

    Returns:
        Dataset with ``smips_mean_px`` and ``smips_std_px`` on the SMIPS grid.
    """
    filename = (f"{query.tmp_dir}/Environmental/"
                f"{query.stub}_smips_{var}_clim{step_days}.nc")
    if not reload and exists(filename):
        print(f"  cached: {filename}")
        with xr.open_dataset(filename) as ds:
            return ds.load()

    makedirs(f"{query.tmp_dir}/Environmental", exist_ok=True)
    days = [d.date() for d in
            pd.date_range(query.start, query.end, freq=f"{step_days}D")]
    print(f"  fetching SMIPS climatology ({var}) for bbox {query.bbox}: "
          f"{len(days)} sample days...", flush=True)
    cube = smips_cube(days[0], days[-1], tuple(query.bbox), var=var,
                      workers=workers, days=days)
    clim = xr.Dataset({"smips_mean_px": cube.mean("time"),
                       "smips_std_px": cube.std("time")})
    clim.attrs.update(cube.attrs, step_days=step_days,
                      period=f"{query.start}..{query.end}", n_samples=len(days))
    clim.to_netcdf(filename)
    print(f"  saved: {filename}")
    return clim


def test():
    from PaddockTS.query import Query
    q = Query.from_lat_lon(-35.41928, 147.60408, 2.0,
                           date(2020, 6, 1), date(2020, 6, 7), stub="SMIPS_TEST")
    cube = download_smips(q)
    print(cube)
    print("mean per day (mm):\n", cube.mean(("x", "y")).to_pandas().round(2))


if __name__ == "__main__":
    test()
