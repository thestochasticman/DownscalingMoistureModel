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
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from os import makedirs
from os.path import exists
from pathlib import Path

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


UNAVAILABLE_ATTR = "unavailable_days"


def missing_days(cube: xr.DataArray, start: date, end: date) -> list[date]:
    """Days in ``[start, end]`` absent from ``cube``, minus the known-unavailable.

    ``smips_cube`` drops any day whose request fails (``skip_missing``), so a
    cube fetched at high concurrency can carry the odd hole from a transient
    ``RemoteDisconnected``. Days recorded in the cube's ``unavailable_days``
    attribute are excluded: those were retried and are genuinely absent
    upstream, so re-requesting them on every load would cost a round trip per
    call forever.
    """
    have = pd.DatetimeIndex(cube["time"].values).normalize()
    known = str(cube.attrs.get(UNAVAILABLE_ATTR, ""))
    skip = {d for d in known.split(",") if d}
    return [d.date() for d in pd.date_range(start, end, freq="D").difference(have)
            if d.date().isoformat() not in skip]


def repair_cube(cube: xr.DataArray, bbox, start: date, end: date,
                var: str = "totalbucket", workers: int = 8) -> tuple[xr.DataArray, int]:
    """Refetch any day missing from ``cube`` and merge it in.

    Returns ``(cube, n_recovered)``. Days that fail again are recorded in the
    ``unavailable_days`` attribute so later calls skip them. Refetching uses the
    cube's original ``bbox``, so the returned day lands on the same snapped grid;
    it is aligned onto the cube's exact coordinates as a guard.
    """
    gaps = missing_days(cube, start, end)
    if not gaps:
        return cube, 0

    print(f"  repairing {len(gaps)} missing day(s): "
          f"{[d.isoformat() for d in gaps[:5]]}"
          f"{' ...' if len(gaps) > 5 else ''}", flush=True)
    template = cube.isel(time=0, drop=True)
    got, failed = [], []
    for d in gaps:
        try:
            new = smips_day(d, bbox, var=var)
        except (requests.RequestException, RuntimeError) as e:
            print(f"    {d}: unavailable ({type(e).__name__})")
            failed.append(d)
            continue
        new = new.reindex_like(template, method="nearest", tolerance=1e-6)
        if new.shape != template.shape:
            print(f"    {d}: grid mismatch {new.shape} vs {template.shape}, skipped")
            failed.append(d)
            continue
        got.append(new.expand_dims(time=[pd.Timestamp(d)]))

    attrs = dict(cube.attrs)
    if failed:
        known = [s for s in str(attrs.get(UNAVAILABLE_ATTR, "")).split(",") if s]
        attrs[UNAVAILABLE_ATTR] = ",".join(sorted(set(known) | {d.isoformat() for d in failed}))
    if not got:
        cube.attrs = attrs
        return cube, 0

    name = cube.name
    cube = xr.concat([cube, *got], dim="time").sortby("time")
    cube.name = name
    cube.attrs = attrs
    print(f"    recovered {len(got)} day(s) -> {cube.sizes['time']} days", flush=True)
    return cube, len(got)


def download_smips(query: Query, var: str = "totalbucket", workers: int = 8,
                   reload: bool = False, repair: bool = True) -> xr.DataArray:
    """Download (and cache) the raw SMIPS cube for ``query``.

    Args:
        query: PaddockTS :class:`Query` (provides bbox, dates, and cache dir).
        var: ``"totalbucket"`` (mm, the downscaling target) or ``"smindex"``.
        workers: Concurrent per-day WCS requests.
        reload: If True, ignore any cached file and refetch.
        repair: Fill day-gaps before returning, whether the cube was just
            fetched or loaded from cache, rewriting the cache when anything is
            recovered (see :func:`repair_cube`). On a complete cube this is an
            index comparison and costs nothing, so it is on by default -- a
            silently short cube otherwise propagates into the training table as
            quietly dropped station-days.

    Returns:
        ``(time, y, x)`` DataArray of raw SMIPS values, also written to
        ``{query.tmp_dir}/Environmental/{query.stub}_smips_{var}.nc``.
    """
    filename = get_filename(query, var)
    if not reload and exists(filename):
        print(f"  cached: {filename}")
        with xr.open_dataset(filename) as ds:
            name = [v for v in ds.data_vars if v != "spatial_ref"][0]
            cube = ds[name].load()
        if repair:
            cube, n = repair_cube(cube, tuple(query.bbox), query.start, query.end,
                                  var=var, workers=workers)
            if n:
                cube.to_dataset(name=cube.name).to_netcdf(filename)
        return cube

    makedirs(f"{query.tmp_dir}/Environmental", exist_ok=True)
    print(f"  fetching SMIPS ({var}) for bbox {query.bbox} ({query.start} → {query.end})...", flush=True)
    cube = smips_cube(query.start, query.end, tuple(query.bbox), var=var, workers=workers).compute()
    if repair:
        cube, _ = repair_cube(cube, tuple(query.bbox), query.start, query.end,
                              var=var, workers=workers)
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


def smips_lookback_series(query: Query, days, var: str = "totalbucket",
                          windows=(7, 30, 365), workers: int = 8) -> dict:
    """Per-day SMIPS lookback layers for several ``days``, fetching ONE cube.

    Fetches the SMIPS cube once over ``[min(days) - max(windows), max(days)]`` and
    slices each day's trailing means from it — the multi-date equivalent of
    :func:`smips_lookback_day`, used by the seasonal downscale galleries so a
    9-date gallery makes one ~2-year fetch instead of nine 1-year fetches.

    Returns ``{date: {smips_7d, smips_30d, smips_365d, smips_anom}}`` (each a
    DataArray on the SMIPS grid; ``smips_totalbucket`` is omitted — ``downscale``
    sets it from the day's SMIPS). Every window looks strictly backward.
    """
    ds = sorted(pd.Timestamp(d) for d in days)
    start = (ds[0] - pd.Timedelta(days=max(windows))).date()
    cube = smips_cube(start, ds[-1].date(), tuple(query.bbox), var=var,
                      workers=workers).sortby("time")
    out: dict = {}
    for d in ds:
        upto = cube.sel(time=slice(None, d))
        layers = {f"smips_{w}d": upto.isel(time=slice(-w, None)).mean("time")
                  for w in windows}
        layers["smips_anom"] = upto.isel(time=-1) - layers["smips_365d"]
        out[d.date()] = {k: v.rio.write_crs(4326) for k, v in layers.items()}
    return out


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


_STUB_DATES = re.compile(r"_(\d{8})_(\d{8})_smips_")


def repair_cache(root, var: str = "totalbucket", workers: int = 8,
                 dry_run: bool = False) -> int:
    """Sweep every cached SMIPS cube under ``root`` and fill its day-gaps.

    :func:`download_smips` already repairs on every load, so this is only for
    auditing a cache without touching the pipeline (or healing cubes nothing is
    about to read). The period comes from the stub's ``_YYYYMMDD_YYYYMMDD_``
    tag; cubes whose stub carries no dates fall back to their own first/last
    day, which finds interior gaps but not truncated ends.

    Returns the number of days recovered.
    """
    root = Path(root)
    total = 0
    cubes = sorted(root.rglob(f"*_smips_{var}.nc"))
    print(f"{len(cubes)} cached SMIPS cube(s) under {root}")
    for path in cubes:
        with xr.open_dataset(path) as ds:
            name = [v for v in ds.data_vars if v != "spatial_ref"][0]
            cube = ds[name].load()
        m = _STUB_DATES.search(path.name)
        if m:
            start, end = (pd.Timestamp(s).date() for s in m.groups())
        else:
            t = pd.DatetimeIndex(cube["time"].values)
            start, end = t.min().date(), t.max().date()
        gaps = missing_days(cube, start, end)
        if not gaps:
            print(f"  {path.name}: complete ({cube.sizes['time']} days)")
            continue
        if dry_run:
            print(f"  {path.name}: {len(gaps)} missing -> "
                  f"{[d.isoformat() for d in gaps[:5]]}")
            total += len(gaps)
            continue
        print(f"  {path.name}:")
        bbox = (float(cube.x.min()), float(cube.y.min()),
                float(cube.x.max()), float(cube.y.max()))
        cube, n = repair_cube(cube, bbox, start, end, var=var, workers=workers)
        if n:
            tmp = path.with_suffix(".nc.tmp")
            cube.to_dataset(name=cube.name).to_netcdf(tmp)
            tmp.replace(path)
        total += n
    print(f"\n{'would recover' if dry_run else 'recovered'} {total} day(s)")
    return total


def test():
    from PaddockTS.query import Query
    q = Query.from_lat_lon(-35.41928, 147.60408, 2.0,
                           date(2020, 6, 1), date(2020, 6, 7), stub="SMIPS_TEST")
    cube = download_smips(q)
    print(cube)
    print("mean per day (mm):\n", cube.mean(("x", "y")).to_pandas().round(2))


if __name__ == "__main__":
    import argparse
    from PaddockTS.config import config

    ap = argparse.ArgumentParser(description="SMIPS loader: smoke test, or cache repair.")
    ap.add_argument("--repair", nargs="?", const=None, default=False, metavar="CACHE_DIR",
                    help="sweep cached cubes for day-gaps and refill them "
                         "(default: the PaddockTS tmp dir)")
    ap.add_argument("--dry-run", action="store_true", help="--repair: report only")
    ap.add_argument("--var", default="totalbucket")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    if a.repair is not False:
        repair_cache(a.repair or config.tmp_dir, var=a.var, workers=a.workers,
                     dry_run=a.dry_run)
    else:
        test()
