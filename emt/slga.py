"""EMT-local SLGA soil-covariate loader (root-zone soil properties, ~90 m).

Adds static soil covariates to the downscaling model. The per-station and
per-pixel absolute-moisture *baseline* that SMIPS + terrain cannot resolve is set
largely by soil texture and water-holding capacity; these come from the Soil and
Landscape Grid of Australia (SLGA v2, TERN).

Why this exists instead of ``PaddockTS.Environmental.SLGASoils.download_slga_soils``:
that loader hardcodes a single release date in its COG URL
(``..._20210902.tif``), which is correct for clay/sand/silt but 404s for AWC
(``20210614``) and bulk density (``20230607``) — each SLGA attribute is released
on its own date. This module resolves the actual filename per attribute from the
TERN datastore directory listing (robust to date changes), reuses PaddockTS's
TERN API-key auth, and aggregates the standard depth slices into a single
root-zone (0-100 cm) value per attribute.

Requires a TERN API key (``tern_api_key`` in ``~/.config/PaddockTS.json``).
"""
from __future__ import annotations

import re
from os import makedirs
from os.path import exists

import numpy as np
import requests
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from PaddockTS.query import Query
from PaddockTS.Environmental.SLGASoils.utils import load_tern_api_key, _setup_tern_auth

DATASTORE = ("https://data.tern.org.au/model-derived/slga/NationalMaps/"
             "SoilAndLandscapeGrid")

# Model feature name -> SLGA attribute code.
SOIL_VARS = ("soil_clay", "soil_sand", "soil_awc", "soil_bdw")
ATTR_CODE = {"soil_clay": "CLY", "soil_sand": "SND",
             "soil_awc": "AWC", "soil_bdw": "BDW"}

# Standard SLGA depth slices spanning the 0-100 cm root zone, with thickness (cm)
# used as the depth-averaging weight. (100-200 cm is below the root zone.)
DEPTHS = [("000", "005", 5), ("005", "015", 10), ("015", "030", 15),
          ("030", "060", 30), ("060", "100", 40)]

get_filename = lambda q: f"{q.tmp_dir}/Environmental/{q.stub}_slga.nc"

_DIR_CACHE: dict[str, list[str]] = {}


def _list_dir(code: str, api_key: str) -> list[str]:
    """Filenames in the SLGA v2 directory for ``code`` (cached per process)."""
    if code not in _DIR_CACHE:
        r = requests.get(f"{DATASTORE}/{code}/v2/",
                         headers={"x-api-key": api_key}, timeout=60)
        r.raise_for_status()
        _DIR_CACHE[code] = re.findall(
            rf"{code}_\d{{3}}_\d{{3}}_EV_[A-Za-z_]+_\d{{8}}\.tif", r.text)
    return _DIR_CACHE[code]


def _cog_url(code: str, ds: str, de: str, api_key: str) -> str:
    """Resolve the expected-value COG URL for one attribute/depth (date varies)."""
    matches = [f for f in _list_dir(code, api_key) if f.startswith(f"{code}_{ds}_{de}_EV_")]
    if not matches:
        raise RuntimeError(f"No SLGA EV COG for {code} {ds}-{de} cm in datastore listing")
    return f"{DATASTORE}/{code}/v2/{sorted(matches)[-1]}"


def _read_window(url: str, bbox) -> xr.DataArray:
    """Read the COG clipped to ``bbox`` as a 2D DataArray (native ~90 m grid)."""
    da = rioxarray.open_rasterio(f"/vsicurl/{url}", masked=True).squeeze("band", drop=True)
    return da.rio.clip_box(*bbox).load()


def soil_covariates(query: Query, reload: bool = False) -> xr.Dataset:
    """Root-zone (0-100 cm depth-averaged) SLGA soil covariates for ``query.bbox``.

    Returns an :class:`xr.Dataset` (``soil_clay``, ``soil_sand``, ``soil_awc``,
    ``soil_bdw``) on the native SLGA grid (EPSG:4326), cached at
    ``{query.tmp_dir}/Environmental/{query.stub}_slga.nc``. Sample it with
    :func:`emt.covariates.sample_points`.
    """
    filename = get_filename(query)
    if not reload and exists(filename):
        print(f"  cached: {filename}")
        with xr.open_dataset(filename) as ds:
            return ds.load()

    api_key = load_tern_api_key()
    _setup_tern_auth(api_key)
    makedirs(f"{query.tmp_dir}/Environmental", exist_ok=True)
    print(f"  fetching SLGA soil ({', '.join(SOIL_VARS)}) for bbox {query.bbox}...", flush=True)

    layers = {}
    ref = None
    for var in SOIL_VARS:
        code = ATTR_CODE[var]
        stack, weights = [], []
        for ds_, de_, thick in DEPTHS:
            da = _read_window(_cog_url(code, ds_, de_, api_key), query.bbox)
            if ref is None:
                ref = da
            else:
                da = da.rio.reproject_match(ref)   # align all to the first grid
            stack.append(da)
            weights.append(thick)
        arr = xr.concat(stack, dim="depth")
        w = xr.DataArray(np.array(weights, dtype="float64"), dims="depth")
        layers[var] = arr.weighted(w).mean("depth")   # depth-weighted root-zone mean

    out = xr.Dataset(layers).rio.write_crs(ref.rio.crs)
    out.to_netcdf(filename)
    print(f"  saved: {filename} ({ref.sizes.get('y')}x{ref.sizes.get('x')} px)")
    return out


def test():
    from datetime import date
    q = Query.from_lat_lon(-35.38978, 147.45720, 2.0, date(2020, 1, 1), date(2020, 1, 2),
                           stub="SLGA_TEST_K6")
    ds = soil_covariates(q)
    print(ds)
    for v in SOIL_VARS:
        print(f"  {v}: mean {float(ds[v].mean()):.2f}")


if __name__ == "__main__":
    test()
