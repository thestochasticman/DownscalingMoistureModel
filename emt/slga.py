"""EMT-local SLGA soil-covariate loader (root-zone soil properties, ~90 m).

Adds static soil covariates to the downscaling model. The per-station and
per-pixel absolute-moisture *baseline* that SMIPS + terrain cannot resolve is set
largely by soil texture and water-holding capacity; these come from the Soil and
Landscape Grid of Australia (SLGA v2, TERN).

The COG URL resolution and TERN auth are delegated to PaddockTS
(``SLGASoils.utils.get_cog_url`` / ``_setup_tern_auth`` — ``get_cog_url`` resolves
each attribute's release date from the datastore listing, so AWC/BDW work as well
as clay/sand). This module exists on top of ``download_slga_soils`` only to
depth-average the standard SLGA slices into a single root-zone (0-100 cm) value
per attribute (the model's feature), cached as one NetCDF per AOI.

Requires a TERN API key (``tern_api_key`` in ``~/.config/PaddockTS.json``).
"""
from __future__ import annotations

from os import makedirs
from os.path import exists

import numpy as np
from functools import lru_cache

import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from PaddockTS.query import Query
from PaddockTS.Environmental.SLGASoils.utils import (
    load_tern_api_key, _setup_tern_auth, get_cog_url)

# Model feature name -> PaddockTS SLGA attribute name (see SLGASoils.attribute_codes).
SOIL_VARS = ("soil_clay", "soil_sand", "soil_awc", "soil_bdw")
SLGA_ATTR = {"soil_clay": "Clay", "soil_sand": "Sand",
             "soil_awc": "Available_Water_Capacity", "soil_bdw": "Bulk_Density",
             "soil_dul": "Drained_Upper_Limit", "soil_l15": "L15"}

# SLGA's own measured-basis soil hydraulic limits: the drained upper limit
# (field capacity) and the 15-bar lower limit (wilting point). These are the
# quantities emt.pedotransfer *estimates* from texture -- having them directly
# removes both the Saxton-Rawls regression and its organic-matter assumption
# (see emt.model9). They are published only in SLGA **Release 1** (v1); the
# Release 2 (v2) tree that carries clay/sand/AWC/bulk-density has no DUL or L15
# directory contents, so they need their own URL resolver below.
HYDRAULIC_VARS = ("soil_dul", "soil_l15")
_V1_ONLY = set(HYDRAULIC_VARS)

# Standard SLGA depth slices spanning the 0-100 cm root zone, with thickness (cm)
# used as the depth-averaging weight. (100-200 cm is below the root zone.)
DEPTHS = [("0-5cm", 5), ("5-15cm", 10), ("15-30cm", 15),
          ("30-60cm", 30), ("60-100cm", 40)]

get_filename = lambda q: f"{q.tmp_dir}/Environmental/{q.stub}_slga.nc"


@lru_cache(maxsize=None)
def _v1_listing(code: str) -> str:
    """Release-1 directory listing for an attribute code (cached, retried).

    Cached because a station build asks for five depths of each attribute and
    the listing is identical for all of them; retried because a single
    transient timeout would otherwise abort a whole multi-station build.
    """
    import time
    import requests as _rq
    url = ("https://data.tern.org.au/model-derived/slga/NationalMaps/"
           f"SoilAndLandscapeGrid/{code}/v1/")
    last = None
    for attempt in range(4):
        try:
            r = _rq.get(url, timeout=60)
            r.raise_for_status()
            return r.text
        except Exception as e:                                    # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"SLGA v1 listing for {code} unreachable: {last}")


def _v1_cog_url(attribute: str, depth: str, api_key: str) -> str:
    """Resolve a Release-1 (v1) SLGA COG URL.

    PaddockTS's ``get_cog_url`` targets v2 only. DUL and L15 exist solely in
    v1, so resolve those from the v1 directory listing with the same
    "newest matching release date wins" rule.
    """
    import re
    from PaddockTS.Environmental.SLGASoils.slgasoils import SLGASoils as _S

    code = _S.attribute_codes[attribute]
    ds, de = _S.depth_codes[depth]
    hits = sorted(set(re.findall(rf'({code}_{ds}_{de}_EV_[^"<>]*?\.tif)',
                                 _v1_listing(code))))
    if not hits:
        raise RuntimeError(f"No SLGA v1 EV COG for {attribute} {depth}")
    return ("https://data.tern.org.au/model-derived/slga/NationalMaps/"
            f"SoilAndLandscapeGrid/{code}/v1/{hits[-1]}")


def cog_url(var: str, depth: str, api_key: str) -> str:
    """COG URL for an EMT soil variable, from whichever SLGA release has it."""
    attr = SLGA_ATTR[var]
    if var in _V1_ONLY:
        return _v1_cog_url(attr, depth, api_key)
    return get_cog_url(attr, depth, api_key)


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
        attr = SLGA_ATTR[var]
        stack, weights = [], []
        for depth, thick in DEPTHS:
            da = _read_window(get_cog_url(attr, depth, api_key), query.bbox)
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


def smooth_soil(ds: xr.Dataset, sigma_px: float = 2.0) -> xr.Dataset:
    """NaN-aware Gaussian blur of the soil covariates (used by model5).

    SLGA is a categorical-ish product: adjacent map units meet at hard edges, so
    a downscaled field that uses it verbatim inherits blocky boundaries (see the
    model4 demonstration). A modest spatial blur softens those seams while
    preserving the raster's valid footprint. ``sigma_px`` is in pixels of the
    dataset's own grid (SLGA native ~90 m). A no-op if ``sigma_px <= 0``.
    """
    if sigma_px <= 0:
        return ds
    from scipy.ndimage import gaussian_filter
    out = ds.copy()
    for v in SOIL_VARS:
        a = ds[v].values.astype("float64")
        mask = np.isfinite(a)
        num = gaussian_filter(np.where(mask, a, 0.0), sigma_px)
        den = gaussian_filter(mask.astype("float64"), sigma_px)
        with np.errstate(invalid="ignore", divide="ignore"):
            res = num / den
        res[~mask] = np.nan          # keep the original valid footprint
        out[v] = (ds[v].dims, res)
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
