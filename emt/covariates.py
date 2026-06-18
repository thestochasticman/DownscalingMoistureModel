"""Fine-resolution terrain covariates and point sampling for the model.

Composes PaddockTS primitives -- DEM download
(``...TerrainTiles.download_terrain``), flow accumulation, slope, aspect, TWI
and HLI (``...TerrainTiles.utils``) -- but first **reprojects the DEM to a
metric (UTM) CRS**.

Why the reprojection: PaddockTS's terrain utils compute ``np.gradient`` using
the raster's pixel spacing as-is. On the native EPSG:4326 Copernicus DEM that
spacing is in *degrees* (~0.00028) while the elevations are in *metres*, so the
gradient is ~1000x too large and slope saturates at ~90 degrees everywhere
(TWI inherits the error). Feeding the utils a UTM DEM (metre spacing, metre
elevations) makes their math dimensionally correct -- so we reuse them unchanged.
"""
from __future__ import annotations

import numpy as np
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
from pyproj import Transformer

from PaddockTS.query import Query
from PaddockTS.Environmental.TerrainTiles.download_terrain_tiles import download_terrain
from PaddockTS.Environmental.TerrainTiles import utils as terrain_utils

# 30 m terrain predictors the downscaling model uses. Aspect is decomposed into
# northness/eastness (continuous, no 0/360 wraparound) for ML.
TERRAIN_VARS = ("elevation", "slope", "northness", "eastness", "twi", "hli", "accumulation")


def _utm_dem_path(query: Query) -> str:
    return query.terrain_path.replace(".tif", "_utm.tif")


def _ensure_utm_dem(query: Query) -> str:
    """Reproject the cached 4326 DEM to its local UTM CRS (metres); return the path."""
    utm_path = _utm_dem_path(query)
    import os
    if os.path.exists(utm_path):
        return utm_path
    dem = rioxarray.open_rasterio(query.terrain_path, masked=True)
    # Use an explicit numeric nodata (not NaN) so pysheds can represent it.
    dem_utm = dem.rio.reproject(dem.rio.estimate_utm_crs(), nodata=-9999.0).astype("float32")
    dem_utm = dem_utm.rio.write_nodata(-9999.0)
    dem_utm.rio.to_raster(utm_path)
    return utm_path


def terrain_covariates(query: Query) -> xr.Dataset:
    """Return the 30 m terrain covariate stack for ``query`` (on a metric UTM grid).

    Downloads/caches the DEM via PaddockTS, reprojects it to UTM, then derives
    slope, aspect (-> northness/eastness), TWI, HLI and flow accumulation. The
    result carries its UTM CRS; sample it with :func:`sample_points` (which takes
    lon/lat and handles the reprojection).
    """
    download_terrain(query)                  # ensures query.terrain_path (cached, 4326)
    tif = _ensure_utm_dem(query)             # metric DEM for correct gradients

    dem_da = rioxarray.open_rasterio(tif, masked=True).squeeze("band", drop=True)
    dims = dem_da.dims                        # ("y", "x") in UTM metres

    slope = terrain_utils.calculate_slope(tif)
    aspect = terrain_utils.calculate_aspect(tif)
    _grid, _dem, _fdir, acc = terrain_utils.pysheds_accumulation(tif)
    twi = terrain_utils.calculate_twi(acc, slope)
    hli = terrain_utils.calculate_hli(slope, aspect, query.centre_lat)

    # PaddockTS utils read the raster unmasked, so valid pixels adjacent to the
    # UTM nodata corners get a spurious cliff from np.gradient. Restrict every
    # derivative to the DEM's valid footprint, eroded by one pixel to drop that
    # boundary fringe.
    from scipy.ndimage import binary_erosion
    valid = binary_erosion(np.isfinite(dem_da.values), iterations=1)
    m = lambda a: np.where(valid, np.asarray(a, dtype="float64"), np.nan)

    aspect_rad = np.radians(aspect)
    ds = xr.Dataset(
        {
            "elevation": (dims, np.where(valid, dem_da.values.astype("float64"), np.nan)),
            "slope": (dims, m(slope)),
            "northness": (dims, m(np.cos(aspect_rad))),
            "eastness": (dims, m(np.sin(aspect_rad))),
            "twi": (dims, m(np.where(np.isfinite(twi), twi, np.nan))),
            "hli": (dims, m(hli)),
            "accumulation": (dims, m(acc)),
        },
        coords=dem_da.coords,
    )
    return ds.rio.write_crs(dem_da.rio.crs)


def sample_points(obj: xr.DataArray | xr.Dataset, lon: float, lat: float):
    """Nearest-pixel sample of a raster at one lon/lat point.

    Reprojects the point into the raster's CRS if it isn't EPSG:4326, then
    selects with ``.sel(x=, y=, method='nearest')``. Works for SMIPS cubes
    (lon/lat) and the UTM terrain stack alike.
    """
    crs = obj.rio.crs if hasattr(obj, "rio") else None
    x, y = lon, lat
    if crs is not None and crs.to_epsg() != 4326:
        x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    return obj.sel(x=x, y=y, method="nearest")
