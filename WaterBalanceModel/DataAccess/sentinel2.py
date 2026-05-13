"""Load NDVI time series from Sentinel-2 via PaddockTS.

This module wraps PaddockTS Sentinel-2 data access and spectral index
computation to provide NDVI time series for crop coefficient (Kc)
estimation in the water balance model.
"""

from __future__ import annotations

import sys
from os.path import exists
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from WaterBalanceModel.query import WaterBalanceQuery

# Add PaddockTS to path if needed
sys.path.insert(0, '/borevitz_projects/repos/paddock-ts-local')


def load_ndvi(query: WaterBalanceQuery, interpolate_daily: bool = True) -> xr.DataArray:
    """Load NDVI time series from Sentinel-2.

    Downloads Sentinel-2 ARD via PaddockTS, computes NDVI, and optionally
    interpolates to daily resolution for the water balance model.

    Cached: If the output Zarr already exists at ``query.ndvi_path``,
    it is loaded and returned.

    Args:
        query: The water balance query specifying bbox and date range.
        interpolate_daily: If True, interpolate NDVI to daily resolution
            using linear interpolation between cloud-free observations.
            Default True.

    Returns:
        xr.DataArray with dims (time, y, x) containing NDVI values.
        If interpolate_daily=True, time dimension is daily.
    """
    if exists(query.ndvi_path):
        print(f'  cached: {query.ndvi_path}')
        return xr.open_zarr(query.ndvi_path)['ndvi']

    # Import PaddockTS modules
    from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
    from PaddockTS.SpectralIndices.indices import compute_ndvi
    from PaddockTS.query import Query as PaddockQuery

    # Create PaddockTS query
    pquery = PaddockQuery(
        bbox=query.bbox,
        start=query.start,
        end=query.end,
        stub=query.stub,
    )

    # Download Sentinel-2 data
    print('  downloading Sentinel-2 data...', flush=True)
    ds = download_sentinel2(pquery)

    # Compute NDVI
    print('  computing NDVI...', flush=True)
    ndvi = compute_ndvi(ds)  # Returns (y, x, time) array
    ndvi = ndvi.transpose(2, 0, 1)  # -> (time, y, x)

    # Create DataArray with original projected CRS
    ndvi_da = xr.DataArray(
        ndvi,
        dims=['time', 'y', 'x'],
        coords={
            'time': ds.time.values,
            'y': ds.y.values,
            'x': ds.x.values,
        },
        attrs={
            'long_name': 'Normalized Difference Vegetation Index',
            'units': '',
            'valid_range': [-1, 1],
        },
    )

    # Reproject to EPSG:4326 to match terrain/soil coordinates
    print('  reprojecting NDVI to EPSG:4326...', flush=True)
    import rioxarray  # noqa: F401 - needed for .rio accessor

    # Set CRS from source dataset
    if hasattr(ds, 'rio') and ds.rio.crs is not None:
        src_crs = ds.rio.crs
    else:
        src_crs = 'EPSG:3577'  # Default Australian Albers

    ndvi_da = ndvi_da.rio.write_crs(src_crs)
    ndvi_da = ndvi_da.rio.reproject('EPSG:4326')

    if interpolate_daily:
        print('  interpolating NDVI to daily...', flush=True)
        # Create daily time index
        daily_times = pd.date_range(query.start, query.end, freq='D')

        # Interpolate each pixel to daily
        ndvi_da = ndvi_da.interp(time=daily_times, method='linear')

        # Fill NaN at edges with nearest valid value
        # Use interpolate_na which handles edge cases without bottleneck
        ndvi_da = ndvi_da.interpolate_na(dim='time', method='nearest', fill_value='extrapolate')

    # Save to Zarr
    ndvi_ds = xr.Dataset({'ndvi': ndvi_da})
    ndvi_ds.to_zarr(query.ndvi_path, mode='w')
    print(f'  saved: {query.ndvi_path}')

    return ndvi_da
