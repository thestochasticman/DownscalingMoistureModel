"""Load terrain data from PaddockTS TerrainTiles module.

This module wraps PaddockTS terrain data access and provides derived
terrain indices needed for the water balance model:

- DEM (elevation)
- Slope
- Aspect
- TWI (Topographic Wetness Index)
- HLI (Heat Load Index)
- Flow direction (D8)
- Flow accumulation
"""

from __future__ import annotations

import sys
from os.path import exists
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from WaterBalanceModel.query import WaterBalanceQuery

# Add PaddockTS to path if needed
sys.path.insert(0, '/borevitz_projects/repos/paddock-ts-local')


def load_terrain(query: WaterBalanceQuery) -> xr.Dataset:
    """Load terrain data and derived indices for water balance model.

    Downloads Copernicus 30m DEM via PaddockTS, computes slope, aspect,
    TWI, HLI, flow direction, and flow accumulation.

    Cached: If the output NetCDF already exists at ``query.terrain_path``,
    it is loaded and returned.

    Args:
        query: The water balance query specifying bbox.

    Returns:
        xr.Dataset with variables:
            - dem: elevation (m)
            - slope: slope angle (radians)
            - aspect: aspect angle (radians, 0=N, pi/2=E)
            - twi: Topographic Wetness Index
            - hli: Heat Load Index (0-1)
            - flow_dir: D8 flow direction
            - flow_acc: flow accumulation (upstream cell count)
    """
    if exists(query.terrain_path):
        print(f'  cached: {query.terrain_path}')
        return xr.open_dataset(query.terrain_path)

    # Import PaddockTS modules
    from PaddockTS.Environmental.TerrainTiles.download_terrain_tiles import (
        download_terrain,
        get_filename,
    )
    from PaddockTS.Environmental.TerrainTiles.utils import (
        pysheds_accumulation,
        calculate_slope,
        calculate_aspect,
        calculate_twi,
        calculate_hli,
    )
    from PaddockTS.query import Query as PaddockQuery
    import rasterio

    # Create PaddockTS query
    pquery = PaddockQuery(
        bbox=query.bbox,
        start=query.start,
        end=query.end,
        stub=query.stub,
    )

    # Download DEM
    print('  downloading terrain DEM...', flush=True)
    download_terrain(pquery)
    terrain_file = get_filename(pquery)

    # Read DEM
    with rasterio.open(terrain_file) as src:
        dem = src.read(1)
        transform = src.transform
        crs = src.crs

    # Compute derived indices
    print('  computing flow accumulation...', flush=True)
    grid, dem_filled, fdir, acc = pysheds_accumulation(terrain_file)

    print('  computing slope and aspect...', flush=True)
    slope = calculate_slope(terrain_file)  # Takes TIF path, returns degrees
    aspect = calculate_aspect(terrain_file)  # Takes TIF path, returns degrees

    print('  computing TWI and HLI...', flush=True)
    twi = calculate_twi(acc, slope)
    hli = calculate_hli(slope, aspect, query.centre_lat)

    # Build coordinate arrays
    ny, nx = dem.shape
    x = np.arange(nx) * transform.a + transform.c + transform.a / 2
    y = np.arange(ny) * transform.e + transform.f + transform.e / 2

    # Create dataset
    ds = xr.Dataset(
        data_vars={
            'dem': (['y', 'x'], dem, {'units': 'm', 'long_name': 'Elevation'}),
            'slope': (['y', 'x'], slope, {'units': 'degrees', 'long_name': 'Slope angle'}),
            'aspect': (['y', 'x'], aspect, {'units': 'degrees', 'long_name': 'Aspect (0=N)'}),
            'twi': (['y', 'x'], twi, {'units': '', 'long_name': 'Topographic Wetness Index'}),
            'hli': (['y', 'x'], hli, {'units': '', 'long_name': 'Heat Load Index'}),
            'flow_dir': (['y', 'x'], fdir.astype(np.int16), {'long_name': 'D8 flow direction'}),
            'flow_acc': (['y', 'x'], acc, {'units': 'cells', 'long_name': 'Flow accumulation'}),
        },
        coords={
            'x': x,
            'y': y,
        },
        attrs={
            'crs': str(crs),
            'resolution': abs(transform.a),
            'source': 'Copernicus DEM 30m',
        },
    )

    # Save to cache
    ds.to_netcdf(query.terrain_path)
    print(f'  saved: {query.terrain_path}')

    return ds
