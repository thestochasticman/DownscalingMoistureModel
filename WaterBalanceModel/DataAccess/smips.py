"""Load SMIPS soil moisture data via PaddockTS.

This module wraps PaddockTS SMIPS data access to provide coarse-resolution
soil moisture for calibration and constraint in the water balance model.

SMIPS (Soil Moisture Index from Passive Microwave Satellites) provides
daily volumetric soil moisture at ~1km resolution across Australia.
"""

from __future__ import annotations

import sys
from os.path import exists
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from WaterBalanceModel.query import WaterBalanceQuery

# Add PaddockTS to path if needed
sys.path.insert(0, '/borevitz_projects/repos/paddock-ts-local')


def load_smips(query: WaterBalanceQuery) -> xr.DataArray:
    """Load SMIPS soil moisture cube for calibration/constraint.

    Downloads SMIPS data via PaddockTS for the AOI and date range.
    Used as the coarse-resolution constraint in the SMIPS mass
    conservation optimization.

    Cached: If the output NetCDF already exists at ``query.smips_path``,
    it is loaded and returned.

    Args:
        query: The water balance query specifying bbox and date range.

    Returns:
        xr.DataArray with dims (time, y, x) containing soil moisture (mm).
        Resolution is ~1km (0.01 degrees).
    """
    if exists(query.smips_path):
        print(f'  cached: {query.smips_path}')
        ds = xr.open_dataset(query.smips_path)
        return ds['soil_moisture'] if 'soil_moisture' in ds else ds['TotalBucketRaw']

    # Import PaddockTS modules
    from PaddockTS.Environmental.SMIPS.download_smips import smips_cube
    from PaddockTS.query import Query as PaddockQuery

    # Create PaddockTS query
    pquery = PaddockQuery(
        bbox=query.bbox,
        start=query.start,
        end=query.end,
        stub=query.stub,
    )

    # Download SMIPS cube
    print('  downloading SMIPS soil moisture...', flush=True)
    smips = smips_cube(
        bbox=query.bbox,
        start=query.start,
        end=query.end,
        skip_missing=True,
    )

    # Rename and add attributes
    smips = smips.rename('soil_moisture')
    smips.attrs.update({
        'long_name': 'SMIPS Soil Moisture',
        'units': 'mm',
        'source': 'TERN SMIPS (DOI: 10.25901/b020-nm39)',
        'resolution': '~1km (0.01 degrees)',
    })

    # Save to cache
    smips_ds = smips.to_dataset()
    smips_ds.to_netcdf(query.smips_path)
    print(f'  saved: {query.smips_path} ({len(smips.time)} timesteps)')

    return smips
