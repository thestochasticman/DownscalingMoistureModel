"""Load climate data from SILO and OzWALD via PaddockTS.

This module wraps PaddockTS climate data access to provide a unified
interface for the water balance model. Climate forcing includes:

- Precipitation (Pg from OzWALD)
- Temperature (Tmin, Tmax from OzWALD or SILO)
- Radiation (from SILO)
- Vapor pressure (VPeff from OzWALD or vp from SILO)
- Wind speed (Uavg from OzWALD)
- Reference ET (et_short_crop from SILO)
"""

from __future__ import annotations

import sys
from os.path import exists
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from WaterBalanceModel.query import WaterBalanceQuery

# Add PaddockTS to path if needed
sys.path.insert(0, '/borevitz_projects/repos/paddock-ts-local')


def load_climate(query: WaterBalanceQuery) -> pd.DataFrame:
    """Load merged climate data for the water balance model.

    Downloads SILO and OzWALD climate data for the AOI centre point,
    merges them into a single DataFrame with standardized column names.
    Uses OzWALD ``Pg`` for precipitation as specified in design.

    Cached: If the output CSV already exists at ``query.climate_path``,
    it is loaded and returned without contacting the data servers.

    Args:
        query: The water balance query specifying bbox and date range.

    Returns:
        DataFrame with columns:
            - date: datetime index
            - precip: precipitation (mm/day) from OzWALD Pg
            - tmin: minimum temperature (C)
            - tmax: maximum temperature (C)
            - radiation: solar radiation (MJ/m2/day)
            - vp: vapor pressure (kPa)
            - wind: wind speed at 2m (m/s)
            - et0: reference ET (mm/day) from SILO et_short_crop
    """
    if exists(query.climate_path):
        print(f'  cached: {query.climate_path}')
        return pd.read_csv(query.climate_path, parse_dates=['date'])

    # Import PaddockTS modules
    from PaddockTS.Environmental.SILO.download_silo import download_silo
    from PaddockTS.Environmental.OzWALD.download_ozwald_daily import download_ozwald_daily

    # Create a PaddockTS-compatible query object
    from PaddockTS.query import Query as PaddockQuery

    pquery = PaddockQuery(
        bbox=query.bbox,
        start=query.start,
        end=query.end,
        stub=query.stub,
    )

    # Download data
    print('  fetching SILO climate data...', flush=True)
    silo = download_silo(pquery)
    silo['date'] = pd.to_datetime(silo['YYYY-MM-DD'])
    silo = silo.set_index('date')

    print('  fetching OzWALD daily climate data...', flush=True)
    ozwald = download_ozwald_daily(pquery)
    ozwald['date'] = pd.to_datetime(ozwald['time'])  # OzWALD uses 'time' column
    ozwald = ozwald.set_index('date')

    # Merge datasets
    df = silo.join(ozwald, how='inner', rsuffix='_oz')

    # Select and rename columns for water balance model
    result = pd.DataFrame({
        'date': df.index,
        'precip': df['Pg'],  # OzWALD precipitation
        'tmin': df['Tmin'] if 'Tmin' in df.columns else df['min_temp'],
        'tmax': df['Tmax'] if 'Tmax' in df.columns else df['max_temp'],
        'radiation': df['radiation'],  # SILO
        'vp': df['VPeff'] / 10 if 'VPeff' in df.columns else df['vp'],  # Convert hPa to kPa
        'wind': df['Uavg'] if 'Uavg' in df.columns else 2.0,  # Default 2 m/s if unavailable
        'et0': df['et_short_crop'],  # SILO reference ET
    }).reset_index(drop=True)

    # Filter to query date range
    result = result[
        (result['date'] >= pd.Timestamp(query.start)) &
        (result['date'] <= pd.Timestamp(query.end))
    ]

    # Save to cache
    result.to_csv(query.climate_path, index=False)
    print(f'  saved: {query.climate_path} ({len(result)} days)')

    return result
