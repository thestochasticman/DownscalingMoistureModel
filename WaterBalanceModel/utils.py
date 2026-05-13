"""Utility functions for the water balance model."""

from __future__ import annotations

from datetime import date

from WaterBalanceModel.query import WaterBalanceQuery


def get_example_query() -> WaterBalanceQuery:
    """Create an example query for testing.

    Returns a query for a small area in NSW, Australia for Q1 2020.
    """
    return WaterBalanceQuery.from_lat_lon(
        lat=-33.51,
        lon=148.37,
        buffer_km=1.0,
        start=date(2020, 1, 1),
        end=date(2020, 3, 31),
        stub='example_query',
    )


def validate_bbox(bbox: list[float]) -> bool:
    """Validate a bounding box.

    Args:
        bbox: [west, south, east, north] in EPSG:4326.

    Returns:
        True if valid, raises ValueError otherwise.
    """
    if len(bbox) != 4:
        raise ValueError('bbox must have 4 elements: [west, south, east, north]')

    west, south, east, north = bbox

    if west >= east:
        raise ValueError(f'west ({west}) must be less than east ({east})')
    if south >= north:
        raise ValueError(f'south ({south}) must be less than north ({north})')
    if not -180 <= west <= 180:
        raise ValueError(f'west ({west}) must be in [-180, 180]')
    if not -180 <= east <= 180:
        raise ValueError(f'east ({east}) must be in [-180, 180]')
    if not -90 <= south <= 90:
        raise ValueError(f'south ({south}) must be in [-90, 90]')
    if not -90 <= north <= 90:
        raise ValueError(f'north ({north}) must be in [-90, 90]')

    return True


def compute_water_balance_closure(result) -> dict:
    """Compute water balance closure for validation.

    Checks that P = ET + R + D + delta_S at each timestep.

    Args:
        result: Output dataset from water balance model.

    Returns:
        Dict with closure statistics.
    """
    import numpy as np

    # Get variables
    sm = result['soil_moisture'].values
    et = result['et_actual'].values
    runoff = result['runoff'].values
    drainage = result['drainage'].values
    lateral_in = result['lateral_in'].values
    lateral_out = result['lateral_out'].values

    # Compute delta storage
    delta_s = np.diff(sm, axis=0)

    # Compute implied precipitation (from closure)
    # P = delta_S + ET + R + D + L_out - L_in
    implied_p = (
        delta_s
        + et[1:]
        + runoff[1:]
        + drainage[1:]
        + lateral_out[1:]
        - lateral_in[1:]
    )

    # Closure error statistics
    return {
        'mean_closure_error': float(np.nanmean(implied_p)),
        'std_closure_error': float(np.nanstd(implied_p)),
        'max_closure_error': float(np.nanmax(np.abs(implied_p))),
    }
