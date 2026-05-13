"""Precipitation processing for the water balance model.

This module handles spatial disaggregation of coarse precipitation data
to the high-resolution model grid and computes effective precipitation
after canopy interception.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def disaggregate_precipitation(
    precip_coarse: float,
    terrain: xr.Dataset,
    orographic_factor: float = 0.1,
) -> xr.DataArray:
    """Spatially disaggregate coarse precipitation to fine grid.

    Applies elevation-based orographic adjustment to distribute
    point-scale precipitation across the high-resolution terrain grid.

    P_local = P_coarse * (1 + alpha * (elev - elev_mean) / 1000)

    Where alpha is the orographic enhancement factor (typically 0.05-0.2).

    Args:
        precip_coarse: Coarse precipitation value (mm/day) - scalar.
        terrain: Terrain dataset with 'dem' variable.
        orographic_factor: Orographic enhancement factor (default 0.1).
            Higher values give more elevation dependence.

    Returns:
        Disaggregated precipitation at terrain resolution (mm/day).
    """
    dem = terrain['dem']
    dem_mean = float(dem.mean())

    # Orographic adjustment
    elev_diff = (dem - dem_mean) / 1000  # km
    adjustment = 1 + orographic_factor * elev_diff

    # Apply adjustment
    precip = precip_coarse * adjustment

    # Ensure non-negative
    precip = precip.clip(min=0)

    # Mass conservation: scale so spatial mean equals coarse input
    if float(precip.mean()) > 0:
        precip = precip * (precip_coarse / float(precip.mean()))

    precip.attrs.update({
        'long_name': 'Precipitation',
        'units': 'mm/day',
    })

    return precip


def compute_effective_precipitation(
    precip: xr.DataArray,
    ndvi: xr.DataArray,
    interception_capacity: float = 2.0,
    ndvi_full_canopy: float = 0.8,
) -> xr.DataArray:
    """Compute effective precipitation after canopy interception.

    P_eff = P - min(P, I_max * canopy_fraction)

    Where canopy_fraction is derived from NDVI.

    Args:
        precip: Total precipitation (mm/day).
        ndvi: NDVI values for canopy estimation.
        interception_capacity: Maximum interception per unit canopy (mm).
            Default 2.0 mm.
        ndvi_full_canopy: NDVI threshold for full canopy cover.
            Default 0.8.

    Returns:
        Effective precipitation reaching soil surface (mm/day).
    """
    # Estimate canopy fraction from NDVI
    canopy_fraction = (ndvi / ndvi_full_canopy).clip(min=0, max=1)

    # Maximum interception
    interception_max = interception_capacity * canopy_fraction

    # Actual interception (cannot exceed precipitation)
    interception = xr.where(precip < interception_max, precip, interception_max)

    # Effective precipitation
    precip_eff = precip - interception

    precip_eff.attrs.update({
        'long_name': 'Effective precipitation',
        'units': 'mm/day',
    })

    return precip_eff


def compute_throughfall_fraction(ndvi: xr.DataArray, ndvi_full: float = 0.8) -> xr.DataArray:
    """Compute throughfall fraction based on canopy cover.

    Throughfall fraction decreases with increasing canopy cover.

    Args:
        ndvi: NDVI values.
        ndvi_full: NDVI for full canopy (default 0.8).

    Returns:
        Throughfall fraction (0-1).
    """
    canopy_cover = (ndvi / ndvi_full).clip(min=0, max=1)

    # Throughfall fraction (simplified model)
    # At full canopy, ~20% reaches ground directly
    # At bare soil, 100% reaches ground
    throughfall = 1 - 0.8 * canopy_cover

    return throughfall
