"""Drainage computation for the water balance model.

This module computes gravity drainage below the root zone when
soil moisture exceeds field capacity.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def compute_gravity_drainage(
    theta: xr.DataArray,
    theta_fc: xr.DataArray,
    theta_sat: xr.DataArray,
    k_sat: xr.DataArray,
    beta: float = 3.0,
) -> xr.DataArray:
    """Compute gravity drainage below root zone.

    Drainage occurs when soil moisture exceeds field capacity.
    Uses a power-law relationship:

    D = K_sat * ((theta - theta_fc) / (theta_sat - theta_fc))^beta

    Where beta controls the nonlinearity (typically 2-4).

    Args:
        theta: Current soil moisture (mm or m3/m3, same as theta_fc).
        theta_fc: Field capacity.
        theta_sat: Saturated moisture content.
        k_sat: Saturated hydraulic conductivity (mm/day).
        beta: Power-law exponent (default 3.0).

    Returns:
        Drainage flux (mm/day).
    """
    # Available for drainage
    drainable = theta - theta_fc
    drainable = xr.where(drainable > 0, drainable, 0)

    # Relative saturation of drainable pore space
    pore_space = theta_sat - theta_fc
    rel_sat = drainable / (pore_space + 1e-10)
    rel_sat = rel_sat.clip(min=0, max=1)

    # Drainage rate
    drainage = k_sat * (rel_sat ** beta)

    # Cannot drain more than available
    drainage = xr.where(drainage > drainable, drainable, drainage)

    drainage.attrs.update({
        'long_name': 'Gravity drainage',
        'units': 'mm/day',
    })

    return drainage


def compute_percolation(
    theta: xr.DataArray,
    theta_fc: xr.DataArray,
    k_unsat: xr.DataArray,
) -> xr.DataArray:
    """Compute percolation using unsaturated hydraulic conductivity.

    Simplified unsaturated flow where percolation rate depends on
    the unsaturated conductivity at current moisture content.

    Args:
        theta: Current soil moisture.
        theta_fc: Field capacity.
        k_unsat: Unsaturated hydraulic conductivity at current theta.

    Returns:
        Percolation rate (mm/day).
    """
    # Percolation only when above field capacity
    percolation = xr.where(theta > theta_fc, k_unsat, 0)

    percolation.attrs.update({
        'long_name': 'Percolation',
        'units': 'mm/day',
    })

    return percolation


def compute_capillary_rise(
    theta: xr.DataArray,
    theta_wp: xr.DataArray,
    water_table_depth: float,
    max_rise: float = 2.0,
) -> xr.DataArray:
    """Compute capillary rise from shallow water table.

    Simplified model where capillary rise increases as soil dries
    and water table is shallow.

    Args:
        theta: Current soil moisture.
        theta_wp: Wilting point.
        water_table_depth: Depth to water table (m).
        max_rise: Maximum capillary rise rate (mm/day).

    Returns:
        Capillary rise (mm/day, positive upward).
    """
    # Capillary rise increases as soil dries
    dryness = 1 - theta / (theta_wp + 0.1)
    dryness = dryness.clip(min=0, max=1)

    # Decreases with water table depth (exponential decay)
    depth_factor = np.exp(-water_table_depth / 1.0)

    rise = max_rise * dryness * depth_factor

    rise.attrs.update({
        'long_name': 'Capillary rise',
        'units': 'mm/day',
    })

    return rise
