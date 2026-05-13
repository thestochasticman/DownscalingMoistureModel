"""Runoff computation for the water balance model.

This module implements the SCS Curve Number method for estimating
surface runoff from precipitation and soil moisture conditions.

References:
    USDA-NRCS (2004). National Engineering Handbook, Part 630:
    Hydrology, Chapter 10: Estimation of Direct Runoff from Storm Rainfall.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from attrs import frozen


@frozen
class RunoffParameters:
    """Parameters for SCS Curve Number runoff computation.

    Attributes:
        cn_dry: Curve number for dry conditions (AMC I).
        cn_normal: Curve number for normal conditions (AMC II).
        cn_wet: Curve number for wet conditions (AMC III).
        ia_ratio: Initial abstraction ratio Ia/S (default 0.2).
    """

    cn_dry: float = 65.0
    cn_normal: float = 75.0
    cn_wet: float = 85.0
    ia_ratio: float = 0.2


def compute_antecedent_moisture_class(
    theta: xr.DataArray,
    theta_fc: xr.DataArray,
    theta_wp: xr.DataArray,
) -> xr.DataArray:
    """Determine antecedent moisture condition class.

    AMC I: Dry conditions (theta < 0.5 * (theta_wp + theta_fc))
    AMC II: Normal conditions
    AMC III: Wet conditions (theta > theta_fc)

    Args:
        theta: Current soil moisture.
        theta_fc: Field capacity.
        theta_wp: Wilting point.

    Returns:
        AMC class (1, 2, or 3) at each point.
    """
    theta_mid = 0.5 * (theta_wp + theta_fc)

    amc = xr.where(theta < theta_mid, 1, 2)
    amc = xr.where(theta > theta_fc, 3, amc)

    return amc


def adjust_cn_for_amc(
    cn_normal: float,
    amc: xr.DataArray,
    params: RunoffParameters = RunoffParameters(),
) -> xr.DataArray:
    """Adjust curve number for antecedent moisture condition.

    Args:
        cn_normal: Curve number for normal conditions (AMC II).
        amc: Antecedent moisture class (1, 2, or 3).
        params: Runoff parameters with CN values.

    Returns:
        Adjusted curve number.
    """
    cn = xr.where(amc == 1, params.cn_dry, cn_normal)
    cn = xr.where(amc == 3, params.cn_wet, cn)

    return cn


def compute_runoff_scs(
    precip: xr.DataArray,
    theta: xr.DataArray,
    theta_fc: xr.DataArray,
    theta_wp: xr.DataArray,
    cn_normal: float = 75.0,
    params: RunoffParameters = RunoffParameters(),
) -> xr.DataArray:
    """Compute runoff using SCS Curve Number method.

    Q = (P - Ia)^2 / (P - Ia + S)  for P > Ia, else 0

    Where:
    - S = (25400/CN) - 254 (potential retention in mm)
    - Ia = ia_ratio * S (initial abstraction)
    - CN varies with antecedent moisture condition

    Args:
        precip: Precipitation (mm/day).
        theta: Current soil moisture.
        theta_fc: Field capacity.
        theta_wp: Wilting point.
        cn_normal: Curve number for normal conditions.
        params: Runoff parameters.

    Returns:
        Surface runoff (mm/day).
    """
    # Determine AMC
    amc = compute_antecedent_moisture_class(theta, theta_fc, theta_wp)

    # Adjust CN
    cn = adjust_cn_for_amc(cn_normal, amc, params)

    # Potential retention (mm)
    S = (25400 / cn) - 254

    # Initial abstraction
    Ia = params.ia_ratio * S

    # Runoff
    P_excess = precip - Ia
    P_excess = xr.where(P_excess > 0, P_excess, 0)

    runoff = xr.where(
        precip > Ia,
        (P_excess ** 2) / (P_excess + S),
        0,
    )

    runoff.attrs.update({
        'long_name': 'Surface runoff',
        'units': 'mm/day',
    })

    return runoff


def compute_saturation_excess(
    precip: xr.DataArray,
    theta: xr.DataArray,
    theta_sat: xr.DataArray,
    soil_depth: float,
) -> xr.DataArray:
    """Compute saturation excess runoff.

    When soil moisture exceeds saturation capacity, excess becomes runoff.

    Args:
        precip: Precipitation (mm/day).
        theta: Current soil moisture (mm).
        theta_sat: Saturated moisture content (m3/m3).
        soil_depth: Soil depth (mm).

    Returns:
        Saturation excess runoff (mm/day).
    """
    # Saturation capacity
    capacity = theta_sat * soil_depth

    # Potential moisture after adding precipitation
    theta_potential = theta + precip

    # Saturation excess
    excess = xr.where(theta_potential > capacity, theta_potential - capacity, 0)

    excess.attrs.update({
        'long_name': 'Saturation excess runoff',
        'units': 'mm/day',
    })

    return excess
