"""Load soil parameters from SLGA via PaddockTS with pedotransfer functions.

This module wraps PaddockTS SLGA data access and applies Saxton & Rawls (2006)
pedotransfer functions to derive soil hydraulic parameters needed for the
water balance model:

- theta_sat: Saturated water content (m3/m3)
- theta_fc: Field capacity at 33 kPa (m3/m3)
- theta_wp: Wilting point at 1500 kPa (m3/m3)
- k_sat: Saturated hydraulic conductivity (mm/day)

References:
    Saxton, K.E. and Rawls, W.J. (2006). Soil Water Characteristic Estimates
    by Texture and Organic Matter for Hydrologic Solutions. Soil Science
    Society of America Journal, 70(5), 1569-1578.
"""

from __future__ import annotations

import sys
from os.path import exists
from typing import TYPE_CHECKING

import numpy as np
import rasterio
import xarray as xr

if TYPE_CHECKING:
    from WaterBalanceModel.query import WaterBalanceQuery

# Add PaddockTS to path if needed
sys.path.insert(0, '/borevitz_projects/repos/paddock-ts-local')


def saxton_rawls_theta_sat(sand: np.ndarray, clay: np.ndarray, om: float = 2.5) -> np.ndarray:
    """Compute saturated water content using Saxton & Rawls (2006).

    Args:
        sand: Sand content (% by weight, 0-100).
        clay: Clay content (% by weight, 0-100).
        om: Organic matter content (%, default 2.5).

    Returns:
        Saturated water content (m3/m3).
    """
    S = sand / 100
    C = clay / 100
    OM = om / 100

    # Equation 2 from Saxton & Rawls (2006)
    theta_s = (
        0.278 * S
        + 0.034 * C
        + 0.022 * OM
        - 0.018 * S * OM
        - 0.027 * C * OM
        - 0.584 * S * C
        + 0.078
    )

    # Equation 3: density effect
    theta_s = theta_s + (1.636 * theta_s - 0.107)

    return np.clip(theta_s, 0.3, 0.6)


def saxton_rawls_theta_fc(sand: np.ndarray, clay: np.ndarray, om: float = 2.5) -> np.ndarray:
    """Compute field capacity (33 kPa) using Saxton & Rawls (2006).

    Args:
        sand: Sand content (% by weight, 0-100).
        clay: Clay content (% by weight, 0-100).
        om: Organic matter content (%, default 2.5).

    Returns:
        Field capacity water content (m3/m3).
    """
    S = sand / 100
    C = clay / 100
    OM = om / 100

    # Equation 5 from Saxton & Rawls (2006) - 33 kPa moisture
    theta_33t = (
        -0.251 * S
        + 0.195 * C
        + 0.011 * OM
        + 0.006 * S * OM
        - 0.027 * C * OM
        + 0.452 * S * C
        + 0.299
    )

    # Equation 6: adjusted for density
    theta_33 = theta_33t + (1.283 * theta_33t * theta_33t - 0.374 * theta_33t - 0.015)

    return np.clip(theta_33, 0.1, 0.5)


def saxton_rawls_theta_wp(sand: np.ndarray, clay: np.ndarray, om: float = 2.5) -> np.ndarray:
    """Compute wilting point (1500 kPa) using Saxton & Rawls (2006).

    Args:
        sand: Sand content (% by weight, 0-100).
        clay: Clay content (% by weight, 0-100).
        om: Organic matter content (%, default 2.5).

    Returns:
        Wilting point water content (m3/m3).
    """
    S = sand / 100
    C = clay / 100
    OM = om / 100

    # Equation 7 from Saxton & Rawls (2006) - 1500 kPa moisture
    theta_1500t = (
        -0.024 * S
        + 0.487 * C
        + 0.006 * OM
        + 0.005 * S * OM
        - 0.013 * C * OM
        + 0.068 * S * C
        + 0.031
    )

    # Equation 8: adjusted
    theta_1500 = theta_1500t + (0.14 * theta_1500t - 0.02)

    return np.clip(theta_1500, 0.01, 0.3)


def saxton_rawls_k_sat(
    sand: np.ndarray,
    clay: np.ndarray,
    theta_sat: np.ndarray,
    theta_fc: np.ndarray,
) -> np.ndarray:
    """Compute saturated hydraulic conductivity using Saxton & Rawls (2006).

    Args:
        sand: Sand content (% by weight, 0-100).
        clay: Clay content (% by weight, 0-100).
        theta_sat: Saturated water content (m3/m3).
        theta_fc: Field capacity (m3/m3).

    Returns:
        Saturated hydraulic conductivity (mm/day).
    """
    S = sand / 100
    C = clay / 100

    # Equation 15: lambda (pore size distribution)
    # Simplified approximation
    B = (np.log(1500) - np.log(33)) / (np.log(theta_fc) - np.log(theta_sat - 0.001))
    B = np.clip(B, 1, 15)

    # Equation 16: K_sat (mm/hr originally, convert to mm/day)
    k_sat_mm_hr = 1930 * (theta_sat - theta_fc) ** (3 - 1 / B)
    k_sat_mm_day = k_sat_mm_hr * 24

    return np.clip(k_sat_mm_day, 1, 5000)


def load_soil_parameters(query: WaterBalanceQuery, depth: str = '5-15cm') -> xr.Dataset:
    """Load soil hydraulic parameters derived from SLGA texture data.

    Downloads SLGA sand, clay, and silt data via PaddockTS, applies
    Saxton & Rawls (2006) pedotransfer functions to compute hydraulic
    parameters at 90m resolution.

    Cached: If the output NetCDF already exists at ``query.soil_params_path``,
    it is loaded and returned.

    Args:
        query: The water balance query specifying bbox.
        depth: SLGA depth slice (default '5-15cm'). Options include
            '0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm'.

    Returns:
        xr.Dataset with variables:
            - sand: Sand content (%)
            - clay: Clay content (%)
            - silt: Silt content (%)
            - theta_sat: Saturated water content (m3/m3)
            - theta_fc: Field capacity (m3/m3)
            - theta_wp: Wilting point (m3/m3)
            - k_sat: Saturated hydraulic conductivity (mm/day)
    """
    if exists(query.soil_params_path):
        print(f'  cached: {query.soil_params_path}')
        return xr.open_dataset(query.soil_params_path)

    # Import PaddockTS modules
    from PaddockTS.Environmental.SLGASoils.download_slgasoils import (
        download_slga_soils,
        get_filename,
    )
    from PaddockTS.query import Query as PaddockQuery

    # Create PaddockTS query
    pquery = PaddockQuery(
        bbox=query.bbox,
        start=query.start,
        end=query.end,
        stub=query.stub,
    )

    # Download SLGA texture data
    print('  downloading SLGA soil texture data...', flush=True)
    download_slga_soils(pquery, vars=['Clay', 'Sand', 'Silt'], depths=[depth])

    # Read the downloaded GeoTIFFs
    sand_file = get_filename(pquery, 'Sand', depth)
    clay_file = get_filename(pquery, 'Clay', depth)
    silt_file = get_filename(pquery, 'Silt', depth)

    with rasterio.open(sand_file) as src:
        sand = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs

    with rasterio.open(clay_file) as src:
        clay = src.read(1).astype(np.float32)

    with rasterio.open(silt_file) as src:
        silt = src.read(1).astype(np.float32)

    # Apply pedotransfer functions
    print('  computing soil hydraulic parameters...', flush=True)
    theta_sat = saxton_rawls_theta_sat(sand, clay)
    theta_fc = saxton_rawls_theta_fc(sand, clay)
    theta_wp = saxton_rawls_theta_wp(sand, clay)
    k_sat = saxton_rawls_k_sat(sand, clay, theta_sat, theta_fc)

    # Build coordinate arrays
    ny, nx = sand.shape
    x = np.arange(nx) * transform.a + transform.c + transform.a / 2
    y = np.arange(ny) * transform.e + transform.f + transform.e / 2

    # Create dataset
    ds = xr.Dataset(
        data_vars={
            'sand': (['y', 'x'], sand, {'units': '%', 'long_name': 'Sand content'}),
            'clay': (['y', 'x'], clay, {'units': '%', 'long_name': 'Clay content'}),
            'silt': (['y', 'x'], silt, {'units': '%', 'long_name': 'Silt content'}),
            'theta_sat': (['y', 'x'], theta_sat, {'units': 'm3/m3', 'long_name': 'Saturated water content'}),
            'theta_fc': (['y', 'x'], theta_fc, {'units': 'm3/m3', 'long_name': 'Field capacity (33 kPa)'}),
            'theta_wp': (['y', 'x'], theta_wp, {'units': 'm3/m3', 'long_name': 'Wilting point (1500 kPa)'}),
            'k_sat': (['y', 'x'], k_sat, {'units': 'mm/day', 'long_name': 'Saturated hydraulic conductivity'}),
        },
        coords={
            'x': x,
            'y': y,
        },
        attrs={
            'crs': str(crs),
            'resolution': abs(transform.a),
            'source': 'SLGA + Saxton & Rawls (2006) pedotransfer',
            'depth': depth,
        },
    )

    # Save to cache (compute to avoid dask scheduler issues)
    ds.compute().to_netcdf(query.soil_params_path)
    print(f'  saved: {query.soil_params_path}')

    return ds
