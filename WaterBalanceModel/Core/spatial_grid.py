"""Spatial grid utilities for the water balance model.

This module handles grid interpolation, resampling, and alignment
between different data sources (terrain at 30m, Sentinel-2 at 10m,
SMIPS at 1km, soil at 90m).
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage


def resample_to_target(
    source: xr.DataArray,
    target: xr.DataArray,
    method: str = 'bilinear',
) -> xr.DataArray:
    """Resample source array to match target grid.

    Args:
        source: Source DataArray to resample.
        target: Target DataArray defining the output grid.
        method: Interpolation method ('nearest', 'bilinear').

    Returns:
        Resampled array matching target coordinates.
    """
    # Use xarray's interp for coordinate-based interpolation
    result = source.interp(
        y=target.y,
        x=target.x,
        method='linear' if method == 'bilinear' else 'nearest',
    )

    return result


def align_grids(
    *arrays: xr.DataArray,
    target_resolution: float = 10.0,
    method: str = 'bilinear',
) -> list[xr.DataArray]:
    """Align multiple arrays to a common grid.

    Finds the intersection of all array extents and resamples
    to the specified target resolution.

    Args:
        *arrays: Variable number of DataArrays to align.
        target_resolution: Target grid resolution in same units as coords.
        method: Interpolation method.

    Returns:
        List of aligned arrays.
    """
    if not arrays:
        return []

    # Find common extent
    x_min = max(arr.x.min().item() for arr in arrays)
    x_max = min(arr.x.max().item() for arr in arrays)
    y_min = max(arr.y.min().item() for arr in arrays)
    y_max = min(arr.y.max().item() for arr in arrays)

    # Create target coordinates
    x_target = np.arange(x_min, x_max, target_resolution)
    y_target = np.arange(y_max, y_min, -target_resolution)  # Descending for image convention

    # Resample each array
    aligned = []
    for arr in arrays:
        resampled = arr.interp(
            x=x_target,
            y=y_target,
            method='linear' if method == 'bilinear' else 'nearest',
        )
        aligned.append(resampled)

    return aligned


def compute_grid_cell_area(
    y: xr.DataArray,
    x: xr.DataArray,
    crs: str = 'EPSG:6933',
) -> xr.DataArray:
    """Compute grid cell area in square meters.

    For projected CRS, cell area is simply dx * dy.
    For geographic CRS, area varies with latitude.

    Args:
        y: Y coordinates.
        x: X coordinates.
        crs: Coordinate reference system.

    Returns:
        2D array of cell areas (m^2).
    """
    # Get cell sizes
    dx = abs(float(x[1] - x[0])) if len(x) > 1 else 10.0
    dy = abs(float(y[1] - y[0])) if len(y) > 1 else 10.0

    if crs.startswith('EPSG:4326'):
        # Geographic - area varies with latitude
        lat = y.values
        lat_rad = np.radians(lat)

        # Approximate meters per degree
        m_per_deg_lat = 111320
        m_per_deg_lon = 111320 * np.cos(lat_rad)

        area_m2 = (dy * m_per_deg_lat) * (dx * m_per_deg_lon[:, np.newaxis])
    else:
        # Projected - constant cell size
        area_m2 = dx * dy * np.ones((len(y), len(x)))

    return xr.DataArray(
        area_m2,
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'units': 'm^2', 'long_name': 'Grid cell area'},
    )


def broadcast_point_to_grid(
    value: float,
    template: xr.DataArray,
) -> xr.DataArray:
    """Broadcast a scalar value to match a template grid.

    Args:
        value: Scalar value to broadcast.
        template: Template DataArray defining the grid shape.

    Returns:
        DataArray filled with the scalar value.
    """
    return xr.DataArray(
        np.full(template.shape, value, dtype=np.float32),
        dims=template.dims,
        coords=template.coords,
    )


def smooth_field(
    field: xr.DataArray,
    sigma: float = 1.0,
) -> xr.DataArray:
    """Apply Gaussian smoothing to a field.

    Args:
        field: Input field.
        sigma: Smoothing scale (grid cells).

    Returns:
        Smoothed field.
    """
    smoothed = ndimage.gaussian_filter(field.values, sigma=sigma)

    return xr.DataArray(
        smoothed,
        dims=field.dims,
        coords=field.coords,
        attrs=field.attrs,
    )


def fill_nan_nearest(field: xr.DataArray) -> xr.DataArray:
    """Fill NaN values with nearest valid value.

    Args:
        field: Input field with NaN values.

    Returns:
        Field with NaN values filled.
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.isnan(field.values)
    if not mask.any():
        return field

    # Find indices of nearest valid values
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)

    filled = field.values[tuple(indices)]

    return xr.DataArray(
        filled,
        dims=field.dims,
        coords=field.coords,
        attrs=field.attrs,
    )
