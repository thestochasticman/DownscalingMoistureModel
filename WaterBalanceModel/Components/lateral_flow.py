"""Lateral flow routing for the water balance model.

This module implements terrain-based lateral subsurface flow using
D8 flow direction routing. Lateral flow redistributes soil moisture
from upslope to downslope areas based on topographic gradients.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import sparse


# D8 direction encoding (standard pysheds convention)
# Direction: E, SE, S, SW, W, NW, N, NE
D8_DIRECTIONS = {
    1: (0, 1),    # E
    2: (1, 1),    # SE
    4: (1, 0),    # S
    8: (1, -1),   # SW
    16: (0, -1),  # W
    32: (-1, -1), # NW
    64: (-1, 0),  # N
    128: (-1, 1), # NE
}


def build_flow_routing_matrix(
    flow_dir: xr.DataArray,
    slope: xr.DataArray,
) -> sparse.csr_matrix:
    """Build sparse routing matrix for lateral flow.

    Each cell routes water to its downslope neighbor based on D8
    flow direction. The matrix A has shape (n_cells, n_cells) where
    A[j, i] represents the fraction of outflow from cell i that goes
    to cell j.

    Args:
        flow_dir: D8 flow direction grid (values 1, 2, 4, 8, 16, 32, 64, 128).
        slope: Local slope (radians).

    Returns:
        Sparse CSR matrix for routing. A @ outflow gives inflow.
    """
    ny, nx = flow_dir.shape
    n = ny * nx

    # Flatten arrays
    fdir = flow_dir.values.ravel()
    slp = slope.values.ravel()

    # Build sparse matrix entries
    rows = []
    cols = []
    data = []

    for i in range(n):
        # Get 2D coordinates
        iy, ix = divmod(i, nx)

        # Get flow direction
        d = int(fdir[i])
        if d not in D8_DIRECTIONS:
            continue

        # Get downstream neighbor
        dy, dx = D8_DIRECTIONS[d]
        jy, jx = iy + dy, ix + dx

        # Check bounds
        if 0 <= jy < ny and 0 <= jx < nx:
            j = jy * nx + jx

            # Weight by slope (steeper = faster flow)
            weight = 1.0  # Could use slope-based weighting

            rows.append(j)
            cols.append(i)
            data.append(weight)

    # Create sparse matrix
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    return A


def compute_lateral_outflow(
    theta: xr.DataArray,
    theta_fc: xr.DataArray,
    theta_sat: xr.DataArray,
    slope: xr.DataArray,
    k_sat: xr.DataArray,
    cell_width: float,
) -> xr.DataArray:
    """Compute lateral subsurface outflow from each cell.

    Lateral flow occurs when soil moisture exceeds field capacity.
    Flow rate depends on slope and hydraulic conductivity:

    L_out = K_sat * sin(slope) * (theta - theta_fc) / (theta_sat - theta_fc)

    Args:
        theta: Current soil moisture (mm).
        theta_fc: Field capacity (mm).
        theta_sat: Saturated moisture (mm).
        slope: Local slope (radians).
        k_sat: Saturated hydraulic conductivity (mm/day).
        cell_width: Grid cell width (m).

    Returns:
        Lateral outflow (mm/day).
    """
    # Excess above field capacity
    excess = theta - theta_fc
    excess = xr.where(excess > 0, excess, 0)

    # Relative saturation of drainable pore space
    pore_space = theta_sat - theta_fc
    rel_sat = excess / (pore_space + 1e-10)
    rel_sat = rel_sat.clip(min=0, max=1)

    # Lateral velocity (Darcy-based)
    # Using sin(slope) for gradient
    slope_gradient = np.sin(slope)

    # Outflow rate
    outflow = k_sat * slope_gradient * rel_sat

    # Scale by cell width to get volume flux
    # (already in mm/day as k_sat is mm/day)

    outflow.attrs.update({
        'long_name': 'Lateral subsurface outflow',
        'units': 'mm/day',
    })

    return outflow


def compute_lateral_inflow(
    outflow: xr.DataArray,
    routing_matrix: sparse.csr_matrix,
) -> xr.DataArray:
    """Compute lateral inflow from upslope cells.

    Args:
        outflow: Lateral outflow from each cell (mm/day).
        routing_matrix: Sparse routing matrix from build_flow_routing_matrix.

    Returns:
        Lateral inflow (mm/day).
    """
    # Flatten, apply routing, reshape
    shape = outflow.shape
    outflow_flat = outflow.values.ravel()

    inflow_flat = routing_matrix @ outflow_flat

    inflow = xr.DataArray(
        inflow_flat.reshape(shape),
        dims=outflow.dims,
        coords=outflow.coords,
        attrs={
            'long_name': 'Lateral subsurface inflow',
            'units': 'mm/day',
        },
    )

    return inflow


def compute_lateral_flow(
    theta: xr.DataArray,
    theta_fc: xr.DataArray,
    theta_sat: xr.DataArray,
    terrain: xr.Dataset,
    k_sat: xr.DataArray,
    routing_matrix: sparse.csr_matrix,
    cell_width: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute lateral flow (both inflow and outflow).

    Args:
        theta: Current soil moisture (mm).
        theta_fc: Field capacity (mm).
        theta_sat: Saturated moisture (mm).
        terrain: Terrain dataset with 'slope' variable.
        k_sat: Saturated hydraulic conductivity (mm/day).
        routing_matrix: Flow routing matrix.
        cell_width: Grid cell width (m).

    Returns:
        Tuple of (lateral_in, lateral_out) in mm/day.
    """
    # Compute outflow
    outflow = compute_lateral_outflow(
        theta, theta_fc, theta_sat,
        terrain['slope'], k_sat, cell_width,
    )

    # Compute inflow
    inflow = compute_lateral_inflow(outflow, routing_matrix)

    return inflow, outflow


def compute_twi_wetness_factor(twi: xr.DataArray, twi_mean: float = 8.0) -> xr.DataArray:
    """Compute wetness factor based on TWI.

    High TWI areas (convergent zones) tend to be wetter.
    This factor can be used to adjust initial conditions or
    steady-state moisture distribution.

    Args:
        twi: Topographic Wetness Index.
        twi_mean: Reference TWI value (default 8.0).

    Returns:
        Wetness factor (>1 for high TWI, <1 for low TWI).
    """
    factor = twi / twi_mean
    factor = factor.clip(min=0.5, max=2.0)

    factor.attrs.update({
        'long_name': 'TWI-based wetness factor',
        'units': '',
    })

    return factor
