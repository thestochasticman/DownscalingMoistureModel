"""SMIPS-based constraint for soil moisture downscaling.

This module applies a mass conservation constraint to ensure that
the high-resolution water balance output, when aggregated to SMIPS
resolution (~1km), matches the observed SMIPS soil moisture.

Uses convex optimization (CVXPY) to adjust the model output while
preserving spatial patterns and enforcing smoothness.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import sparse
from scipy.sparse import csr_matrix, diags


def build_aggregation_matrix(
    smips: xr.DataArray,
    fine: xr.DataArray,
) -> csr_matrix:
    """Build sparse aggregation matrix from fine to coarse grid.

    Creates matrix A such that A @ fine_values = coarse_values,
    where each row averages the fine pixels within a coarse pixel.

    Args:
        smips: Coarse SMIPS grid defining aggregation zones.
        fine: Fine-resolution grid to aggregate.

    Returns:
        Sparse CSR matrix of shape (n_coarse, n_fine).
    """
    # Get grid shapes
    ny_c, nx_c = smips.shape
    ny_f, nx_f = fine.shape
    n_coarse = ny_c * nx_c
    n_fine = ny_f * nx_f

    # Get coordinate bounds
    smips_y = smips.y.values
    smips_x = smips.x.values
    fine_y = fine.y.values
    fine_x = fine.x.values

    # Compute coarse cell boundaries
    dy_c = abs(smips_y[1] - smips_y[0]) if len(smips_y) > 1 else 0.01
    dx_c = abs(smips_x[1] - smips_x[0]) if len(smips_x) > 1 else 0.01

    # Build mapping
    rows = []
    cols = []
    data = []

    # For each coarse cell, find fine cells within it
    for ic in range(ny_c):
        for jc in range(nx_c):
            k_coarse = ic * nx_c + jc

            # Coarse cell bounds
            y_c = smips_y[ic]
            x_c = smips_x[jc]
            y_min = y_c - dy_c / 2
            y_max = y_c + dy_c / 2
            x_min = x_c - dx_c / 2
            x_max = x_c + dx_c / 2

            # Find fine cells within bounds
            fine_in_coarse = []
            for if_ in range(ny_f):
                for jf in range(nx_f):
                    y_f = fine_y[if_]
                    x_f = fine_x[jf]

                    if y_min <= y_f <= y_max and x_min <= x_f <= x_max:
                        k_fine = if_ * nx_f + jf
                        fine_in_coarse.append(k_fine)

            # Add entries for averaging
            if fine_in_coarse:
                weight = 1.0 / len(fine_in_coarse)
                for k_fine in fine_in_coarse:
                    rows.append(k_coarse)
                    cols.append(k_fine)
                    data.append(weight)

    A = csr_matrix((data, (rows, cols)), shape=(n_coarse, n_fine))
    return A


def build_laplacian_matrix(ny: int, nx: int) -> csr_matrix:
    """Build 2D discrete Laplacian for spatial smoothness.

    Uses 4-connectivity (up, down, left, right neighbors).
    L @ theta computes discrete second derivatives.

    Args:
        ny: Number of rows.
        nx: Number of columns.

    Returns:
        Sparse CSR matrix of shape (n, n) where n = ny * nx.
    """
    n = ny * nx

    # Diagonal: each pixel has up to 4 neighbors
    main_diag = np.ones(n) * 4

    # Off-diagonals for left/right neighbors
    lr_diag = -np.ones(n - 1)
    # Zero out connections across row boundaries
    for i in range(1, ny):
        lr_diag[i * nx - 1] = 0

    # Off-diagonals for up/down neighbors
    ud_diag = -np.ones(n - nx)

    # Construct sparse matrix
    L = diags(
        [main_diag, lr_diag, lr_diag, ud_diag, ud_diag],
        [0, -1, 1, -nx, nx],
        shape=(n, n),
        format='csr',
    )

    return L


def apply_smips_constraint(
    theta_model: xr.DataArray,
    smips_obs: xr.DataArray,
    lambda_smoothness: float = 0.5,
    solver: str = 'SCS',
    verbose: bool = False,
) -> xr.DataArray:
    """Apply SMIPS mass conservation constraint to model output.

    Solves the convex optimization problem:

        minimize ||θ - θ_model||² + λ||L @ θ||²
        subject to: A @ θ = smips_obs (mass conservation)
                   θ >= 0 (non-negativity)

    Where:
        - θ: Adjusted fine-scale soil moisture
        - θ_model: Water balance model output (prior)
        - L: Laplacian matrix (spatial smoothness)
        - A: Aggregation matrix (fine → coarse)
        - λ: Smoothness regularization weight

    Args:
        theta_model: Model output at fine resolution (2D array).
        smips_obs: SMIPS observation at coarse resolution (2D array).
        lambda_smoothness: Regularization weight for smoothness.
        solver: CVXPY solver ('SCS', 'OSQP', 'ECOS').
        verbose: Print solver output.

    Returns:
        Constrained soil moisture at fine resolution.
    """
    import cvxpy as cp

    # Get shapes
    ny_f, nx_f = theta_model.shape
    n_fine = ny_f * nx_f

    # Build matrices
    A = build_aggregation_matrix(smips_obs, theta_model)
    L = build_laplacian_matrix(ny_f, nx_f)

    # Flatten arrays
    theta_prior = theta_model.values.ravel()
    smips_flat = smips_obs.values.ravel()

    # Remove invalid SMIPS pixels
    valid_mask = ~np.isnan(smips_flat)
    A_valid = A[valid_mask, :]
    smips_valid = smips_flat[valid_mask]

    if len(smips_valid) == 0:
        print('  warning: no valid SMIPS observations, returning model output')
        return theta_model

    # Define optimization problem
    theta = cp.Variable(n_fine)

    # Objective: data fidelity + smoothness
    data_term = cp.sum_squares(theta - theta_prior)
    smooth_term = cp.sum_squares(L @ theta)
    objective = cp.Minimize(data_term + lambda_smoothness * smooth_term)

    # Constraints
    constraints = [
        A_valid @ theta == smips_valid,  # Mass conservation
        theta >= 0,  # Non-negativity
    ]

    # Solve
    problem = cp.Problem(objective, constraints)
    try:
        if solver == 'SCS':
            problem.solve(solver=cp.SCS, verbose=verbose, max_iters=5000)
        elif solver == 'OSQP':
            problem.solve(solver=cp.OSQP, verbose=verbose)
        elif solver == 'ECOS':
            problem.solve(solver=cp.ECOS, verbose=verbose)
        else:
            problem.solve(verbose=verbose)
    except Exception as e:
        print(f'  warning: solver failed ({e}), returning model output')
        return theta_model

    if problem.status not in ['optimal', 'optimal_inaccurate']:
        print(f'  warning: solver status {problem.status}, returning model output')
        return theta_model

    # Reshape result
    theta_constrained = theta.value.reshape(ny_f, nx_f)

    result = xr.DataArray(
        theta_constrained,
        dims=theta_model.dims,
        coords=theta_model.coords,
        attrs={
            **theta_model.attrs,
            'smips_constrained': True,
            'lambda_smoothness': lambda_smoothness,
            'solver': solver,
        },
    )

    return result


def apply_smips_constraint_timeseries(
    theta_model: xr.DataArray,
    smips_obs: xr.DataArray,
    lambda_smoothness: float = 0.5,
    max_gap_days: int = 1,
    solver: str = 'SCS',
    verbose: bool = False,
) -> xr.DataArray:
    """Apply SMIPS constraint to each timestep in a time series.

    Args:
        theta_model: Model output with dims (time, y, x).
        smips_obs: SMIPS observations with dims (time, y, x).
        lambda_smoothness: Regularization weight.
        max_gap_days: Maximum temporal gap for SMIPS matching.
        solver: CVXPY solver.
        verbose: Print solver output.

    Returns:
        Constrained soil moisture time series.
    """
    import pandas as pd

    times = pd.to_datetime(theta_model.time.values)
    smips_times = pd.to_datetime(smips_obs.time.values)

    result = theta_model.copy()

    for t, time in enumerate(times):
        # Find nearest SMIPS observation
        time_diffs = np.abs(smips_times - time)
        min_diff = time_diffs.min()

        if min_diff > pd.Timedelta(days=max_gap_days):
            continue

        nearest_idx = time_diffs.argmin()
        smips_t = smips_obs.isel(time=nearest_idx)

        # Apply constraint
        theta_t = theta_model.isel(time=t)
        theta_constrained = apply_smips_constraint(
            theta_t, smips_t, lambda_smoothness, solver, verbose,
        )

        result[t, :, :] = theta_constrained.values

    result.attrs['smips_constrained'] = True

    return result
