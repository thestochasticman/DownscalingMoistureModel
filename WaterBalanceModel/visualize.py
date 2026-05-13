"""Visualization module for water balance model outputs.

Provides functions to plot:
- Spatial maps of soil moisture and fluxes
- Time series at specific locations
- Water balance component summaries
- Spatial-temporal animations
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from WaterBalanceModel.query import WaterBalanceQuery


def plot_spatial_snapshot(
    ds: xr.Dataset,
    time_idx: int | str = 0,
    variables: list[str] | None = None,
    figsize: tuple[float, float] = (14, 10),
    cmap_moisture: str = 'Blues',
    cmap_flux: str = 'YlOrRd',
) -> Figure:
    """Plot spatial maps of model variables at a single timestep.

    Args:
        ds: Water balance model output dataset.
        time_idx: Time index (int) or date string ('YYYY-MM-DD').
        variables: List of variables to plot. If None, plots all.
        figsize: Figure size.
        cmap_moisture: Colormap for soil moisture.
        cmap_flux: Colormap for flux variables.

    Returns:
        Matplotlib Figure.
    """
    if variables is None:
        variables = ['soil_moisture', 'et_actual', 'runoff', 'drainage', 'lateral_in', 'lateral_out']

    # Select timestep
    if isinstance(time_idx, str):
        ds_t = ds.sel(time=time_idx, method='nearest')
        time_label = time_idx
    else:
        ds_t = ds.isel(time=time_idx)
        time_label = str(ds_t.time.values)[:10]

    n_vars = len(variables)
    ncols = min(3, n_vars)
    nrows = (n_vars + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, var in enumerate(variables):
        ax = axes[i]
        data = ds_t[var]

        cmap = cmap_moisture if var == 'soil_moisture' else cmap_flux
        im = ax.pcolormesh(
            data.x, data.y, data.values,
            cmap=cmap, shading='auto'
        )
        ax.set_aspect('equal')
        ax.set_title(f'{var}\n{data.attrs.get("long_name", "")}')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(data.attrs.get('units', ''))

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Water Balance Model Output - {time_label}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_timeseries_at_point(
    ds: xr.Dataset,
    x: float | None = None,
    y: float | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Plot time series of all variables at a specific point.

    Args:
        ds: Water balance model output dataset.
        x: X coordinate. If None, uses center of domain.
        y: Y coordinate. If None, uses center of domain.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    # Default to center of domain
    if x is None:
        x = float(ds.x.values[len(ds.x) // 2])
    if y is None:
        y = float(ds.y.values[len(ds.y) // 2])

    # Select nearest point
    ds_point = ds.sel(x=x, y=y, method='nearest')
    actual_x = float(ds_point.x)
    actual_y = float(ds_point.y)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

    # Panel 1: Soil moisture
    ax1 = fig.add_subplot(gs[0])
    sm = ds_point['soil_moisture']
    ax1.fill_between(sm.time.values, 0, sm.values, alpha=0.3, color='blue')
    ax1.plot(sm.time.values, sm.values, 'b-', linewidth=1.5, label='Soil Moisture')
    ax1.set_ylabel('Soil Moisture (mm)')
    ax1.set_title(f'Time Series at x={actual_x:.1f}, y={actual_y:.1f}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: ET and precipitation proxy (runoff as indicator of precip events)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    et = ds_point['et_actual']
    runoff = ds_point['runoff']
    ax2.plot(et.time.values, et.values, 'g-', linewidth=1.5, label='ET')
    ax2.bar(runoff.time.values, runoff.values, alpha=0.5, color='red', label='Runoff', width=1)
    ax2.set_ylabel('Flux (mm/day)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Drainage and lateral flow
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    drainage = ds_point['drainage']
    lat_in = ds_point['lateral_in']
    lat_out = ds_point['lateral_out']
    ax3.plot(drainage.time.values, drainage.values, 'brown', linewidth=1.5, label='Drainage')
    ax3.plot(lat_in.time.values, lat_in.values, 'c-', linewidth=1, label='Lateral In')
    ax3.plot(lat_out.time.values, lat_out.values, 'm-', linewidth=1, label='Lateral Out')
    ax3.set_ylabel('Flux (mm/day)')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    fig.tight_layout()
    return fig


def plot_water_balance_summary(
    ds: xr.Dataset,
    figsize: tuple[float, float] = (14, 6),
) -> Figure:
    """Plot domain-averaged water balance components over time.

    Args:
        ds: Water balance model output dataset.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    # Compute spatial means
    sm_mean = ds['soil_moisture'].mean(dim=['x', 'y'])
    et_mean = ds['et_actual'].mean(dim=['x', 'y'])
    runoff_mean = ds['runoff'].mean(dim=['x', 'y'])
    drainage_mean = ds['drainage'].mean(dim=['x', 'y'])
    lat_net = (ds['lateral_in'] - ds['lateral_out']).mean(dim=['x', 'y'])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left panel: Time series
    ax1 = axes[0]
    ax1.plot(sm_mean.time.values, sm_mean.values, 'b-', linewidth=2, label='Soil Moisture (mm)')
    ax1.set_ylabel('Soil Moisture (mm)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel('Date')
    ax1.set_title('Domain-Averaged Water Balance')
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(et_mean.time.values, et_mean.values, 'g-', linewidth=1.5, label='ET')
    ax1b.plot(runoff_mean.time.values, runoff_mean.values, 'r-', linewidth=1.5, label='Runoff')
    ax1b.plot(drainage_mean.time.values, drainage_mean.values, 'brown', linewidth=1.5, label='Drainage')
    ax1b.set_ylabel('Flux (mm/day)', color='gray')
    ax1b.tick_params(axis='y', labelcolor='gray')
    ax1b.legend(loc='upper right')

    # Right panel: Cumulative fluxes
    ax2 = axes[1]
    time_days = np.arange(len(et_mean.time))

    cum_et = np.cumsum(et_mean.values)
    cum_runoff = np.cumsum(runoff_mean.values)
    cum_drainage = np.cumsum(drainage_mean.values)

    ax2.fill_between(time_days, 0, cum_et, alpha=0.3, color='green', label='Cumulative ET')
    ax2.fill_between(time_days, cum_et, cum_et + cum_runoff, alpha=0.3, color='red', label='Cumulative Runoff')
    ax2.fill_between(time_days, cum_et + cum_runoff, cum_et + cum_runoff + cum_drainage,
                     alpha=0.3, color='brown', label='Cumulative Drainage')

    ax2.set_xlabel('Days')
    ax2.set_ylabel('Cumulative Flux (mm)')
    ax2.set_title('Cumulative Water Balance Components')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_soil_moisture_animation_frames(
    ds: xr.Dataset,
    output_dir: str | Path,
    step: int = 1,
    figsize: tuple[float, float] = (8, 6),
    cmap: str = 'Blues',
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[Path]:
    """Save individual frames for soil moisture animation.

    Args:
        ds: Water balance model output dataset.
        output_dir: Directory to save frames.
        step: Save every nth frame.
        figsize: Figure size.
        cmap: Colormap.
        vmin: Minimum value for colorbar.
        vmax: Maximum value for colorbar.

    Returns:
        List of saved frame paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sm = ds['soil_moisture']
    if vmin is None:
        vmin = float(sm.min())
    if vmax is None:
        vmax = float(sm.max())

    frame_paths = []
    n_times = len(sm.time)

    for i in range(0, n_times, step):
        fig, ax = plt.subplots(figsize=figsize)

        data = sm.isel(time=i)
        time_str = str(data.time.values)[:10]

        im = ax.pcolormesh(
            data.x, data.y, data.values,
            cmap=cmap, shading='auto',
            norm=Normalize(vmin=vmin, vmax=vmax)
        )
        ax.set_aspect('equal')
        ax.set_title(f'Soil Moisture - {time_str}')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('mm')

        frame_path = output_dir / f'frame_{i:04d}.png'
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_path)

    return frame_paths


def plot_comparison_with_smips(
    ds: xr.Dataset,
    smips: xr.DataArray,
    figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """Plot comparison between model output and SMIPS observations.

    Args:
        ds: Water balance model output dataset.
        smips: SMIPS soil moisture observations.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    # Aggregate model to SMIPS resolution
    sm_model = ds['soil_moisture'].mean(dim=['x', 'y'])

    # Get SMIPS time series (assuming single grid cell or averaged)
    if 'x' in smips.dims and 'y' in smips.dims:
        smips_ts = smips.mean(dim=['x', 'y'])
    else:
        smips_ts = smips

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Time series comparison
    ax1 = axes[0]
    ax1.plot(sm_model.time.values, sm_model.values, 'b-', linewidth=1.5, label='Model (domain mean)')
    ax1.plot(smips_ts.time.values, smips_ts.values, 'ro', markersize=4, alpha=0.7, label='SMIPS')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Soil Moisture (mm)')
    ax1.set_title('Model vs SMIPS Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Scatter plot
    ax2 = axes[1]
    # Interpolate SMIPS to model times for scatter
    smips_interp = smips_ts.interp(time=sm_model.time, method='nearest')
    valid = ~np.isnan(smips_interp.values)

    if valid.sum() > 0:
        ax2.scatter(smips_interp.values[valid], sm_model.values[valid], alpha=0.5, s=20)

        # 1:1 line
        min_val = min(smips_interp.values[valid].min(), sm_model.values[valid].min())
        max_val = max(smips_interp.values[valid].max(), sm_model.values[valid].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='1:1')

        # Compute R2
        corr = np.corrcoef(smips_interp.values[valid], sm_model.values[valid])[0, 1]
        ax2.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax2.transAxes,
                 fontsize=12, verticalalignment='top')

    ax2.set_xlabel('SMIPS (mm)')
    ax2.set_ylabel('Model (mm)')
    ax2.set_title('Model vs SMIPS Scatter')
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_flux_maps(
    ds: xr.Dataset,
    time_idx: int | str = 0,
    figsize: tuple[float, float] = (14, 8),
) -> Figure:
    """Plot spatial maps of all flux components with unified scale per row.

    Args:
        ds: Water balance model output dataset.
        time_idx: Time index or date string.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    if isinstance(time_idx, str):
        ds_t = ds.sel(time=time_idx, method='nearest')
        time_label = time_idx
    else:
        ds_t = ds.isel(time=time_idx)
        time_label = str(ds_t.time.values)[:10]

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Row 1: Main fluxes
    flux_vars_1 = ['et_actual', 'runoff', 'drainage']
    # Row 2: Lateral flow and net
    flux_vars_2 = ['lateral_in', 'lateral_out']

    cmaps = {
        'et_actual': 'Greens',
        'runoff': 'Reds',
        'drainage': 'YlOrBr',
        'lateral_in': 'Blues',
        'lateral_out': 'Purples',
    }

    # Plot first row
    for i, var in enumerate(flux_vars_1):
        ax = axes[0, i]
        data = ds_t[var]
        im = ax.pcolormesh(data.x, data.y, data.values, cmap=cmaps[var], shading='auto')
        ax.set_aspect('equal')
        ax.set_title(var.replace('_', ' ').title())
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('mm/day')

    # Plot second row
    for i, var in enumerate(flux_vars_2):
        ax = axes[1, i]
        data = ds_t[var]
        im = ax.pcolormesh(data.x, data.y, data.values, cmap=cmaps[var], shading='auto')
        ax.set_aspect('equal')
        ax.set_title(var.replace('_', ' ').title())
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('mm/day')

    # Net lateral flow
    ax = axes[1, 2]
    net_lateral = ds_t['lateral_in'] - ds_t['lateral_out']
    vmax = max(abs(float(net_lateral.min())), abs(float(net_lateral.max())))
    im = ax.pcolormesh(net_lateral.x, net_lateral.y, net_lateral.values,
                       cmap='RdBu', shading='auto', vmin=-vmax, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_title('Net Lateral Flow')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('mm/day')

    fig.suptitle(f'Water Balance Fluxes - {time_label}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_moisture_vs_indices(
    ds: xr.Dataset,
    query: 'WaterBalanceQuery',
    time_idx: int | str | None = None,
    figsize: tuple[float, float] = (18, 10),
) -> Figure:
    """Plot soil moisture alongside NDVI, NDWI, and SMIPS from Sentinel-2.

    Args:
        ds: Water balance model output dataset.
        query: WaterBalanceQuery used to run the model (for loading Sentinel-2 data).
        time_idx: Time index or date string. If None, uses mid-point.
        figsize: Figure size.

    Returns:
        Matplotlib Figure with soil moisture, NDVI, NDWI, and SMIPS comparison.
    """
    import sys
    sys.path.insert(0, '/borevitz_projects/repos/paddock-ts-local')

    from PaddockTS.query import Query as PaddockQuery
    from PaddockTS.Sentinel2.download_sentinel2 import download_sentinel2
    from WaterBalanceModel.DataAccess.smips import load_smips

    # Select timestep
    if time_idx is None:
        time_idx = len(ds.time) // 2

    if isinstance(time_idx, str):
        ds_t = ds.sel(time=time_idx, method='nearest')
        target_time = ds_t.time.values
        time_label = time_idx
    else:
        ds_t = ds.isel(time=time_idx)
        target_time = ds_t.time.values
        time_label = str(target_time)[:10]

    # Load Sentinel-2 data via PaddockTS
    print(f'Loading Sentinel-2 data for {time_label}...')
    pquery = PaddockQuery(
        bbox=query.bbox,
        start=query.start,
        end=query.end,
        stub=query.stub,
    )
    s2_ds = download_sentinel2(pquery)

    # Load SMIPS data
    print('Loading SMIPS data...')
    smips = load_smips(query)

    # Select nearest Sentinel-2 observation to target time
    s2_t = s2_ds.sel(time=target_time, method='nearest')
    s2_time = str(s2_t.time.values)[:10]

    # Select nearest SMIPS observation to target time
    smips_t = smips.sel(time=target_time, method='nearest')
    smips_time = str(smips_t.time.values)[:10]

    # Compute NDVI: (NIR - Red) / (NIR + Red)
    nir = s2_t['nbart_nir_1'].values.astype(np.float32)
    red = s2_t['nbart_red'].values.astype(np.float32)
    nir[nir == 0] = np.nan
    red[red == 0] = np.nan
    ndvi = (nir - red) / (nir + red)

    # Compute NDWI: (Green - NIR) / (Green + NIR)
    green = s2_t['nbart_green'].values.astype(np.float32)
    green[green == 0] = np.nan
    ndwi = (green - nir) / (green + nir)

    # Create figure with 4 columns
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, height_ratios=[1, 1], hspace=0.25, wspace=0.3)

    # Row 1: Spatial maps
    # Soil Moisture (model output - high res)
    ax1 = fig.add_subplot(gs[0, 0])
    sm = ds_t['soil_moisture']
    im1 = ax1.pcolormesh(sm.x, sm.y, sm.values, cmap='Blues', shading='auto')
    ax1.set_aspect('equal')
    ax1.set_title(f'Model Soil Moisture (10m)\n{time_label}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('mm')

    # SMIPS (coarse resolution ~1km)
    ax2 = fig.add_subplot(gs[0, 1])
    # SMIPS coords are in lat/lon, need to handle appropriately
    if hasattr(smips_t, 'x') and hasattr(smips_t, 'y'):
        im2 = ax2.pcolormesh(smips_t.x, smips_t.y, smips_t.values, cmap='Blues', shading='auto')
    elif hasattr(smips_t, 'longitude') and hasattr(smips_t, 'latitude'):
        im2 = ax2.pcolormesh(smips_t.longitude, smips_t.latitude, smips_t.values, cmap='Blues', shading='auto')
    else:
        # Fallback - try to plot with whatever coords exist
        im2 = ax2.imshow(smips_t.values, cmap='Blues', aspect='auto')
    ax2.set_aspect('equal')
    ax2.set_title(f'SMIPS (~1km)\n{smips_time}')
    ax2.set_xlabel('lon')
    ax2.set_ylabel('lat')
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('mm')

    # NDVI
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.pcolormesh(s2_t.x, s2_t.y, ndvi, cmap='YlGn', shading='auto', vmin=-0.2, vmax=0.9)
    ax3.set_aspect('equal')
    ax3.set_title(f'NDVI (Sentinel-2)\n{s2_time}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('NDVI')

    # NDWI
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.pcolormesh(s2_t.x, s2_t.y, ndwi, cmap='RdYlBu', shading='auto', vmin=-0.5, vmax=0.5)
    ax4.set_aspect('equal')
    ax4.set_title(f'NDWI (Sentinel-2)\n{s2_time}')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    cbar4 = fig.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label('NDWI')

    # Row 2: Time series at domain center
    cx = float(ds.x.values[len(ds.x) // 2])
    cy = float(ds.y.values[len(ds.y) // 2])

    # Model soil moisture time series
    ax5 = fig.add_subplot(gs[1, 0])
    sm_ts = ds['soil_moisture'].sel(x=cx, y=cy, method='nearest')
    ax5.plot(sm_ts.time.values, sm_ts.values, 'b-', linewidth=1.5, label='Model')
    ax5.axvline(target_time, color='red', linestyle='--', alpha=0.7)
    ax5.set_ylabel('Soil Moisture (mm)')
    ax5.set_xlabel('Date')
    ax5.set_title(f'Model Time Series')
    ax5.grid(True, alpha=0.3)

    # SMIPS time series (domain mean since it's coarse)
    ax6 = fig.add_subplot(gs[1, 1])
    smips_ts = smips.mean(dim=['x', 'y']) if 'x' in smips.dims else smips.mean(dim=['longitude', 'latitude'])
    ax6.plot(smips.time.values, smips_ts.values, 'co-', markersize=4, linewidth=1, alpha=0.7, label='SMIPS')
    ax6.axvline(target_time, color='red', linestyle='--', alpha=0.7)
    ax6.set_ylabel('Soil Moisture (mm)')
    ax6.set_xlabel('Date')
    ax6.set_title('SMIPS Time Series (domain mean)')
    ax6.grid(True, alpha=0.3)

    # NDVI time series from all Sentinel-2 observations
    ax7 = fig.add_subplot(gs[1, 2])
    # Compute NDVI for all times
    nir_all = s2_ds['nbart_nir_1'].values.astype(np.float32)
    red_all = s2_ds['nbart_red'].values.astype(np.float32)
    nir_all[nir_all == 0] = np.nan
    red_all[red_all == 0] = np.nan
    ndvi_all = (nir_all - red_all) / (nir_all + red_all)

    # Find pixel closest to center
    s2_cx_idx = np.argmin(np.abs(s2_ds.x.values - cx))
    s2_cy_idx = np.argmin(np.abs(s2_ds.y.values - cy))
    ndvi_ts = ndvi_all[:, s2_cy_idx, s2_cx_idx]

    ax7.plot(s2_ds.time.values, ndvi_ts, 'go-', markersize=4, linewidth=1, alpha=0.7)
    ax7.axvline(target_time, color='red', linestyle='--', alpha=0.7)
    ax7.set_ylabel('NDVI')
    ax7.set_xlabel('Date')
    ax7.set_title('NDVI Time Series')
    ax7.set_ylim(-0.2, 1.0)
    ax7.grid(True, alpha=0.3)

    # NDWI time series
    ax8 = fig.add_subplot(gs[1, 3])
    green_all = s2_ds['nbart_green'].values.astype(np.float32)
    green_all[green_all == 0] = np.nan
    ndwi_all = (green_all - nir_all) / (green_all + nir_all)
    ndwi_ts = ndwi_all[:, s2_cy_idx, s2_cx_idx]

    ax8.plot(s2_ds.time.values, ndwi_ts, 'mo-', markersize=4, linewidth=1, alpha=0.7)
    ax8.axvline(target_time, color='red', linestyle='--', alpha=0.7)
    ax8.set_ylabel('NDWI')
    ax8.set_xlabel('Date')
    ax8.set_title('NDWI Time Series')
    ax8.set_ylim(-0.6, 0.6)
    ax8.grid(True, alpha=0.3)

    fig.suptitle('Soil Moisture vs SMIPS & Vegetation Indices', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def load_output(zarr_path: str | Path) -> xr.Dataset:
    """Load water balance model output from Zarr store.

    Args:
        zarr_path: Path to the soil_moisture.zarr output.

    Returns:
        xr.Dataset with model output.
    """
    return xr.open_zarr(zarr_path).load()  # Load into memory to avoid dask scheduler issues


def visualize_run(
    ds: xr.Dataset,
    output_dir: str | Path | None = None,
    show: bool = True,
    query: 'WaterBalanceQuery | None' = None,
) -> dict[str, Figure]:
    """Generate all standard visualizations for a model run.

    Args:
        ds: Water balance model output dataset.
        output_dir: Directory to save plots (optional).
        show: Whether to display plots interactively.
        query: WaterBalanceQuery to load Sentinel-2 data for NDVI/NDWI comparison.

    Returns:
        Dict of figure names to Figure objects.
    """
    figures = {}

    # Pick a mid-run timestep for spatial plots
    mid_idx = len(ds.time) // 2

    print('Creating spatial snapshot...')
    figures['spatial_snapshot'] = plot_spatial_snapshot(ds, time_idx=mid_idx)

    print('Creating time series plot...')
    figures['timeseries'] = plot_timeseries_at_point(ds)

    print('Creating water balance summary...')
    figures['summary'] = plot_water_balance_summary(ds)

    print('Creating flux maps...')
    figures['flux_maps'] = plot_flux_maps(ds, time_idx=mid_idx)

    # Add moisture vs indices plot if query is provided
    if query is not None:
        print('Creating moisture vs NDVI/NDWI comparison...')
        figures['moisture_vs_indices'] = plot_moisture_vs_indices(ds, query, time_idx=mid_idx)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            path = output_dir / f'{name}.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f'Saved: {path}')

    if show:
        plt.show()

    return figures


if __name__ == '__main__':
    import sys

    # Usage: python visualize.py <path_to_soil_moisture.zarr>
    # Or run model first and pass output directly

    if len(sys.argv) > 1:
        zarr_path = sys.argv[1]
        print(f'Loading output from: {zarr_path}')
        ds = load_output(zarr_path)
        visualize_run(ds, output_dir='/tmp/wb_plots')
    else:
        print('Usage: python visualize.py <path_to_soil_moisture.zarr>')
        print('')
        print('Or use in Python:')
        print('  from WaterBalanceModel.visualize import visualize_run, load_output')
        print('  ds = load_output("path/to/output.zarr")')
        print('  visualize_run(ds)')
        print('')
        print('Or after running pipeline:')
        print('  from WaterBalanceModel.run_model import run_pipeline')
        print('  from WaterBalanceModel.visualize import visualize_run')
        print('  result = run_pipeline(query)')
        print('  visualize_run(result)')
