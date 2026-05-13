"""Main pipeline orchestration for the water balance model.

This module provides the main entry point for running the complete
water balance pipeline, following the PaddockTS pattern with rich
progress display.
"""

from __future__ import annotations

import time
from datetime import datetime
from os.path import exists

import xarray as xr
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from WaterBalanceModel.Calibration.smips_constraint import apply_smips_constraint_timeseries
from WaterBalanceModel.Core.water_balance import WaterBalanceModel
from WaterBalanceModel.Core.water_balance_config import WaterBalanceConfig
from WaterBalanceModel.DataAccess.climate import load_climate
from WaterBalanceModel.DataAccess.sentinel2 import load_ndvi
from WaterBalanceModel.DataAccess.smips import load_smips
from WaterBalanceModel.DataAccess.soils import load_soil_parameters
from WaterBalanceModel.DataAccess.terrain import load_terrain
from WaterBalanceModel.query import WaterBalanceQuery
from WaterBalanceModel.visualize import visualize_run

STEPS = [
    'Load terrain data',
    'Load climate data',
    'Load NDVI from Sentinel-2',
    'Load soil parameters (SLGA)',
    'Load SMIPS (calibration)',
    'Run water balance model',
    'Apply SMIPS constraint',
    'Save output to Zarr',
    'Visualize results',
]


def _make_table(statuses: list[str], times: list[float | None]) -> Table:
    """Create a status table for display."""
    table = Table(title='Water Balance Model Pipeline')
    table.add_column('Step', style='cyan')
    table.add_column('Status', style='bold')
    table.add_column('Time', style='dim')

    status_styles = {
        'pending': '[dim]pending[/dim]',
        'running': '[yellow]running[/yellow]',
        'done': '[green]done[/green]',
        'error': '[red]error[/red]',
    }

    for i, step in enumerate(STEPS):
        status = status_styles.get(statuses[i], statuses[i])
        time_str = f'{times[i]:.1f}s' if times[i] is not None else ''
        table.add_row(step, status, time_str)

    return table


def run_pipeline(
    query: WaterBalanceQuery,
    config: WaterBalanceConfig = WaterBalanceConfig(),
    reload: bool = False,
    visualize: bool = True,
) -> xr.Dataset:
    """Run the complete water balance pipeline.

    This is the main entry point for the water balance model. It:
    1. Loads all required data (terrain, climate, NDVI, soil, SMIPS)
    2. Runs the water balance model
    3. Applies SMIPS constraint if configured
    4. Saves output to Zarr
    5. Generates visualization plots

    Args:
        query: Water balance query specifying domain and period.
        config: Model configuration.
        reload: If True, delete cached data and reprocess.
        visualize: If True, generate and save visualization plots.

    Returns:
        Output dataset with soil moisture and flux components.
    """
    console = Console(stderr=True)

    if reload:
        import shutil
        if exists(query.tmp_dir):
            shutil.rmtree(query.tmp_dir)
        if exists(query.out_dir):
            shutil.rmtree(query.out_dir)

    # Check for cached output
    if exists(query.soil_moisture_path) and not reload:
        console.print(f'[green]Cached output found:[/green] {query.soil_moisture_path}')
        result = xr.open_zarr(query.soil_moisture_path).load()  # Load into memory for plotting
        if visualize:
            plot_dir = f'{query.out_dir}/plots'
            console.print(f'Generating plots in {plot_dir}...')
            visualize_run(result, output_dir=plot_dir, show=False, query=query)
            console.print(f'[green]Plots saved![/green]')
        return result

    statuses = ['pending'] * len(STEPS)
    times = [None] * len(STEPS)

    progress = Progress(
        TextColumn('[bold blue]{task.description}'),
        BarColumn(),
        TextColumn('{task.completed}/{task.total}'),
        TimeElapsedColumn(),
        console=console,
    )
    task_id = progress.add_task('Pipeline', total=len(STEPS))

    # Variables to hold loaded data
    terrain = None
    climate = None
    ndvi = None
    soil_params = None
    smips = None
    result = None

    with Live(_make_table(statuses, times), console=console, refresh_per_second=4) as live:
        for i in range(len(STEPS)):
            statuses[i] = 'running'
            live.update(_make_table(statuses, times))

            t0 = time.time()
            try:
                if i == 0:
                    # Load terrain
                    terrain = load_terrain(query)

                elif i == 1:
                    # Load climate
                    climate = load_climate(query)

                elif i == 2:
                    # Load NDVI
                    ndvi = load_ndvi(query)

                elif i == 3:
                    # Load soil parameters
                    soil_params = load_soil_parameters(query)

                elif i == 4:
                    # Load SMIPS
                    if config.calibration.use_smips_constraint:
                        smips = load_smips(query)
                    else:
                        console.print('  [dim]skipping SMIPS (constraint disabled)[/dim]')

                elif i == 5:
                    # Run water balance model
                    model = WaterBalanceModel(config=config)
                    result = model.run(
                        query=query,
                        climate=climate,
                        terrain=terrain,
                        ndvi=ndvi,
                        soil_params=soil_params,
                        smips=smips,
                    )

                elif i == 6:
                    # Apply SMIPS constraint
                    if config.calibration.use_smips_constraint and smips is not None:
                        console.print('  applying SMIPS constraint...')
                        result['soil_moisture'] = apply_smips_constraint_timeseries(
                            result['soil_moisture'],
                            smips,
                            lambda_smoothness=config.calibration.lambda_smoothness,
                            max_gap_days=config.calibration.max_gap_days,
                            solver=config.calibration.solver,
                        )
                    else:
                        console.print('  [dim]skipping SMIPS constraint[/dim]')

                elif i == 7:
                    # Save output
                    console.print(f'  saving to {query.soil_moisture_path}...')
                    result.to_zarr(query.soil_moisture_path, mode='w')
                    console.print(f'  [green]saved![/green]')

                elif i == 8:
                    # Visualize results
                    if visualize:
                        plot_dir = f'{query.out_dir}/plots'
                        console.print(f'  generating plots in {plot_dir}...')
                        visualize_run(result, output_dir=plot_dir, show=False, query=query)
                        console.print(f'  [green]plots saved![/green]')
                    else:
                        console.print('  [dim]skipping visualization[/dim]')

                statuses[i] = 'done'

            except Exception as e:
                statuses[i] = 'error'
                times[i] = time.time() - t0
                live.update(_make_table(statuses, times))
                console.print(f'[red]Error in step {i+1}:[/red] {e}')
                raise

            times[i] = time.time() - t0
            progress.update(task_id, completed=i + 1)
            live.update(_make_table(statuses, times))

    total_time = sum(t for t in times if t is not None)
    console.print(f'\n[green]Pipeline complete![/green] Total time: {total_time:.1f}s')
    console.print(f'Output: {query.soil_moisture_path}')
    if visualize:
        console.print(f'Plots:  {query.out_dir}/plots/')

    return result


def run_water_balance(
    bbox: list[float],
    start: str,
    end: str,
    stub: str | None = None,
    config: WaterBalanceConfig | None = None,
    visualize: bool = True,
) -> xr.Dataset:
    """Convenience function to run water balance model.

    Args:
        bbox: Bounding box [west, south, east, north] in EPSG:4326.
        start: Start date as 'YYYY-MM-DD'.
        end: End date as 'YYYY-MM-DD'.
        stub: Optional identifier for caching.
        config: Optional model configuration.
        visualize: If True, generate and save visualization plots.

    Returns:
        Output dataset.

    Example:
        >>> result = run_water_balance(
        ...     bbox=[148.36, -33.52, 148.38, -33.50],
        ...     start='2020-01-01',
        ...     end='2020-12-31',
        ... )
    """
    from datetime import date

    query = WaterBalanceQuery(
        bbox=bbox,
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        stub=stub,
    ) if stub else WaterBalanceQuery(
        bbox=bbox,
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
    )

    config = config or WaterBalanceConfig()

    return run_pipeline(query, config, visualize=visualize)


if __name__ == '__main__':
    # Example usage
    from datetime import date

    query = WaterBalanceQuery.from_lat_lon(
        lat=-33.51,
        lon=148.37,
        buffer_km=1.0,
        start=date(2020, 1, 1),
        end=date(2020, 3, 31),
        stub='test_run',
    )

    result = run_pipeline(query)
    print(result)
