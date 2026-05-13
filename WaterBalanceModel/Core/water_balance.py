"""Main water balance model implementation.

This module contains the WaterBalanceModel class that integrates all
components to simulate soil moisture dynamics at high spatial resolution.

Core equation: dθ/dt = P - ET - R - D + L_in - L_out
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from attrs import frozen

from WaterBalanceModel.Components.drainage import compute_gravity_drainage
from WaterBalanceModel.Components.evapotranspiration import ETComputer
from WaterBalanceModel.Components.lateral_flow import build_flow_routing_matrix, compute_lateral_flow
from WaterBalanceModel.Components.precipitation import compute_effective_precipitation, disaggregate_precipitation
from WaterBalanceModel.Components.runoff import compute_runoff_scs
from WaterBalanceModel.Core.spatial_grid import resample_to_target
from WaterBalanceModel.Core.water_balance_config import WaterBalanceConfig

if TYPE_CHECKING:
    from WaterBalanceModel.query import WaterBalanceQuery


@frozen
class WaterBalanceModel:
    """Water balance process model for soil moisture simulation.

    Implements the water balance equation:
    dθ/dt = P - ET - R - D + L_in - L_out

    The model runs at daily timestep on a high-resolution spatial grid
    (typically 10m from Sentinel-2), using terrain-derived parameters
    for lateral flow and SLGA-derived soil hydraulic properties.

    Attributes:
        config: Model configuration parameters.
    """

    config: WaterBalanceConfig = WaterBalanceConfig()

    def run(
        self,
        query: 'WaterBalanceQuery',
        climate: pd.DataFrame,
        terrain: xr.Dataset,
        ndvi: xr.DataArray,
        soil_params: xr.Dataset,
        smips: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """Run the water balance model.

        Args:
            query: Water balance query specifying domain and period.
            climate: Daily climate data with columns: date, precip, tmin,
                tmax, radiation, vp, wind, et0.
            terrain: Terrain dataset with dem, slope, aspect, twi, flow_dir.
            ndvi: NDVI time series at model resolution.
            soil_params: Soil hydraulic parameters (theta_sat, theta_fc,
                theta_wp, k_sat) at model resolution.
            smips: Optional SMIPS soil moisture for constraint.

        Returns:
            xr.Dataset with soil moisture and flux components at each timestep.
        """
        print('  initializing water balance model...')

        # Resample soil parameters to terrain grid if needed
        theta_sat = resample_to_target(soil_params['theta_sat'], terrain['dem'])
        theta_fc = resample_to_target(soil_params['theta_fc'], terrain['dem'])
        theta_wp = resample_to_target(soil_params['theta_wp'], terrain['dem'])
        k_sat = resample_to_target(soil_params['k_sat'], terrain['dem'])

        # Convert volumetric fractions to mm
        soil_depth = self.config.soil.soil_depth
        theta_sat_mm = theta_sat * soil_depth
        theta_fc_mm = theta_fc * soil_depth
        theta_wp_mm = theta_wp * soil_depth

        # Initialize soil moisture
        theta = theta_fc_mm * self.config.soil.initial_moisture_fraction
        theta = theta.copy()  # Make writable

        # Build flow routing matrix for lateral flow
        routing_matrix = None
        if self.config.lateral.enable:
            print('  building flow routing matrix...')
            routing_matrix = build_flow_routing_matrix(
                terrain['flow_dir'],
                terrain['slope'],
            )

        # Initialize ET computer
        et_computer = ETComputer(params=self.config.et)

        # Get cell width for lateral flow
        cell_width = abs(float(terrain.x[1] - terrain.x[0])) if len(terrain.x) > 1 else 10.0

        # Prepare output arrays
        dates = pd.date_range(query.start, query.end, freq='D')
        n_times = len(dates)
        ny, nx = terrain['dem'].shape

        output_vars = {
            'soil_moisture': np.zeros((n_times, ny, nx), dtype=np.float32),
            'et_actual': np.zeros((n_times, ny, nx), dtype=np.float32),
            'runoff': np.zeros((n_times, ny, nx), dtype=np.float32),
            'drainage': np.zeros((n_times, ny, nx), dtype=np.float32),
            'lateral_in': np.zeros((n_times, ny, nx), dtype=np.float32),
            'lateral_out': np.zeros((n_times, ny, nx), dtype=np.float32),
        }

        # Time stepping loop
        print(f'  running water balance for {n_times} days...')
        for t, date in enumerate(dates):
            if t % 30 == 0:
                print(f'    day {t+1}/{n_times} ({date.strftime("%Y-%m-%d")})')

            # Get climate for this day
            climate_row = climate[climate['date'] == date]
            if len(climate_row) == 0:
                continue
            climate_day = climate_row.iloc[0].to_dict()

            # Get NDVI for this day
            if 'time' in ndvi.dims:
                ndvi_day = ndvi.sel(time=date, method='nearest')
            else:
                ndvi_day = ndvi

            # Resample NDVI to terrain grid
            ndvi_day = resample_to_target(ndvi_day, terrain['dem'])

            # Run one timestep
            result = self.step(
                theta=theta,
                climate=climate_day,
                ndvi=ndvi_day,
                terrain=terrain,
                theta_sat_mm=theta_sat_mm,
                theta_fc_mm=theta_fc_mm,
                theta_wp_mm=theta_wp_mm,
                k_sat=k_sat,
                routing_matrix=routing_matrix,
                cell_width=cell_width,
                doy=date.timetuple().tm_yday,
                et_computer=et_computer,
            )

            # Update state
            theta = result['theta_new']

            # Store outputs
            output_vars['soil_moisture'][t] = theta.values
            output_vars['et_actual'][t] = result['et'].values
            output_vars['runoff'][t] = result['runoff'].values
            output_vars['drainage'][t] = result['drainage'].values
            output_vars['lateral_in'][t] = result['lateral_in'].values
            output_vars['lateral_out'][t] = result['lateral_out'].values

        # Build output dataset
        print('  building output dataset...')
        ds = xr.Dataset(
            data_vars={
                'soil_moisture': (['time', 'y', 'x'], output_vars['soil_moisture'], {
                    'units': 'mm',
                    'long_name': 'Soil moisture content',
                }),
                'et_actual': (['time', 'y', 'x'], output_vars['et_actual'], {
                    'units': 'mm/day',
                    'long_name': 'Actual evapotranspiration',
                }),
                'runoff': (['time', 'y', 'x'], output_vars['runoff'], {
                    'units': 'mm/day',
                    'long_name': 'Surface runoff',
                }),
                'drainage': (['time', 'y', 'x'], output_vars['drainage'], {
                    'units': 'mm/day',
                    'long_name': 'Deep drainage',
                }),
                'lateral_in': (['time', 'y', 'x'], output_vars['lateral_in'], {
                    'units': 'mm/day',
                    'long_name': 'Lateral subsurface inflow',
                }),
                'lateral_out': (['time', 'y', 'x'], output_vars['lateral_out'], {
                    'units': 'mm/day',
                    'long_name': 'Lateral subsurface outflow',
                }),
            },
            coords={
                'time': dates,
                'y': terrain.y,
                'x': terrain.x,
            },
            attrs={
                'source': 'WaterBalanceModel',
                'created': datetime.now().isoformat(),
                'bbox': query.bbox,
                'config': str(self.config.to_dict()),
            },
        )

        return ds

    def step(
        self,
        theta: xr.DataArray,
        climate: dict,
        ndvi: xr.DataArray,
        terrain: xr.Dataset,
        theta_sat_mm: xr.DataArray,
        theta_fc_mm: xr.DataArray,
        theta_wp_mm: xr.DataArray,
        k_sat: xr.DataArray,
        routing_matrix,
        cell_width: float,
        doy: int,
        et_computer: ETComputer,
        dt: float = 1.0,
    ) -> dict[str, xr.DataArray]:
        """Execute one timestep of the water balance.

        Args:
            theta: Current soil moisture (mm).
            climate: Dict with precip, tmin, tmax, radiation, vp, wind, et0.
            ndvi: NDVI for this timestep.
            terrain: Terrain dataset.
            theta_sat_mm: Saturated moisture (mm).
            theta_fc_mm: Field capacity (mm).
            theta_wp_mm: Wilting point (mm).
            k_sat: Saturated hydraulic conductivity (mm/day).
            routing_matrix: Flow routing matrix.
            cell_width: Grid cell width (m).
            doy: Day of year.
            et_computer: ET computation engine.
            dt: Timestep (days).

        Returns:
            Dict with theta_new, et, runoff, drainage, lateral_in, lateral_out.
        """
        # 1. Precipitation
        precip = disaggregate_precipitation(climate['precip'], terrain)
        precip_eff = compute_effective_precipitation(precip, ndvi)

        # 2. Evapotranspiration
        # Use SILO ET0 directly and adjust spatially
        et0 = et_computer.compute_et0(climate, terrain, doy)
        kc = et_computer.compute_kc(ndvi)
        ks = et_computer.compute_ks(theta, theta_fc_mm, theta_wp_mm)
        et = et_computer.compute_et(et0, kc, ks)

        # 3. Runoff
        runoff = compute_runoff_scs(
            precip_eff, theta, theta_fc_mm, theta_wp_mm,
            cn_normal=self.config.runoff.cn_normal,
            params=self.config.runoff,
        )

        # Infiltration
        infiltration = precip_eff - runoff

        # 4. Drainage
        drainage = compute_gravity_drainage(
            theta, theta_fc_mm, theta_sat_mm, k_sat,
        )

        # 5. Lateral flow
        if self.config.lateral.enable and routing_matrix is not None:
            lateral_in, lateral_out = compute_lateral_flow(
                theta, theta_fc_mm, theta_sat_mm,
                terrain, k_sat, routing_matrix, cell_width,
            )
        else:
            lateral_in = xr.zeros_like(theta)
            lateral_out = xr.zeros_like(theta)

        # 6. Water balance update
        theta_new = theta + dt * (
            infiltration
            - et
            - drainage
            + lateral_in
            - lateral_out
        )

        # Apply constraints
        theta_new = theta_new.clip(min=theta_wp_mm * 0.5, max=theta_sat_mm)

        return {
            'theta_new': theta_new,
            'et': et,
            'runoff': runoff,
            'drainage': drainage,
            'lateral_in': lateral_in,
            'lateral_out': lateral_out,
        }
