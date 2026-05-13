"""Water Balance Process Model for Soil Moisture Downscaling.

This package implements a water balance model to downscale coarse SMIPS
soil moisture (~1km) to Sentinel-2 resolution (10m) using process-based
physics rather than pure optimization.

Core equation: dθ/dt = P - ET - R - D + L_in - L_out

Where:
    θ: Soil moisture (mm)
    P: Precipitation (mm/day)
    ET: Evapotranspiration (mm/day)
    R: Runoff (mm/day)
    D: Drainage (mm/day)
    L_in/L_out: Lateral flow (mm/day)
"""

from WaterBalanceModel.config import config, Config
from WaterBalanceModel.query import WaterBalanceQuery

__all__ = ['config', 'Config', 'WaterBalanceQuery']
__version__ = '0.1.0'
