"""Data access layer wrapping PaddockTS modules.

Modules:
    climate: SILO and OzWALD climate data
    terrain: DEM, slope, TWI, HLI, flow routing
    sentinel2: NDVI time series from Sentinel-2
    smips: SMIPS soil moisture for calibration
    soils: SLGA soil parameters with pedotransfer functions
"""

from WaterBalanceModel.DataAccess.climate import load_climate
from WaterBalanceModel.DataAccess.terrain import load_terrain
from WaterBalanceModel.DataAccess.sentinel2 import load_ndvi
from WaterBalanceModel.DataAccess.smips import load_smips
from WaterBalanceModel.DataAccess.soils import load_soil_parameters

__all__ = [
    'load_climate',
    'load_terrain',
    'load_ndvi',
    'load_smips',
    'load_soil_parameters',
]
