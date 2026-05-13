"""Water balance process components.

Modules:
    evapotranspiration: ET = ET0 × Kc × Ks computation
    precipitation: Spatial disaggregation of rainfall
    runoff: SCS Curve Number method
    drainage: Gravity drainage below root zone
    lateral_flow: Terrain-based subsurface routing
"""

from WaterBalanceModel.Components.evapotranspiration import ETComputer, compute_et0_fao56, compute_kc_from_ndvi, compute_ks
from WaterBalanceModel.Components.precipitation import disaggregate_precipitation, compute_effective_precipitation
from WaterBalanceModel.Components.runoff import compute_runoff_scs
from WaterBalanceModel.Components.drainage import compute_gravity_drainage
from WaterBalanceModel.Components.lateral_flow import build_flow_routing_matrix, compute_lateral_flow

__all__ = [
    'ETComputer',
    'compute_et0_fao56',
    'compute_kc_from_ndvi',
    'compute_ks',
    'disaggregate_precipitation',
    'compute_effective_precipitation',
    'compute_runoff_scs',
    'compute_gravity_drainage',
    'build_flow_routing_matrix',
    'compute_lateral_flow',
]
