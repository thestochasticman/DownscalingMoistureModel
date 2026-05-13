"""Configuration classes for the water balance model.

This module defines frozen attrs classes for all model parameters,
following the PaddockTS pattern of immutable configuration objects.
"""

from __future__ import annotations

from attrs import frozen

from WaterBalanceModel.Components.evapotranspiration import ETParameters
from WaterBalanceModel.Components.runoff import RunoffParameters


@frozen
class SoilConfig:
    """Soil configuration parameters.

    Note: Actual hydraulic parameters (theta_sat, theta_fc, theta_wp, k_sat)
    are loaded spatially from SLGA via pedotransfer functions. These are
    fallback defaults and model structure parameters.

    Attributes:
        soil_depth: Effective soil depth / root zone depth (mm).
        n_layers: Number of soil layers (1 for bucket model).
        initial_moisture_fraction: Initial moisture as fraction of field capacity.
    """

    soil_depth: float = 1000.0
    n_layers: int = 1
    initial_moisture_fraction: float = 0.5


@frozen
class LateralFlowParameters:
    """Parameters for lateral subsurface flow.

    Attributes:
        enable: Toggle lateral flow computation.
        velocity_factor: Scaling factor for lateral flow velocity.
        twi_threshold: TWI threshold for saturation excess (not used currently).
    """

    enable: bool = True
    velocity_factor: float = 1.0
    twi_threshold: float = 10.0


@frozen
class CalibrationParameters:
    """Parameters for SMIPS calibration/constraint.

    Attributes:
        use_smips_constraint: Enable mass conservation with SMIPS.
        lambda_smoothness: Spatial smoothness regularization weight.
        max_gap_days: Maximum temporal gap for SMIPS matching.
        solver: CVXPY solver to use ('SCS', 'OSQP', 'ECOS').
    """

    use_smips_constraint: bool = True
    lambda_smoothness: float = 0.5
    max_gap_days: int = 1
    solver: str = 'SCS'


@frozen
class OutputConfig:
    """Output configuration.

    Attributes:
        variables: Tuple of variable names to include in output.
        save_daily: Save daily outputs (vs. summary statistics only).
        zarr_chunks: Chunk sizes for Zarr output.
    """

    variables: tuple[str, ...] = (
        'soil_moisture',
        'et_actual',
        'runoff',
        'drainage',
        'lateral_in',
        'lateral_out',
    )
    save_daily: bool = True
    zarr_chunks: dict = None

    def __attrs_post_init__(self):
        if self.zarr_chunks is None:
            object.__setattr__(self, 'zarr_chunks', {'time': 30, 'y': 256, 'x': 256})


@frozen
class WaterBalanceConfig:
    """Complete configuration for the water balance model.

    Aggregates all component configurations into a single immutable
    object that can be passed through the pipeline.

    Attributes:
        soil: Soil configuration.
        et: Evapotranspiration parameters.
        runoff: Runoff parameters.
        lateral: Lateral flow parameters.
        calibration: Calibration parameters.
        output: Output configuration.
    """

    soil: SoilConfig = SoilConfig()
    et: ETParameters = ETParameters()
    runoff: RunoffParameters = RunoffParameters()
    lateral: LateralFlowParameters = LateralFlowParameters()
    calibration: CalibrationParameters = CalibrationParameters()
    output: OutputConfig = OutputConfig()

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            'soil': {
                'soil_depth': self.soil.soil_depth,
                'n_layers': self.soil.n_layers,
                'initial_moisture_fraction': self.soil.initial_moisture_fraction,
            },
            'et': {
                'kc_min': self.et.kc_min,
                'kc_max': self.et.kc_max,
                'ndvi_min': self.et.ndvi_min,
                'ndvi_max': self.et.ndvi_max,
                'p_factor': self.et.p_factor,
            },
            'runoff': {
                'cn_dry': self.runoff.cn_dry,
                'cn_normal': self.runoff.cn_normal,
                'cn_wet': self.runoff.cn_wet,
                'ia_ratio': self.runoff.ia_ratio,
            },
            'lateral': {
                'enable': self.lateral.enable,
                'velocity_factor': self.lateral.velocity_factor,
            },
            'calibration': {
                'use_smips_constraint': self.calibration.use_smips_constraint,
                'lambda_smoothness': self.calibration.lambda_smoothness,
            },
        }
