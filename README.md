# Water Balance Model for Soil Moisture Downscaling

A process-based water balance model that downscales SMIPS (~1km) soil moisture to Sentinel-2 resolution (10m) using terrain, vegetation, and soil properties.

## Overview

This model implements a daily water balance equation to simulate soil moisture dynamics at high spatial resolution:

```
dθ/dt = P - ET - R - D + L_in - L_out
```

Where:
- **θ**: Soil moisture content (mm)
- **P**: Precipitation (mm/day)
- **ET**: Evapotranspiration (mm/day)
- **R**: Surface runoff (mm/day)
- **D**: Deep drainage (mm/day)
- **L_in / L_out**: Lateral subsurface flow (mm/day)

The model output is constrained to match SMIPS observations using convex optimization, ensuring mass conservation at the coarse scale while preserving high-resolution spatial patterns.

## Installation

```bash
pip install -e .
```

For PaddockTS data integration:
```bash
pip install -e ".[paddockts]"
```

### Dependencies

- Python >= 3.11
- numpy, xarray, pandas
- cvxpy (for SMIPS constraint optimization)
- scipy, matplotlib
- PaddockTS (for data access)

## Quick Start

```python
from datetime import date
from WaterBalanceModel.query import WaterBalanceQuery
from WaterBalanceModel.run_model import run_pipeline

# Define query for a region
query = WaterBalanceQuery.from_lat_lon(
    lat=-33.51,           # Centre latitude
    lon=148.37,           # Centre longitude
    buffer_km=1.0,        # 2km x 2km area
    start=date(2023, 1, 1),
    end=date(2023, 3, 31),
    stub='my_run',        # Optional identifier for caching
)

# Run the model
result = run_pipeline(query)

# Access outputs
soil_moisture = result['soil_moisture']  # (time, y, x) in mm
et = result['et_actual']                 # mm/day
runoff = result['runoff']                # mm/day
```

Alternatively, use the convenience function:

```python
from WaterBalanceModel.run_model import run_water_balance

result = run_water_balance(
    bbox=[148.36, -33.52, 148.38, -33.50],
    start='2023-01-01',
    end='2023-03-31',
)
```

## Data Sources

The model integrates data from multiple sources via PaddockTS:

| Data | Source | Resolution | Description |
|------|--------|------------|-------------|
| Terrain | DEM-S | 1 arcsec (~30m) | Elevation, slope, aspect, flow direction |
| Climate | SILO | 5km | Precipitation, temperature, radiation, ET0 |
| Vegetation | Sentinel-2 | 10m | NDVI for crop coefficient |
| Soil | SLGA | 90m | Hydraulic properties via pedotransfer |
| Reference | SMIPS | ~1km | Soil moisture for constraint |

## Model Components

### Evapotranspiration

Implements FAO-56 Penman-Monteith with spatial adjustment:

```
ET = ET0 × Kc × Ks
```

- **ET0**: Reference ET from SILO, adjusted for terrain aspect
- **Kc**: Crop coefficient derived from NDVI (linear scaling)
- **Ks**: Soil water stress coefficient (linear reduction below threshold)

### Runoff

SCS Curve Number method with antecedent moisture adjustment:

```
Q = (P - Ia)² / (P - Ia + S)
```

Where S (potential retention) is adjusted based on soil moisture state.

### Drainage

Gravity drainage when soil moisture exceeds field capacity:

```
D = Ksat × ((θ - θfc) / (θsat - θfc))^β
```

### Lateral Flow

Terrain-driven subsurface flow using D8 flow routing:
- Flow proportional to hydraulic gradient (slope)
- Routed using flow direction derived from DEM

### SMIPS Constraint

Convex optimization to match coarse-scale SMIPS observations:

```
minimize   ||θ - θ_model||² + λ||∇²θ||²
subject to A·θ = smips_obs    (mass conservation)
           θ ≥ 0              (non-negativity)
```

Where A is the aggregation matrix from fine (10m) to coarse (~1km) resolution.

## Configuration

The model uses immutable configuration objects:

```python
from WaterBalanceModel.Core.water_balance_config import (
    WaterBalanceConfig,
    SoilConfig,
    ETParameters,
    CalibrationParameters,
)

config = WaterBalanceConfig(
    soil=SoilConfig(
        soil_depth=1000.0,              # Root zone depth (mm)
        initial_moisture_fraction=0.5,   # Initial state as fraction of FC
    ),
    et=ETParameters(
        kc_min=0.15,    # Bare soil Kc
        kc_max=1.20,    # Full vegetation Kc
    ),
    calibration=CalibrationParameters(
        use_smips_constraint=True,
        lambda_smoothness=0.5,          # Spatial regularization
        solver='SCS',                   # CVXPY solver
    ),
)

result = run_pipeline(query, config=config)
```

### Configuration Parameters

#### Soil
| Parameter | Default | Description |
|-----------|---------|-------------|
| `soil_depth` | 1000.0 | Effective root zone depth (mm) |
| `initial_moisture_fraction` | 0.5 | Initial moisture as fraction of field capacity |

#### Evapotranspiration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `kc_min` | 0.15 | Minimum crop coefficient |
| `kc_max` | 1.20 | Maximum crop coefficient |
| `ndvi_min` | 0.10 | NDVI threshold for bare soil |
| `ndvi_max` | 0.80 | NDVI threshold for full vegetation |
| `p_factor` | 0.50 | Depletion factor for stress |

#### Calibration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_smips_constraint` | True | Enable SMIPS mass conservation |
| `lambda_smoothness` | 0.5 | Spatial smoothness weight |
| `solver` | 'SCS' | CVXPY solver (SCS, OSQP, ECOS) |

## Output

The model outputs an xarray Dataset with the following variables:

| Variable | Units | Description |
|----------|-------|-------------|
| `soil_moisture` | mm | Soil moisture content |
| `et_actual` | mm/day | Actual evapotranspiration |
| `runoff` | mm/day | Surface runoff |
| `drainage` | mm/day | Deep drainage |
| `lateral_in` | mm/day | Lateral subsurface inflow |
| `lateral_out` | mm/day | Lateral subsurface outflow |

Outputs are saved to Zarr format at `{out_dir}/{stub}_soil_moisture.zarr`.

## Visualization

The model automatically generates visualization plots:

```python
from WaterBalanceModel.visualize import visualize_run, plot_moisture_vs_indices

# Generate all standard plots
visualize_run(result, output_dir='plots/', query=query)

# Compare with Sentinel-2 indices
plot_moisture_vs_indices(result, query, output_dir='plots/')
```

Available plots:
- Spatial snapshots of soil moisture
- Time series at selected points
- Water balance summary (cumulative fluxes)
- Flux component maps
- Comparison with NDVI, NDWI, and SMIPS

## Project Structure

```
WaterBalanceModel/
├── Core/
│   ├── water_balance.py       # Main model implementation
│   ├── water_balance_config.py # Configuration classes
│   └── spatial_grid.py        # Grid resampling utilities
├── Components/
│   ├── evapotranspiration.py  # FAO-56 ET calculation
│   ├── precipitation.py       # Precipitation disaggregation
│   ├── runoff.py              # SCS curve number runoff
│   ├── drainage.py            # Gravity drainage
│   └── lateral_flow.py        # D8 lateral flow routing
├── Calibration/
│   └── smips_constraint.py    # CVXPY optimization
├── DataAccess/
│   ├── terrain.py             # DEM and terrain attributes
│   ├── climate.py             # SILO climate data
│   ├── sentinel2.py           # Sentinel-2 NDVI
│   ├── smips.py               # SMIPS soil moisture
│   └── soils.py               # SLGA soil properties
├── query.py                   # Query specification
├── run_model.py               # Pipeline orchestration
├── visualize.py               # Plotting functions
└── config.py                  # Global configuration
```

## Limitations

- **Initial state**: Set to 50% of field capacity (not from observations)
- **Single layer**: Bucket model, no vertical soil profile
- **Daily timestep**: May miss sub-daily dynamics
- **Static soil properties**: No temporal variation in Ksat, etc.

## References

- Allen, R.G. et al. (1998). FAO Irrigation and Drainage Paper No. 56: Crop Evapotranspiration
- USDA-SCS (1972). National Engineering Handbook, Section 4: Hydrology
- Richter, H. et al. (2004). SMIPS: Soil Moisture from Integrated Passive Systems

## License

MIT
