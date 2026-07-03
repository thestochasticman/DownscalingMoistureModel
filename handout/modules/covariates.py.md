# `covariates.py`: terrain predictors (30 m)

<!-- NAV -->
[← SMIPS coarse predictor](smips.py.md) · [Index](../README.md) · [Soil covariates (SLGA) →](slga.py.md)
<!-- /NAV -->

Source: [`../../emt/covariates.py`](../../emt/covariates.py)

Derives the 30 m terrain covariates that supply sub-grid structure to the model,
together with a CRS-aware point sampler.

| Function | Role |
|---|---|
| `terrain_covariates(query)` | 30 m covariate stack (`xr.Dataset`) on a metric UTM grid |
| `sample_points(obj, lon, lat)` | Nearest-pixel sample at one lon/lat, with CRS reprojection |

`TERRAIN_VARS = (elevation, slope, northness, eastness, twi, hli, accumulation)`.
Aspect is decomposed into northness/eastness to provide continuous predictors
without the 0°/360° discontinuity.

## Coordinate-system handling

The PaddockTS terrain utilities compute `np.gradient` using the raster's pixel
spacing directly. The native Copernicus DEM is in EPSG:4326, where spacing is in
degrees (≈0.00028) while elevations are in metres; applied directly this
overstates the gradient by roughly three orders of magnitude and saturates slope
at ≈90° (with TWI and HLI inheriting the error). The DEM is therefore reprojected
to a metric UTM coordinate system before the utilities are applied, which makes
the derivatives dimensionally consistent. The same defect affects the PaddockTS
terrain plots upstream.

The utilities read the raster without a nodata mask, so valid pixels adjacent to
the UTM nodata corners produce spurious gradients. Each derivative is masked to
the DEM's valid footprint, eroded by one pixel.

The covariates are static per station (sampled once and broadcast across days)
and are joined into the training table in [`features.py`](features.py.md).

---
<!-- NAV -->
[← SMIPS coarse predictor](smips.py.md) · [Index](../README.md) · [Soil covariates (SLGA) →](slga.py.md)
<!-- /NAV -->
