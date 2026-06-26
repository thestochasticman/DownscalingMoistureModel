# `covariates.py` — fine terrain predictors (30 m)

Source: [`../../emt/covariates.py`](../../emt/covariates.py)

Builds the **fine-resolution predictors** that carry the sub-grid structure the
model uses to sharpen SMIPS: 30 m Copernicus-DEM terrain derivatives, plus a
CRS-aware point sampler.

| function | role |
|---|---|
| `terrain_covariates(query)` | 30 m covariate stack as an `xr.Dataset` on a metric UTM grid |
| `sample_points(obj, lon, lat)` | nearest-pixel sample at one lon/lat (handles CRS reprojection) |

`TERRAIN_VARS = (elevation, slope, northness, eastness, twi, hli, accumulation)`.
Aspect is split into **northness/eastness** (continuous, no 0°/360° wrap) so it
is usable by the regressor.

## Key decision — reproject the DEM to UTM first ⚠️
PaddockTS's terrain utils compute `np.gradient` using the raster's pixel spacing
as-is. The native Copernicus DEM is EPSG:4326: spacing is in **degrees**
(~0.00028) while elevations are in **metres**, so the gradient comes out ~1000×
too large and **slope saturates at ~90° everywhere** (TWI/HLI inherit the
error). EMT reprojects the DEM to a metric **UTM** CRS first, then calls the
PaddockTS utils unchanged — now dimensionally correct. (This bug also affects
PaddockTS's own terrain plots upstream.)

Also: the utils read the raster unmasked, so valid pixels next to UTM nodata
corners get a spurious cliff. EMT masks every derivative to the DEM's valid
footprint, eroded by one pixel.

These covariates are **static per station** (one sample, broadcast across all
days) and join into the table in [`features.py`](features.py.md).
