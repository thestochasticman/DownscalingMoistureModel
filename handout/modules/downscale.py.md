# `downscale.py`: 30 m field generation (Stage 6)

Source: [`../../emt/downscale.py`](../../emt/downscale.py)

Applies the fitted model per pixel to produce a 30 m root-zone soil-moisture
field for one day over a focus area.

| Function | Role |
|---|---|
| `downscale(model, query, day)` | `xr.Dataset` on the 30 m grid with `sm_pred` (%) and `smips_native` (mm) |

## Procedure

1. Build the 30 m terrain covariate stack ([`covariates.py`](covariates.py.md))
   and retrieve the coarse SMIPS field for the day ([`smips.py`](smips.py.md)).
2. Resample SMIPS to the 30 m grid by nearest neighbour, so each fine pixel
   inherits its overlying ≈1 km value.
3. Assemble the model features per pixel (in `FEATURES` order), predict over
   pixels with complete covariates, and reshape to the grid; pixels outside the
   DEM footprint are set to NaN.

Sub-grid structure in the output derives entirely from the terrain covariates
varying within each coarse SMIPS cell (see panel (c) of the Stage 6 figure in the
[README](../README.md)).

## Future work (not yet implemented)

- **Mass conservation.** A downscaled field is conventionally constrained to
  aggregate to the coarse observation within each cell. This requires a coarse
  reference in the target units (%); the coarse driver here (SMIPS) is in mm and
  is a model input rather than an observed coarse-% field. A suitable formulation
  decomposes the 30 m prediction into a cell mean and a terrain anomaly and
  rebases the cell mean onto a coarse-% reference (e.g. a calibrated SMIPS-to-%
  transfer, or resampled SMAP/ASCAT), preserving the fine structure while removing
  the per-cell offset.
- **Soil covariates.** The principal residual error is an absolute-level bias on
  transfer to a new catchment; static soil properties (e.g. SLGA texture/clay)
  are the most direct candidate and would be added as static per-pixel features
  here. See the [README](../README.md#future-work).
