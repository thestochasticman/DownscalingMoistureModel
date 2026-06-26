# `downscale.py` — apply the model to a 30 m field (Stage 6)

Source: [`../../emt/downscale.py`](../../emt/downscale.py)

Turns the fitted regressor into the actual product: a **30 m map of root-zone
soil moisture** for one day over a focus AOI.

| function | role |
|---|---|
| `downscale(model, query, day)` | `xr.Dataset` on the 30 m terrain grid with `sm_pred` (%) and `smips_native` (mm) |

## How it works
1. Build the 30 m terrain covariate stack ([`covariates.py`](covariates.py.md))
   and fetch the coarse SMIPS field for the day ([`smips.py`](smips.py.md)).
2. Resample SMIPS onto the 30 m grid with **nearest** so every fine pixel
   inherits its overlying ~1 km value — the field the model *sharpens*.
3. Assemble the model's features per pixel (in `FEATURES` order), predict on the
   finite-covariate pixels, and reshape to the grid (outside the DEM footprint →
   NaN).

The lift comes entirely from the terrain covariates varying at 30 m *within* each
coarse SMIPS cell — see panel (c) of the Stage 6 figure in the
[README](../README.md), where dendritic drainage structure appears that the
~1 km input cannot resolve.

## Documented future work (not yet implemented)
- **Mass conservation.** A downscaled field is normally constrained so it
  aggregates back to the coarse observation within each cell. That needs a
  *coarse reference in the target's units* (%) — but here the coarse driver
  (SMIPS) is in mm and is an *input*, not an observed coarse-% field. A clean
  version would decompose the 30 m prediction into a cell mean + a
  terrain-driven anomaly and rebase the cell mean onto a coarse-% reference
  (e.g. a calibrated SMIPS-to-% transfer, or resampled SMAP/ASCAT). This would
  remove the per-cell offset while preserving the fine structure.
- **SLGA soil covariate.** See the [README](../README.md#future-work) — the main
  remaining error is an absolute *level* bias when transferring to a new
  catchment; static soil properties (texture/clay from SLGA) are the most likely
  fix and would slot in as additional static per-pixel features here.
