# `downscale.py`: 30 m field generation (Stage 6)

<!-- NAV -->
[← model9 · Pedotransfer readout](model9.md) · [Index](../README.md) · [Use the model →](predict.py.md)
<!-- /NAV -->

Source: [`../../emt/downscale.py`](../../emt/downscale.py)

Applies the fitted model per pixel to produce a 30 m root-zone soil-moisture
field for one day over a focus area — the product the pipeline exists to generate.

| Function | Role |
|---|---|
| `downscale(model, query, day, features, extra_layers=…)` | `xr.Dataset` on the 30 m grid with `sm_pred` (%) and `smips_native` (mm) |

## Procedure

1. Build the 30 m terrain covariate stack ([`covariates.py`](covariates.py.md))
   and retrieve the coarse SMIPS field for the day ([`smips.py`](smips.py.md)).
2. Resample SMIPS to the 30 m grid by nearest neighbour, so each fine pixel
   inherits its overlying ≈1 km value.
3. Supply the model's static/among-day covariates as `extra_layers` — the SMIPS
   **lookback** rasters (past 7/30/365-day means + anomaly, as of the day, from
   [`smips.smips_lookback_day`](../../emt/smips.py)), SLGA soil
   ([`slga.soil_covariates`](slga.py.md)) and, for [model6](model6.md), the
   gridded antecedent meteorology
   ([`antecedent.antecedent_grid`](../../emt/antecedent.py)); each raster is
   reprojected onto the 30 m grid. Every window is strictly backward-looking — the
   old full-period `smips_mean_px` climatology (a look-ahead leak) has been
   removed (see [Evaluation correction](../README.md#evaluation-correction)).
4. Assemble the features per pixel (in `FEATURES` order), predict over pixels with
   complete covariates, and reshape to the grid; pixels outside the DEM footprint
   are NaN.

The fine 30 m structure derives from the terrain, soil and SMIPS-lookback
covariates varying within each coarse SMIPS cell; the antecedent meteorology adds
the ≈5 km recent-weather gradient. In-situ observations are never an input — they
are only the training label and the validation reference. The shipped one-call
inference tool is [`emt/predict.py`](../../emt/predict.py) (see
[Use the model](../README.md#use-the-model)).

## The generated product

Coarse ≈1 km SMIPS input (left, mm) beside the generated 30 m field (right, %),
over the Kyeamba focus area on nine dates through 2008, one row per date:

![Coarse SMIPS beside generated 30 m field](../figures/downscale_gallery_paired.png)

Reading across each row is the downscaling (blocky ≈1 km cells → resolved terrain
structure: drainage lines, wet valleys, dry ridges); reading down the rows is the
seasonal cycle (wetter through the June–September austral winter, drier in
summer). The same two galleries at full size, each on its own shared colour scale,
are [`downscale_gallery_smips.png`](../figures/downscale_gallery_smips.png) and
[`downscale_gallery.png`](../figures/downscale_gallery.png). These
are qualitative; the reported skill is the leave-site-out estimate on the
[evaluation page](evaluation.py.md).

The recommended model's version (model6, with the gridded antecedent
meteorology):

![model6 downscaled 30 m field, Kyeamba seasonal gallery](../figures/downscale_gallery_model6.png)

Reproduced by [`plot_downscale_gallery.py`](../plot_downscale_gallery.py) and
[`plot_downscale_gallery_model6.py`](../plot_downscale_gallery_model6.py). A
single-date, leave-region-out validation against held-out stations is
[`plot_downscale_model4.py`](../plot_downscale_model4.py) /
[`plot_downscale_model6.py`](../plot_downscale_model6.py); note single-date NSE is
unstable, so those are visual checks, not the reported metric.

## Future work: mass conservation

A downscaled field is conventionally constrained to aggregate to the coarse
observation within each cell. This needs a coarse reference *in the target units
(%)*; the coarse driver here (SMIPS) is in mm and is a model input, not an
observed coarse-% field. A suitable formulation decomposes the 30 m prediction
into a cell mean and a terrain anomaly and rebases the cell mean onto a coarse-%
reference (e.g. a calibrated SMIPS-to-% transfer, or resampled SMAP/ASCAT),
preserving the fine structure while removing the per-cell offset.

---
<!-- NAV -->
[← model9 · Pedotransfer readout](model9.md) · [Index](../README.md) · [Use the model →](predict.py.md)
<!-- /NAV -->
