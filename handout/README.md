# Statistical downscaling of SMIPS soil moisture to 30 m

## Objective

Produce a 30 m daily estimate of root-zone soil moisture by statistically
downscaling the TERN SMIPS `TotalBucket` profile soil-water product (mm,
≈1 km, daily). A regression model is trained against in-situ observations to
learn how fine-scale terrain redistributes moisture within each ≈1 km SMIPS
cell; the model is then applied at the 30 m resolution of the Copernicus DEM.

This document describes the processing pipeline, two data-quality issues
identified and resolved during development, and the cross-validation and
spatial-transfer results obtained to date. Each pipeline component has a
corresponding note under [`modules/`](modules/); all figures are reproducible
(see [Reproducibility](#reproducibility)).

## Pipeline

| Stage | Module | Output |
|---|---|---|
| 1. Ground truth | [`oznet.py`](modules/oznet.py.md) | Daily root-zone (0–90 cm) soil moisture per OzNet station (the regression target) |
| 2. Study areas | [`queries.py`](modules/queries.py.md) | PaddockTS `Query` extents (per-station windows and focus catchments) |
| 3a. Coarse predictor | [`smips.py`](modules/smips.py.md) | SMIPS `TotalBucket` field (mm), the quantity being downscaled |
| 3b. Fine predictors | [`covariates.py`](modules/covariates.py.md) | 30 m terrain covariates (elevation, slope, northness/eastness, TWI, HLI, flow accumulation) |
| 4. Feature assembly | [`features.py`](modules/features.py.md) | Training table: one record per station-day (target, SMIPS, terrain, seasonality) |
| 5. Model | [`model.py`](modules/model.py.md) | Random Forest regressor with leave-site-out cross-validation |
| 6. Downscaling | [`downscale.py`](modules/downscale.py.md) | Per-pixel application of the model to produce a 30 m field |

## Model specification

```
sm_rootzone_pct  ~  smips_totalbucket + terrain(...) + doy_sin + doy_cos
```

Station coordinates (`lat`/`lon`) are excluded from the feature set to prevent
the model from encoding station identity; `station` is retained solely as the
grouping variable for spatial cross-validation. Reported generalisation skill is
always the leave-site-out (or leave-region-out) estimate, never an in-sample
fit.

## Data quality: SMIPS WCS grid resampling

The TERN GeoServer Web Coverage Service resamples each `GetCoverage` response to
fit an integer pixel count within the requested bounding box. Consequently the
grid origin and cell size of the returned raster depend on the request extent,
and a fixed geographic point can fall in different ≈1 km cells depending on the
window requested. For station K6 the sampled value was 43.7, 49.6, or 61.6 mm
across three request windows for the same location and date (a ≈40 % range).

The resolution (`snap_bbox` in [`smips.py`](modules/smips.py.md)) aligns every
request to the native SMIPS grid, taken from the WCS `DescribeCoverage`
(origin 112.90499°E, −43.73500°N; cell size 0.0099976°). With an aligned
envelope the server's integer-pixel fit coincides with the native grid, making
the sampled value independent of the request window. (The `scaleFactor=1`
parameter was tested and confirmed to have no effect.)

![SMIPS sampling correction](figures/smips_correction.png)

- **(a)** Pre- and post-correction SMIPS values. Three of the four stations were
  unaffected (their original windows already resolved to the correct cell);
  station K6 changed.
- **(b)** Mean per-station change: K6 +15.4 mm, others ≈0.
- **(c)** Corrected SMIPS against the target. The four Kyeamba stations occupy
  near-horizontal bands: SMIPS varies within a site but provides limited
  separation between site-mean moisture levels. This is relevant to the result
  in the next section.

Cached SMIPS extracted before this correction used window-dependent values; the
training table and model were rebuilt after clearing the cache.

## Initial evaluation: single-cluster training set

The model was first evaluated on four Kyeamba stations (June–July 2020).
Leave-site-out cross-validation yielded negative pooled skill
(r = −0.45, r² = −1.16), and SMIPS received negligible feature importance
(0.006).

![Leave-site-out cross-validation, Kyeamba](figures/leave_site_out_cv.png)

- **(a)** Predicted versus observed for each held-out station. Per-station
  correlation is high (each cluster aligns along the diagonal) but offset from
  the 1:1 line.
- **(b)** Per-station correlation is ≈0.9 throughout, while per-station bias is
  large (K10 +3.5, K12 −4.6). High correlation combined with large bias produces
  the negative pooled r².
- **(c)** Feature importance is dominated by terrain (≈88 %); SMIPS contributes
  ≈0.006.

![Per-site time series, Kyeamba](figures/per_site_timeseries.png)

The per-station time series show that temporal dynamics are reproduced while the
absolute level is offset.

**Assessment.** The four stations lie within approximately one SMIPS cell, so
the coarse predictor carries no between-station signal. The model reproduces
temporal dynamics but cannot determine the absolute level of a held-out station.
This is a limitation of training-set spatial coverage rather than of the
pipeline, and motivates expansion to multiple catchments.

## Expanded training set: Yanco, Kyeamba, Adelong (2006–2010)

The training table was rebuilt across the three clustered catchments
(30 stations, 40,590 station-days). SMIPS was retrieved as three site-level
cubes via cluster-fetch ([`features.py`](modules/features.py.md)). The sites are
separated by 100–150 km, providing between-site variation in the coarse
predictor.

![Catchment cross-validation results](figures/catchment_results.png)

- **(a)** Site-level SMIPS distributions differ (Adelong ≈53, Kyeamba ≈38,
  Yanco ≈26 mm), supplying the between-site signal absent in the single-cluster
  set.
- **(b)** Leave-site-out fit across 30 held-out stations: pooled r = 0.54,
  r² = +0.16.
- **(c)** Feature importance: SMIPS becomes the highest-ranked predictor (0.34),
  followed by slope (0.27).
- **(d)** Residual per-station bias persists (e.g. K12 −16 %, A5 +11 %).
  Per-station ubRMSE remains low (≈3–4 %), indicating that temporal dynamics are
  well reproduced and the residual is a level offset.

| Metric | Kyeamba only (4 stations) | Catchment (30 stations) |
|---|---|---|
| Pooled leave-site-out r | −0.45 | +0.54 |
| Pooled leave-site-out r² | −1.16 | +0.16 |
| SMIPS feature importance | 0.006 | 0.34 |
| corr(target, SMIPS) | 0.25 | 0.53 |

Between-site variation in SMIPS changes its feature importance from negligible
to dominant and brings pooled cross-validation skill into the positive range.

## 30 m downscaling and spatial transfer

Stage 6 applies the model per pixel to produce the 30 m field
([`downscale.py`](modules/downscale.py.md)). To obtain an out-of-sample
assessment, the model was trained on Kyeamba and Adelong only, with Yanco
withheld entirely; the resulting field and its validation therefore represent
transfer to an unobserved catchment. Evaluation date: 2008-07-31 (all 12 Yanco
stations reporting).

![30 m downscaling over Yanco](figures/downscale_yanco.png)

- **(a)–(c)** The ≈1 km SMIPS input is resolved to a 30 m field; the detail view
  (c) shows drainage-network structure not present in the coarse input.
- **(d)** Validation at the 12 withheld Yanco stations:

  | Metric | Value | Interpretation |
  |---|---|---|
  | RMSE | 11.5 % | Dominated by bias |
  | ubRMSE | 2.4 % | Bias-removed error is low; spatial pattern transfers |
  | Bias | +11.3 % | Systematic over-prediction of the semi-arid Yanco plains by a model trained on wetter upland sites |
  | r | 0.41 | Moderate, across 12 stations |

**Assessment.** Under full-region transfer the relative spatial structure is
reproduced (ubRMSE 2.4 %) while the absolute level carries a substantial regional
bias. This is consistent with the residual per-station bias observed in Stage 5:
SMIPS and terrain do not encode the local soil and climate properties that set
the absolute moisture level. See [Limitations](#limitations) and
[Future work](#future-work).

## Metrics

| Metric | Definition |
|---|---|
| RMSE | Root-mean-square error, √mean((pred − obs)²) |
| ubRMSE | Bias-removed RMSE, √(RMSE² − bias²); the standard soil-moisture skill statistic |
| Bias | mean(pred − obs) |
| r | Pearson correlation |
| r² | 1 − SS_res / SS_tot (may be negative when bias dominates) |

Generalisation skill is reported as the leave-site-out or leave-region-out
value in all cases.

## Limitations

- **Absolute level bias on transfer.** The principal residual error is a regional
  level offset (+11.3 % for Yanco under leave-region-out). Predictors currently
  encode moisture *dynamics* (low ubRMSE) but not the local *baseline*.
- **Training-set coverage.** Three catchments constrain between-site
  generalisation; additional sites would improve the estimate of transfer skill.
- **Single-date downscaling demonstration.** Stage 6 is evaluated on one date;
  multi-date and seasonal evaluation remains outstanding (Stage 7).

## Future work

1. **Soil covariates (priority).** Static soil properties (clay/sand fraction,
   bulk density) from the
   [Soil and Landscape Grid of Australia](https://www.clw.csiro.au/aclep/soilandlandscapegrid/)
   (SLGA, ≈90 m) are the most direct candidate for encoding the absolute-level
   baseline. They would be added as static per-pixel features in
   [`features.py`](modules/features.py.md) and
   [`downscale.py`](modules/downscale.py.md) with no change to model structure.
   Not yet implemented.
2. **Mass conservation.** Constrain the 30 m field to aggregate to a coarse
   reference within each cell (decomposition into cell mean plus terrain anomaly,
   with the mean rebased onto the reference). Requires a coarse reference in the
   target units (%); see the
   [`downscale.py` note](modules/downscale.py.md#future-work-not-yet-implemented).
3. **Bias correction.** A per-site offset or quantile mapping to SMIPS
   climatology would address the regional level bias directly.

## Reproducibility

Run from the repository root with the `paddockts` conda environment active (so
that `PaddockTS` and the cached inputs are available):

```bash
PYTHONPATH=. python handout/plot_results.py     # smips_correction, leave_site_out_cv, per_site_timeseries
PYTHONPATH=. python handout/plot_catchment.py   # catchment_results
PYTHONPATH=. python handout/plot_downscale.py   # downscale_yanco (30 m field)
```

`plot_results.py` rebuilds the Kyeamba June–July 2020 table, reconstructs the
pre-correction SMIPS values for comparison, and runs the cross-validation.
`plot_catchment.py` and `plot_downscale.py` operate on the expanded table
(`data/train_catchment_2006_2010.csv`).

The module notes under [`modules/`](modules/) summarise each source file; the
implementation of record is in [`../emt/`](../emt/).
