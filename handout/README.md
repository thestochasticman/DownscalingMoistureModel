# Statistical downscaling of SMIPS soil moisture to 30 m

## Objective

Produce a 30 m daily estimate of root-zone soil moisture by statistically
downscaling the TERN SMIPS `TotalBucket` profile soil-water product (mm,
≈1 km, daily). A regression model is trained against in-situ observations to
learn how fine-scale terrain, soil and recent weather redistribute moisture
within each ≈1 km SMIPS cell; the model is then applied at the 30 m resolution of
the Copernicus DEM.

The intended end product is **national (all of Australia)**. Every covariate used
here is Australia-wide, so the method already runs anywhere; the work to date
develops and validates it in the Murrumbidgee catchment, and the national in-situ
set needed to validate it Australia-wide is being assembled in a separate
repository.

This page is only the landing index and a snapshot of the final model. The full
pipeline, methodology, experiments and results live in the per-page module notes
below — each self-contained and chained with prev/next links, so the handout can
be read straight through. **Start here → [Ground truth (OzNet)](modules/oznet.py.md).**

## Contents

The story runs in three acts: build the dataset, evaluate and improve the model,
then apply it. Each item links to a self-contained page.

**1 · Building the dataset** — assemble one table of predictors + target.

| # | Page | What it contributes |
|---|---|---|
| 1 | [Ground truth — OzNet in-situ](modules/oznet.py.md) | Daily root-zone (0–90 cm) soil moisture per station — the target |
| 2 | [Study areas](modules/queries.py.md) | PaddockTS `Query` extents (per-station windows, focus catchments) |
| 3 | [SMIPS coarse predictor](modules/smips.py.md) | The ≈1 km `TotalBucket` field (mm) being downscaled |
| 4 | [Terrain covariates](modules/covariates.py.md) | 30 m DEM derivatives (slope, aspect, TWI, HLI, accumulation) |
| 5 | [Soil covariates (SLGA)](modules/slga.py.md) | Root-zone clay/sand/AWC/bulk-density |
| 6 | [Training table](modules/features.py.md) | One row per station-day: target + all predictors |

**2 · Evaluating and modelling** — how skill is measured, and six models.

| # | Page | Role |
|---|---|---|
| 7 | [Evaluation](modules/evaluation.py.md) | Leave-site-out cross-validation and the metrics |
| 8 | [model1 · Random Forest](modules/model1.md) | Baseline (pooled NSE +0.15) |
| 9 | [model2 · Linear](modules/model2.md) | Interpretable comparison |
| 10 | [model3 · Gradient boosting](modules/model3.md) | Stock-boosting reference |
| 11 | [model4 · Climatology + soil](modules/model4.md) | The improved model (pooled NSE +0.37) |
| 12 | [model5 · Soil smoothing](modules/model5.md) | A documented tradeoff, not recommended |
| 13 | [model6 · Antecedent meteorology](modules/model6.md) | **Recommended** — model4 + climate (pooled NSE +0.39) |

**3 · Applying the model**

| # | Page | Output |
|---|---|---|
| 14 | [Downscaling to 30 m](modules/downscale.py.md) | Per-pixel application → the 30 m field |

## The model

The recommended model is [**model6**](modules/model6.md): a regularised
histogram gradient-boosting regressor that predicts the in-situ root-zone target
from the coarse SMIPS value and a set of fine, national covariates —

```
sm_rootzone_pct  ~  smips_totalbucket                        (coarse predictor, mm)
                  + elevation, slope, northness, eastness,
                    twi, hli, accumulation                    (30 m terrain)
                  + smips_mean_px, smips_std_px,
                    smips_anom, smips_z                       (SMIPS pixel climatology)
                  + soil_clay, soil_sand, soil_awc, soil_bdw  (SLGA soil)
                  + rain_7/30/365, ppet_30/365, vpd_30,
                    rain_365_anom                             (antecedent SILO weather)
                  + doy_sin, doy_cos                          (seasonality)
```

Every predictor is a gridded product available at any Australian pixel; the
in-situ observation is used **only** as the training label and the validation
reference, never as a model input. Reported skill is the held-out
leave-site-out estimate (train on all stations but one, predict the unseen one):

| Skill (36 stations, 2006–2010, leave-site-out) | model6 |
|---|---|
| Pooled NSE / r | **+0.39 / 0.64** |
| Median per-station \|bias\| | 3.16 % |
| Per-station NSE > 0 | 17/36 |

How this model was arrived at — the baseline, the diagnosis of the residual as a
site-level *bias*, and the feature/estimator changes that reduced it — is
documented across the model pages ([model1](modules/model1.md) →
[model6](modules/model6.md)); the 30 m product it generates is on the
[downscaling page](modules/downscale.py.md).

## Training table (inputs and target)

One row per station-day. The target is the OzNet root-zone observation; every
other column is a predictor. Three representative rows — a dry site, a mid site
and a wet site:

| column | group | dry (M7) | mid (M6) | wet (A3) |
|---|---|---|---|---|
| **sm_rootzone_pct** | **target (%)** | **12.5** | **21.5** | **33.9** |
| smips_totalbucket | coarse (mm) | 6.0 | 0.0 | 77.6 |
| elevation | terrain (m) | 135 | 91 | 500 |
| slope | terrain (°) | 0.4 | 0.6 | 11.3 |
| northness | terrain | −0.99 | 0.65 | −0.75 |
| eastness | terrain | −0.13 | 0.76 | 0.67 |
| twi | terrain | 4.9 | 4.5 | 2.7 |
| hli | terrain | 0.85 | 0.86 | 0.94 |
| accumulation | terrain | 1 | 1 | 3 |
| smips_mean_px | SMIPS clim (mm) | 16.7 | 23.8 | 46.2 |
| smips_std_px | SMIPS clim (mm) | 20.5 | 31.1 | 21.9 |
| smips_anom | SMIPS clim (mm) | −10.7 | −23.8 | 31.4 |
| smips_z | SMIPS clim | −0.52 | −0.77 | 1.43 |
| soil_clay | SLGA (%) | 27.3 | 35.5 | 32.5 |
| soil_sand | SLGA (%) | 61.7 | 47.3 | 52.1 |
| soil_awc | SLGA | 11.8 | 9.7 | 12.0 |
| soil_bdw | SLGA | 1.52 | 1.49 | 1.42 |
| rain_7 | antecedent (mm) | 2.5 | 0.0 | 148.0 |
| rain_30 | antecedent (mm) | 11.7 | 5.8 | 274.4 |
| rain_365 | antecedent (mm) | 299 | 301 | 1188 |
| ppet_30 | antecedent (mm) | −60.6 | −82.7 | 224.1 |
| ppet_365 | antecedent (mm) | −1915 | −1819 | −509 |
| vpd_30 | antecedent | 7.1 | 8.4 | 3.4 |
| rain_365_anom | antecedent (mm) | −5.2 | 27.7 | 403.3 |
| doy_sin | seasonality | −0.78 | 0.71 | −0.94 |
| doy_cos | seasonality | −0.62 | −0.71 | −0.35 |
| *station / site / time* | *identifiers* | *M7 · 2009-08-23* | *M6 · 2006-05-17* | *A3 · 2010-09-10* |

Station coordinates are **not** predictors (they would let the model memorise
station identity); `station` is used only to group the leave-site-out
cross-validation. The table is built reproducibly by
[`emt/build_dataset.py`](../emt/build_dataset.py) (see the
[training-table page](modules/features.py.md)).

## Reproducibility

Run from the repository root with the `paddockts` conda environment active. The
training table and every figure are regenerated by the scripts listed on the
[downscaling page](modules/downscale.py.md) and each model page; the builder is
`python -m emt.build_dataset`. The module notes under [`modules/`](modules/)
summarise each source file; the implementation of record is in
[`../emt/`](../emt/).
