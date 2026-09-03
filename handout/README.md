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

**2 · Evaluating and modelling** — how skill is measured, and ten models.

| # | Page | Role |
|---|---|---|
| 7 | [Evaluation](modules/evaluation.py.md) | Leave-site-out cross-validation and the metrics |
| 8 | [model1 · Random Forest](modules/model1.md) | Baseline (pooled NSE +0.15) |
| 9 | [model2 · Linear](modules/model2.md) | Interpretable comparison |
| 10 | [model3 · Gradient boosting](modules/model3.md) | Stock-boosting reference |
| 11 | [model4 · SMIPS lookback + soil](modules/model4.md) | +0.17 (30-stn) → +0.31 (with M-sites) |
| 12 | [model5 · Soil smoothing](modules/model5.md) | A documented tradeoff, not recommended |
| 13 | [model6 · Antecedent meteorology](modules/model6.md) | **Recommended** — model4 + weather (pooled NSE +0.38) |
| 14 | [model7 · Process model](modules/model7.md) | The process track's foundation: bucket water balance, no ML (median station r 0.83) |
| 15 | [model8 · Process model + SLGA soil](modules/model8.md) | **Recommended process model** — station-out NSE +0.41, blocked +0.32 (full stack: soil + aridity + capacity + weights) |
| 16 | [Blocked validation](modules/blocked_validation.md) | Leave-one-**block**-out transfer skill for model6 & model8 — the honest numbers for a national product |
| 17 | [model9 · Pedotransfer readout](modules/model9.md) | Per-site soil hydraulic limits replace two global readout constants — best blocked transfer (+0.35), narrow gain |
| 18 | [Temporal validation](modules/temporal_validation.md) | Leave-one-**year**-out across the drought break — time is the easy axis, space is the hard one |
| 19 | [model10 · The hybrid](modules/model10.md) | Bucket storage as an ML feature — a negative result, with one instructive exception |
| 20 | [nn-mlp · Neural net on model6 features](modules/nn_mlp.md) | model6 parity; static noise beats the station-identity leak |
| 21 | [nn-transformer · Forcing sequence model](modules/nn_transformer.md) | model6-level skill with **no SMIPS**; a learned water balance drifts in level |
| 22 | [nn-hybrid · Differentiable bucket](modules/nn_hybrid.md) | **Best blocked transfer** (+0.35 > model8's +0.32); quantile statics were the lever; M7 solved |
| 23 | [nn-stack · Learned combiner](modules/nn_stack.md) | Negative result — but the diversity mean is the repo's best (blocked +0.40) |
| 24 | [In-situ networks](modules/insitu_networks.md) | The loader contract, the survey of more data, and what is blocking it |
| 25 | [Data quality](modules/qc.md) | Sentinels, two rejected detectors, and the calibration drift that bounds the record |

**3 · Applying the model**

| # | Page | Output |
|---|---|---|
| 21 | [Downscaling to 30 m](modules/downscale.py.md) | Per-pixel application → the 30 m field |
| 22 | [Use the model (`predict.py`)](modules/predict.py.md) | Clone-and-run tool: bbox + date → 30 m GeoTIFF |
| 23 | [Use the process model](modules/model8.md#run-it-for-any-date) | model8 for **any date** — point series or 30 m map, no SMIPS |

## The model

The recommended model is [**model6**](modules/model6.md): a regularised
histogram gradient-boosting regressor that predicts the in-situ root-zone target
from the coarse SMIPS value and a set of fine, national covariates —

```
sm_rootzone_pct  ~  smips_totalbucket                        (coarse predictor, mm)
                  + elevation, slope, northness, eastness,
                    twi, hli, accumulation                    (30 m terrain)
                  + smips_7d, smips_30d, smips_365d,
                    smips_anom                                (SMIPS lookback: past 7/30/365-day means)
                  + soil_clay, soil_sand, soil_awc, soil_bdw  (SLGA soil)
                  + rain_7/30/365, ppet_30/365, vpd_30,
                    rain_365_anom                             (antecedent SILO weather)
                  + doy_sin, doy_cos                          (seasonality)
```

Every predictor is a gridded product available at any Australian pixel; the
in-situ observation is used **only** as the training label and the validation
reference, never as a model input. Every SMIPS/soil/weather statistic is
**backward-looking** (as of the prediction day), so no feature can see the
future. Reported skill is the held-out leave-site-out estimate (train on all
stations but one, predict the unseen one):

| Skill (36 stations, 2006–2010, leave-site-out) | model6 |
|---|---|
| Pooled NSE / r | **+0.38 / 0.62** |
| Median per-station \|bias\| | 3.85 % |
| Per-station NSE > 0 | 16/36 |
| Median per-station NSE | −0.19 |

**Read the pooled +0.38 with care.** It is the leave-*one*-station-out figure;
the more demanding grouped cross-validation (holding out ~7 stations at once)
gives a conservative **≈+0.25**, and the *per-station* NSE is still negative at
most sites (median −0.19). The model tracks moisture *dynamics* well (median
per-station r 0.81) but a per-station *level* bias remains — it is **not** a
solved product. An earlier version of this handout reported inflated skill from a
look-ahead leak in the SMIPS-climatology features; see
[Evaluation correction](#evaluation-correction).

**And read every station-out figure as an interpolation estimate.** 26 of the
37 stations sit in two tight clusters whose members share ~5 km forcing cells,
so holding out one station leaves its near-neighbours in training. Holding out
whole **spatially independent blocks** instead (9 folds) gives pooled
**≈+0.32** for the shipped model8 (whose full-stack configuration was adopted
from that experiment) and drops model6's block-median to +0.09, with the
failures concentrated at the climate-envelope edges — the honest transfer
numbers for a national product are on the
[blocked-validation page](modules/blocked_validation.md).

How this model was arrived at is documented across the model pages
([model1](modules/model1.md) → [model6](modules/model6.md)); the 30 m product it
generates is on the [downscaling page](modules/downscale.py.md).

## Use the model

The trained model ships in the repo (`data/models/model6.joblib`), so you can
produce a 30 m map for any Australian area and day **without retraining** — the
tool fetches every covariate for you and predicts per pixel.

**1 · Install.** Clone this repo and
[PaddockTS](https://github.com/thestochasticman/paddock-ts-local) side by side,
and use the `paddockts` conda environment (it has rioxarray, scikit-learn, etc.):

```bash
git clone git@github.com:thestochasticman/DownscalingMoistureModel.git
git clone git@github.com:thestochasticman/paddock-ts-local.git
conda activate paddockts
```

**2 · Credentials.** Three covariate sources need free accounts (put keys in
`~/.config/PaddockTS.json` as PaddockTS documents):

| Covariate | Source | Needs |
|---|---|---|
| SMIPS + its lookback | TERN GeoServer WCS | nothing (public) |
| Soil (SLGA) | TERN | a TERN API key |
| Antecedent weather | SILO / DataDrill | an email address |
| 30 m terrain | Copernicus DEM (public AWS bucket) | nothing (read anonymously) |

**3 · Run.** From the repo root, either the CLI —

```bash
PYTHONPATH=. python -m emt.predict \
    --bbox 147.30 -35.52 147.62 -35.10 \
    --date 2008-07-31
# writes soil_moisture_2008-07-31.tif + .png into the AOI's PaddockTS query dir
```

— or the Python function:

```python
from emt.predict import predict
ds = predict(bbox=(147.30, -35.52, 147.62, -35.10), day="2008-07-31")
ds["sm_pred"]                      # xr.DataArray, 30 m root-zone soil moisture (%)
ds.attrs["output_tif"], ds.attrs["output_png"]   # where they were saved
```

`--bbox` is `W S E N` in EPSG:4326; the output is a single-band GeoTIFF on the
30 m Copernicus-DEM grid plus a quick-look PNG, saved into that AOI's PaddockTS
query directory (alongside its cached covariates) — pass `-o PATH` for an extra
copy elsewhere, `--no-plot` to skip the PNG. A first run over a new area fetches
~a year of SMIPS and SILO (a few minutes); everything is cached under the AOI stub
for subsequent days.

> **Caveat — Murrumbidgee-trained, unvalidated elsewhere.** The shipped model is
> trained and validated only in the Murrumbidgee catchment. Run anywhere else and
> it will still produce a field, but with an uncorrected per-site level bias and
> no out-of-region validation (see the [skill caveats](#the-model) above). Treat
> off-catchment output as **indicative, not calibrated** — the tool prints this
> warning on every run.

## Evaluation correction

An earlier version of this work computed the SMIPS "climatology" features
(`smips_mean_px` etc.) as the **full-period** mean/std of each pixel's SMIPS —
which let every training day peek at the rest of the record, including its own
future. That look-ahead **inflated the reported leave-site-out NSE by ≈0.13**
(model6 read 0.39 where the honest figure is ≈0.26 untuned) and made the
per-station level bias look solved when it was not.

The fix (this version): the SMIPS features are now strictly **lookback** —
trailing means over the past 7 / 30 / 365 days (`smips_7d/30d/365d`), seeded from
real pre-2006 SMIPS so nothing is discarded, with the same treatment applied to
`rain_365_anom`. The estimators were then re-tuned honestly (GroupKFold on
station, never on the leave-site-out score). Two things came out of the honest
re-analysis:

- **The features still add real value** — base Random Forest is *negative*
  (−0.05 on 36 stations); the lookback + soil + antecedent features lift it to
  +0.40. So the skill is genuine, just to a lower ceiling than the leak implied.
- **The leaky-era "extreme regularisation" tuning was itself an artefact.** With
  leak-free features the optimum flips from tiny trees to large trees controlled
  by feature subsampling (`max_features=0.15`); proper tuning then recovers pooled
  NSE **≈0.38–0.40** — close to the retracted headline, but legitimately. (The
  unlimited-tree optimum scores 0.40 but takes ~90 min per cross-validation, so
  the production config caps trees at 127 for 0.377 at ~1/8 the cost.)

## Training table (inputs and target)

One row per station-day. The target is the OzNet root-zone observation; every
other column is a predictor. Three representative rows — a dry site, a mid site
and a wet site:

| column | group | dry (Y3) | mid (Y11) | wet (A5) |
|---|---|---|---|---|
| **sm_rootzone_pct** | **target (%)** | **12.4** | **21.5** | **33.8** |
| smips_totalbucket | coarse (mm) | 13.9 | 0.0 | 85.6 |
| elevation | terrain (m) | 145 | 115 | 377 |
| slope | terrain (°) | 2.4 | 0.1 | 0.2 |
| northness | terrain | 0.31 | 0.76 | −0.97 |
| eastness | terrain | 0.95 | 0.65 | −0.24 |
| twi | terrain | 6.9 | 6.4 | 9.8 |
| hli | terrain | 0.89 | 0.84 | 0.84 |
| accumulation | terrain | 43 | 1 | 65 |
| smips_7d | SMIPS past-7d mean (mm) | 16.4 | 3.3 | 87.2 |
| smips_30d | SMIPS past-30d mean (mm) | 13.2 | 0.8 | 83.7 |
| smips_365d | SMIPS past-year mean (mm) | 25.1 | 8.2 | 60.3 |
| smips_anom | today − past-year (mm) | −11.2 | −8.2 | 25.3 |
| soil_clay | SLGA (%) | 28.4 | 38.0 | 32.2 |
| soil_sand | SLGA (%) | 60.3 | 46.7 | 51.3 |
| soil_awc | SLGA | 11.0 | 10.8 | 11.0 |
| soil_bdw | SLGA | 1.52 | 1.50 | 1.44 |
| rain_7 | antecedent (mm) | 3.0 | 10.9 | 26.3 |
| rain_30 | antecedent (mm) | 26.7 | 12.1 | 96.4 |
| rain_365 | antecedent (mm) | 248 | 185 | 833 |
| ppet_30 | antecedent (mm) | −84 | −292 | 52 |
| ppet_365 | antecedent (mm) | −2118 | −2044 | −923 |
| vpd_30 | antecedent | 11.9 | 27.4 | 3.6 |
| rain_365_anom | antecedent (mm) | −45 | −171 | 175 |
| doy_sin | seasonality | 0.73 | 0.20 | −0.72 |
| doy_cos | seasonality | −0.68 | 0.98 | −0.70 |
| *station / site / time* | *identifiers* | *Y3 · 2007-05-15* | *Y11 · 2007-01-12* | *A5 · 2008-08-16* |

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
