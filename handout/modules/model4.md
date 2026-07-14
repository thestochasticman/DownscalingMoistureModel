# `model4`: regularised boosting + SMIPS climatology + soil (the improved model)

<!-- NAV -->
[← model3 · Gradient boosting](model3.md) · [Index](../README.md) · [model5 · Soil smoothing →](model5.md)
<!-- /NAV -->

Source: [`../../emt/model4/model.py`](../../emt/model4/model.py)

model4 adds two feature groups to the base predictors — **SMIPS lookback
windows** and **SLGA soil** — with a tuned boosting estimator. All numbers here
are the leak-free ones (see the
[Evaluation correction](../README.md#evaluation-correction)); an earlier version
reported inflated skill (model4 "0.35, doubling model1") from a look-ahead in the
SMIPS-climatology features.

| Leave-site-out NSE | model1 (RF, base) | **model4** |
|---|---|---|
| 30-station (dense catchments) | +0.15 | +0.17 |
| 36-station (+ regional M-sites) | −0.05 | **+0.31** |
| 36-station median per-station NSE | −1.01 | −0.57 |
| 36-station median per-station \|bias\| | 4.84 % | 4.88 % |

The honest story is different from the leaky one: on the **dense clusters alone**
model4 barely beats the base Random Forest (+0.17 vs +0.15) — the features need
spatial spread to pay off. Once the scattered regional
[M-sites](../README.md#extending-coverage-regional-sites-30--36-stations) broaden
the set to 36 stations, model1 goes *negative* (−0.05, no features to place the
new sites) while model4 reaches **+0.31** — so the features add real value, but as
a function of training coverage, not the "doubling" the leak implied.

The recommended model ([`model6`](model6.md)) adds antecedent-meteorology on top
(pooled +0.40). The training table is built by
[`emt/build_dataset.py`](../../emt/build_dataset.py).

| Function | Role |
|---|---|
| `build_estimator(**kw)` | Tuned `HistGradientBoostingRegressor` (small trees: `max_leaf_nodes=3, min_samples_leaf=150, max_features=0.5, max_iter=300`) |
| `ensure_features(table)` | Derives SMIPS lookback + soil columns on a standard table |
| `fit(table)` / `leave_site_out_cv(table)` | As in the other model packages (features auto-derived) |
| `feature_importance(model, table)` | Permutation importance |

## The features and estimator

1. **SMIPS lookback** (`smips_7d`, `smips_30d`, `smips_365d`, `smips_anom`;
   [`features.add_smips_climatology`](../../emt/features.py)). Trailing means of
   the pixel's SMIPS over the past 7 / 30 / 365 days, plus today's departure from
   the past-year level — the level/departure split that base models 1–3 lack.
   Strictly backward-looking and SMIPS-only (no in-situ, cannot memorise
   stations). **Correction:** an earlier *full-period* form of these features saw
   the future and inflated their apparent contribution (the retracted "+0.08
   lever"); the honest lookback form helps, but by less.
2. **SLGA soil** ([`slga.py`](slga.py.md)). With a level signal present the four
   soil covariates add texture; used *alone* (models 1–3) they acted as a
   near-unique station ID and hurt leave-site-out skill — a conditional result,
   not absolute.
3. **Tuned estimator.** Small trees with heavy leaf/feature regularisation
   (`max_leaf_nodes=3`, `min_samples_leaf=150`, `max_features=0.5`), selected by
   GroupKFold-on-station on the leak-free features. The recommended
   [`model6`](model6.md), with its larger feature set, tunes to the *opposite*
   (unlimited trees + feature subsampling) — the earlier "tiny trees are always
   best" conclusion was itself an artefact of the leak, not a general truth.

## What was tested and did NOT make the cut

- **SILO climate dynamics** (rain/PET/VPD, via PaddockTS + the open S3 archive):
  +0.02 alone, ~0 once soil is present *in this coarse form*. **Revisited and now
  used:** a targeted trailing-window form (last week/month/year water balance) does
  help and is the basis of [`model6`](model6.md), which supersedes model4.
- **SMIPS temporal lags/rolling means**: no gain (the anomaly features already
  carry the state).
- **Aridity statics, equal-station weighting**: hurt pooled skill.
- **Two-stage decomposition** (explicit site-mean model + anomaly model): 0.24 —
  the flat feature set lets the trees do the same split better.
- **RF+HGB ensembling**: pooled tie (0.356), worse per-station; not worth the
  complexity.

## The oracle diagnostic

Replacing the model's implicit level with each station's *true* mean (keeping the
learned anomaly model) yields a large jump in NSE — nearly the whole residual is
the site-level baseline, which makes the future-work priority precise: more sites
buys a better *level* model specifically. (The exact figures — "0.83, 28/30" —
were computed on the leaky features and are pending re-derivation on the corrected
features; the qualitative conclusion, that the remaining error is level not
dynamics, holds and is confirmed by the still-negative per-station NSE.)

## Validation notes

The winning configuration was selected on the same 30-station LOSO used
throughout, so a selection-on-test check was run: under **leave-region-out**
(train two catchments, predict the third — a split never used for selection)
model4 scores +0.12 where model1 scores −0.72, with every regional bias smaller
(Kyeamba +0.45; Yanco bias +9.2 → +6.1; Adelong −10.4 → −7.3). The gains
generalise.

## Inference

Applying model4 per pixel ([`downscale.py`](downscale.py.md)) needs the SMIPS
lookback rasters (the past 7/30/365-day SMIPS means over the AOI, ending at the
prediction day) and the SLGA soil stack ([`slga.soil_covariates`](slga.py.md)),
passed via `extra_layers`. **The inference path is being updated** from the old
full-period climatology (`smips_mean_px`) to these trailing windows; the
downscaling galleries pre-date the correction and are being regenerated.

The per-station-skill vs spatial-texture tension the soil covariates introduce is
explored in [`model5`](model5.md).

---
<!-- NAV -->
[← model3 · Gradient boosting](model3.md) · [Index](../README.md) · [model5 · Soil smoothing →](model5.md)
<!-- /NAV -->
