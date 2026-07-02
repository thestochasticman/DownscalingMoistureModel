# `model4`: regularised boosting + SMIPS climatology + soil (the improved model)

Source: [`../../emt/model4/model.py`](../../emt/model4/model.py)

The product of a systematic improvement search (2026-07-02) over features,
estimators, weighting, ensembling and problem restructuring — all scored by the
same leave-site-out harness ([`evaluation.py`](evaluation.py.md)). It more than
doubles the cross-site skill of [`model1`](model1.md) **and** improves the
per-station profile at the same time, breaking the tradeoff the earlier models
were stuck on.

| Metric (30 stations, 2006–2010) | model1 (RF) | **model4** |
|---|---|---|
| Pooled LOSO NSE / r | +0.15 / 0.53 | **+0.35 / 0.62** |
| Per-station NSE > 0 | 7/30 | **14/30** |
| Median per-station NSE | −0.56 | **−0.07** |
| Median per-station r | 0.75 | 0.80 |
| Leave-region-out NSE | −0.72 | **+0.12** |

The leave-site-out fit, feature importance, the paired model1 → model4
per-station NSE/bias comparison, and the 30-station held-out time series are
plotted by [`plot_model4_results.py`](../plot_model4_results.py)
(`figures/model4_results.png`, `figures/model4_per_station.png`) — see the
[README](../README.md#the-improvement-search-model4).

| Function | Role |
|---|---|
| `build_estimator(**kw)` | `HistGradientBoostingRegressor(max_leaf_nodes=3, learning_rate=0.03, max_iter=800, l2=1)` |
| `ensure_features(table)` | Derives climatology + soil columns on a standard table |
| `fit(table)` / `leave_site_out_cv(table)` | As in the other model packages (features auto-derived) |
| `feature_importance(model, table)` | Permutation importance |

## The three ingredients

1. **SMIPS pixel climatology** (`smips_mean_px`, `smips_std_px`, `smips_anom`,
   `smips_z`; [`features.add_smips_climatology`](../../emt/features.py)). The
   pixel's long-term SMIPS mean/std plus the day's anomaly factor the coarse
   predictor into a static *level* and a dynamic *departure*. This supplies the
   local-baseline signal whose absence caused the per-station bias in models
   1–3 — and because it is derived from SMIPS alone, it is computable at every
   30 m pixel at inference and cannot memorise stations. Largest single lever:
   +0.08 pooled NSE on every estimator tested.
2. **Extreme regularisation.** Pooled skill rises monotonically as boosting
   trees shrink (`max_leaf_nodes` 31 → 0.20, 15 → 0.25, 8 → 0.28, 6 → 0.29,
   4 → 0.31, **3 → 0.34**; 2 falls back). Three-leaf trees cannot memorise site
   quirks, forcing transferable structure. Seed-stable (±0.004 over 3 seeds).
3. **SLGA soil, rehabilitated** ([`slga.py`](slga.py.md)). With the climatology
   anchoring the level, the four soil covariates add skill and give the best
   per-station profile (median −0.07). The earlier
   [negative soil result](../README.md#soil-covariate-experiment-negative-result)
   was *conditional* on the missing level feature, not absolute: without an
   anchor, soil acted as a station ID; with one, it contributes texture signal.

## What was tested and did NOT make the cut

- **SILO climate dynamics** (rain/PET/VPD, via PaddockTS + the open S3 archive):
  +0.02 alone, ~0 once soil is present. Loader path validated; worth revisiting
  with more sites.
- **SMIPS temporal lags/rolling means**: no gain (the anomaly features already
  carry the state).
- **Aridity statics, equal-station weighting**: hurt pooled skill.
- **Two-stage decomposition** (explicit site-mean model + anomaly model): 0.24 —
  the flat feature set lets the trees do the same split better.
- **RF+HGB ensembling**: pooled tie (0.356), worse per-station; not worth the
  complexity.

## The oracle diagnostic

Replacing the model's implicit level with each station's *true* mean (keeping
the learned anomaly model) yields **NSE 0.83, 28/30 stations positive**. The
remaining gap (0.35 → 0.83) is almost entirely the site-level baseline, which
makes the future-work priority precise: more sites buys a better *level* model
specifically.

## Validation notes

The winning configuration was selected on the same 30-station LOSO used
throughout, so a selection-on-test check was run: under **leave-region-out**
(train two catchments, predict the third — a split never used for selection)
model4 scores +0.12 where model1 scores −0.72, with every regional bias smaller
(Kyeamba +0.45; Yanco bias +9.2 → +6.1; Adelong −10.4 → −7.3). The gains
generalise.

## Inference

Applying model4 per pixel ([`downscale.py`](downscale.py.md)) needs two static
rasters passed via `extra_layers`: the SMIPS climatology over the AOI
([`smips.smips_climatology`](../../emt/smips.py), thinned multi-year fetch,
cached) and the SLGA soil stack ([`slga.soil_covariates`](slga.py.md)).
`smips_anom`/`smips_z` are derived per pixel inside `downscale` from the day's
SMIPS and the climatology.

Demonstrated end-to-end by
[`plot_downscale_model4.py`](../plot_downscale_model4.py) (leave-Yanco-out,
2008-07-31): regional bias improves over model1 (+11.3 → +9.0 %, RMSE 11.5 →
10.3 %) but the single-day spatial pattern is worse (ubRMSE 2.4 → 4.9 %) — the
30 m field inherits blocky SLGA map-unit boundaries and mutes some terrain
detail. The full-record leave-region-out numbers (Yanco NSE −1.81 → −0.44)
remain the meaningful transfer measure; see the
[README](../README.md#the-same-demonstration-with-model4) for the figure and
the calibration-vs-texture tradeoff.
