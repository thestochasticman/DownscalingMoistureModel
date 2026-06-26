# `model.py` — the downscaling regressor (Stage 5)

Source: [`../../emt/model.py`](../../emt/model.py)

A Random Forest learns the in-situ root-zone target from the coarse SMIPS value
plus the fine terrain covariates and seasonality:

```
sm_rootzone_pct ~ smips_totalbucket + terrain(...) + doy_sin + doy_cos
```

Applied per 30 m pixel (Stage 6) this sharpens the ~1 km SMIPS field using the
terrain structure inside each coarse cell.

| function | role |
|---|---|
| `build_estimator(**kw)` | a sensible default `RandomForestRegressor` |
| `fit(table)` | fit on the full table |
| `leave_site_out_cv(table)` | the honest spatial skill estimate (below) |
| `metrics(y, ŷ)` | `rmse, ubrmse, bias, r, r2, n` |
| `feature_importance(model)` | sorted RF importances |

## Key decisions
- **`lat`/`lon` are deliberately NOT features.** They would let the forest
  memorise station location and inflate skill. `station` id is used *only* as the
  cross-validation group.
- **Leave-site-out CV is the headline metric** (`LeaveOneGroupOut` over
  `station`): train on all sites but one, predict the held-out site. At inference
  the model meets terrain/SMIPS combinations from unseen locations, so this — not
  a random split — is the honest estimate.

## What the current run shows
On the 4-station Kyeamba table the CV is **degenerate** (pooled `r ≈ −0.45`,
`r² ≈ −1.16`): each held-out site is tracked well in *shape* (within-site
`r ≈ 0.9`) but its absolute *level* is badly biased, and SMIPS importance is
~0.006. With only four stations sharing ~one SMIPS pixel there is no cross-site
signal for the forest to anchor the level. See Figures 2–3 in the
[README](../README.md). The fix here is **more spatial spread** (more sites /
AOIs / years), not a code change.
