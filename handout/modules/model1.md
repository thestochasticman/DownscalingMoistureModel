# `model1`: Random Forest regressor (the baseline model)

<!-- NAV -->
[← Evaluation (leave-site-out)](evaluation.py.md) · [Index](../README.md) · [model2 · Linear →](model2.md)
<!-- /NAV -->

Source: [`../../emt/model1/model.py`](../../emt/model1/model.py)

A Random Forest predicts the in-situ root-zone target from the coarse SMIPS
value, the terrain covariates, and seasonality:

```
sm_rootzone_pct ~ smips_totalbucket + terrain(...) + doy_sin + doy_cos
```

Applied per pixel ([`downscale.py`](../../emt/downscale.py)) the model resolves
the ≈1 km SMIPS field to 30 m using the terrain variation within each coarse
cell. This is the production model; [`model2`](model2.md) is an alternative kept
for comparison.

| Function | Role |
|---|---|
| `build_estimator(**kw)` | Default `RandomForestRegressor` configuration |
| `fit(table)` | Fit on the full table |
| `leave_site_out_cv(table)` | Spatial cross-validation (via `emt.evaluation`) |
| `feature_importance(model)` | Sorted impurity-based importances |

Scoring (`metrics`, `leave_site_out_cv`) lives in shared
[`emt/evaluation.py`](../../emt/evaluation.py) so every model is graded
identically; this module just supplies the feature list and the estimator.

## Design decisions

- **Coordinates excluded from features.** `lat`/`lon` would allow the model to
  encode station identity and inflate apparent skill; `station` is used only as
  the cross-validation grouping variable.
- **Leave-site-out cross-validation as the reported skill** (`LeaveOneGroupOut`
  over `station`): the model is trained on all stations but one and evaluated on
  the held-out station. Because inference occurs at unobserved locations, this
  provides the appropriate generalisation estimate rather than a random split.

## Evaluation

On the initial four-station Kyeamba table the cross-validation gives negative
pooled skill (r = −0.45, NSE = −1.16): per-station correlation is high (≈0.9) but
the absolute level is biased, and SMIPS importance is ≈0.006. The four stations
fall within approximately one SMIPS cell, leaving no between-station signal in
the coarse predictor. Expanding the training set to three catchments
(see the [README](../README.md)) raises pooled skill to r = 0.54, NSE = +0.15 and
increases SMIPS importance to 0.34. It was the best-performing model until the
climatology + soil feature work produced [`model4`](model4.md), which supersedes
it on every metric (see [`model4`](model4.md) and [`model6`](model6.md));
`model1` is retained as the baseline the improvement is measured against.

## Figures

Leave-site-out fit, feature importance and residual per-station bias across the
three catchments ([`plot_catchment.py`](../plot_catchment.py)):

![model1 leave-site-out results across Yanco, Kyeamba and Adelong](../figures/catchment_results.png)

Held-out predicted-vs-observed time series for every station
([`plot_per_station.py`](../plot_per_station.py)) — within-site correlation is
high; what remains is a per-station *level* offset:

![model1 per-station held-out time series, all 30 stations](../figures/catchment_per_station.png)

The four dense Kyeamba stations on their own (one SMIPS cell, no between-station
signal in the coarse predictor):

![model1 per-station held-out time series, Kyeamba](../figures/kyeamba_per_station.png)

That offset is **shrinkage toward the training mean** — dry stations predicted
too wet, wet ones too dry ([`plot_shrinkage.py`](../plot_shrinkage.py)); it is the
error the later feature work targets:

![Per-station bias is shrinkage toward the training mean](../figures/shrinkage_diagnostic.png)

The baseline 30 m field this model produces over Yanco
([`plot_downscale.py`](../plot_downscale.py)) — terrain structure resolved within
the coarse SMIPS cells:

![model1 downscaled 30 m field over Yanco](../figures/downscale_yanco.png)

---
<!-- NAV -->
[← Evaluation (leave-site-out)](evaluation.py.md) · [Index](../README.md) · [model2 · Linear →](model2.md)
<!-- /NAV -->
