# `model3`: gradient-boosted trees on SMIPS + terrain

<!-- NAV -->
[← model2 · Linear](model2.md) · [Index](../README.md) · [model4 · Climatology + soil →](model4.md)
<!-- /NAV -->

Source: [`../../emt/model3/model.py`](../../emt/model3/model.py)

Same target and features as [`model1`](model1.md) and [`model2`](model2.md), but
the estimator is `HistGradientBoostingRegressor` — histogram-binned gradient
boosting. It was built to test a middle ground: a tree ensemble that, unlike the
Random Forest, reduces bias by fitting each tree to the previous trees' residuals.

| Function | Role |
|---|---|
| `build_estimator(**kw)` | `HistGradientBoostingRegressor` (stock defaults, fixed `random_state`) |
| `fit(table)` | Fit on the full table |
| `leave_site_out_cv(table)` | Spatial cross-validation (via `emt.evaluation`) |
| `feature_importance(model, table)` | Permutation importance (needs the table — see below) |

Scoring (`metrics`, `leave_site_out_cv`) lives in shared
[`evaluation.py`](evaluation.py.md) so every model is graded identically.

## Why gradient boosting

The two existing models sit at opposite ends of a limitation:

- [`model1`](model1.md) (Random Forest) is a **bagging** ensemble — independent
  trees averaged together — which captures the nonlinear SMIPS × terrain relation
  but shrinks predictions toward the training mean.
- [`model2`](model2.md) (linear) **can** extrapolate but fits a single global
  hyperplane, so it extracts less of the between-site signal and scores lowest on
  cross-site skill.

Gradient boosting is the untested middle: an **additive** ensemble where each
small tree corrects the residual of the ones before it, lowering bias for a given
amount of averaging while still modelling nonlinearity. It remains tree-based, so
like `model1` it cannot predict outside the training range.

`HistGradientBoostingRegressor` (histogram binning) is used for speed on the
≈40,600-row table. Its stock defaults are used unchanged: on this table they give
pooled NSE +0.123, statistically tied with a hand-tuned configuration
(`learning_rate=0.05, max_iter=500, max_leaf_nodes=31` → +0.124) that runs ≈6×
slower, so tuning was not warranted. `lat`/`lon` are excluded for the same reason
as the other models (they would let the trees memorise station identity and
inflate leave-site-out skill); `station` is the CV grouping only.

## Feature importance

Unlike the Random Forest, histogram gradient boosting exposes **no impurity-based
importances**, so `feature_importance(model, table)` uses **permutation
importance** — the drop in score when each feature is shuffled — which requires
the data the model was fit on (hence the extra `table` argument). On the
three-catchment table the ranking is slope (0.73), `smips_totalbucket` (0.59),
elevation (0.28), then seasonality and the remaining terrain terms — the same top
predictors as the Random Forest, in a slightly different order.

## Evaluation

On the three-catchment table (30 stations, leave-site-out):

| Metric | model3 (gradient boosting) |
|---|---|
| Pooled NSE / r | +0.12 / 0.47 |
| Per-station NSE > 0 | 8 / 30 |
| Median per-station r | 0.74 |
| Median per-station ubRMSE | 3.5 % |

At stock settings, gradient boosting lands **between** the linear model and the
Random Forest on cross-site skill (+0.12 vs +0.15), clearing positive
per-station NSE at only 8 of 30 stations — though its per-station ubRMSE (3.5 %)
is the best of the three, confirming that dynamics are reproduced well and the
residual is a per-station level bias.

**Postscript:** this "ceiling" was later broken — not by tuning this model, but
by a feature (the SMIPS pixel climatology) plus much heavier regularisation of
this same estimator family. [`model4`](model4.md) is that configuration
(pooled NSE +0.35); `model3` is retained as the stock-defaults boosting
reference point. See [`model6`](model6.md), the recommended model.

---
<!-- NAV -->
[← model2 · Linear](model2.md) · [Index](../README.md) · [model4 · Climatology + soil →](model4.md)
<!-- /NAV -->
