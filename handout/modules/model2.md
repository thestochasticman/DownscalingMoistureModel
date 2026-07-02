# `model2`: linear regression on SMIPS + terrain

<!-- NAV -->
[← model1 · Random Forest](model1.md) · [Index](../README.md) · [model3 · Gradient boosting →](model3.md)
<!-- /NAV -->

Source: [`../../emt/model2/model.py`](../../emt/model2/model.py)

Same target and features as [`model1`](model1.md), but the estimator is a linear
model (`StandardScaler → Ridge`) instead of a Random Forest. Features are
standardised because, unlike trees, a linear model is not scale-invariant.

| Function | Role |
|---|---|
| `build_estimator(alpha=1.0)` | `StandardScaler → Ridge` pipeline |
| `fit(table)` | Fit on the full table |
| `leave_site_out_cv(table)` | Spatial cross-validation (via `emt.evaluation`) |
| `feature_importance(model)` / `coefficients(model)` | Standardised coefficient magnitudes / signed values |

## What it offers

- **Interpretability.** It returns signed, standardised coefficients with sensible
  physics: slope strongly negative (steeper terrain is drier), SMIPS positive
  (wetter coarse cell is wetter), elevation and HLI positive. That is a direct
  explanation of what drives root-zone moisture, not a black box.
- **Per-station tracking.** It puts 12/30 stations at positive per-station NSE
  (vs 7/30 for the Random Forest), with a better median (−0.34 vs −0.61).
- **Speed and simplicity.** A scaler plus a linear solve is orders of magnitude
  cheaper to fit and to apply over a large area than 300 trees.

## Where the Random Forest leads

On **cross-site ranking** (the metric that governs a spatial map) the Random
Forest is better: pooled NSE +0.15 vs +0.02, r 0.53 vs 0.36. A single global
hyperplane extracts less of the limited between-site signal than the forest's
nonlinear splits. The two models are different operating points on a
cross-site ↔ per-station tradeoff (see
[Model comparison](../README.md#model-comparison)), so the better choice depends
on the objective: the Random Forest for the downscaled map,
the linear model for interpretable per-site analysis.

## Strengthening was tested

Richer linear models do not improve the headline metric. Tuning the ridge penalty
(`RidgeCV`) gave an identical result; a robust Huber loss and a few targeted
SMIPS×terrain interactions moved further toward per-station tracking at the cost
of cross-site skill; and full pairwise/quadratic features **exploded** under
leave-site-out (NSE ~ −36000), because polynomial terms extrapolate violently to
unseen feature combinations. The plain ridge is therefore the robust, correct
linear configuration.

---

---
<!-- NAV -->
[← model1 · Random Forest](model1.md) · [Index](../README.md) · [model3 · Gradient boosting →](model3.md)
<!-- /NAV -->
