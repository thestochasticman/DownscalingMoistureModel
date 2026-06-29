# `model2`: linear regression on SMIPS + terrain

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

## Motivation

`model1` (RF) shrinks predictions toward the training mean because a tree leaf
can only average, so it cannot extrapolate. A linear model *can* extrapolate, so
this was built to test whether the per-station level bias is partly the
estimator's inability to predict beyond the training range.

## Result: it did not help

The hypothesis did not hold. The linear model is worse than the Random Forest on
the metrics that matter for a spatial product (see
[Model comparison](../README.md#model-comparison)): pooled NSE +0.02 vs +0.15,
r 0.36 vs 0.53, and a *steeper* shrinkage slope (−0.98 vs −0.61). A single global
hyperplane is less able to extract the limited between-site signal than the
Random Forest's nonlinear splits, so it collapses the cross-site ranking. The
shrinkage is driven by features that do not identify station baselines, not by
the estimator, consistent with the soil experiment and with too-few-sites as
the binding constraint. The standardised coefficients are physically plausible
(slope negative: steeper is drier; SMIPS positive), so the model is sound; it is
simply outmatched on this nonlinear, site-limited problem. Retained as a
documented comparison.
