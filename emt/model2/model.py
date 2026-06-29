"""model2 -- linear regression on SMIPS + terrain (an extrapolating baseline).

Same target and features as :mod:`emt.model1.model`, but the estimator is a
linear model (ridge) on standardised features rather than a Random Forest::

    sm_rootzone_pct ~ smips_totalbucket + terrain(...) + doy_sin + doy_cos

Motivation: model1 (RF) shrinks predictions toward the training mean because a
tree leaf can only average — it cannot extrapolate. A linear model *can*
extrapolate, so it tests whether the per-station level bias is partly the
estimator's inability to predict beyond the training range. Features are
standardised (`StandardScaler`) because, unlike trees, a linear model is not
scale-invariant.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emt.covariates import TERRAIN_VARS
from emt.features import SMIPS_COL
from emt.evaluation import metrics, leave_site_out_cv as _cv, TARGET  # noqa: F401

# Same predictor set as model1 (see that module for the lat/lon exclusion).
FEATURES = [SMIPS_COL, *TERRAIN_VARS, "doy_sin", "doy_cos"]


def build_estimator(**kwargs):
    """StandardScaler -> Ridge. Pass ``alpha=`` to tune regularisation."""
    alpha = kwargs.pop("alpha", 1.0)
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, **kwargs))


def fit(table: pd.DataFrame, estimator=None):
    """Fit the model on the full training table."""
    est = estimator if estimator is not None else build_estimator()
    sub = table.dropna(subset=FEATURES + [TARGET])
    est.fit(sub[FEATURES], sub[TARGET])
    return est


def leave_site_out_cv(table: pd.DataFrame, group_col: str = "station") -> dict:
    """Leave-site-out CV for this model (delegates to :mod:`emt.evaluation`)."""
    return _cv(table, FEATURES, build_estimator, group_col=group_col)


def feature_importance(model) -> pd.Series:
    """Standardised-coefficient magnitudes as importances (features are scaled,
    so |coef| is directly comparable). ``model`` is the fitted pipeline."""
    coef = model[-1].coef_
    return pd.Series(abs(coef), index=FEATURES).sort_values(ascending=False)


def coefficients(model) -> pd.Series:
    """Signed standardised coefficients (sign shows direction of effect)."""
    return pd.Series(model[-1].coef_, index=FEATURES)


if __name__ == "__main__":
    import sys
    table = (pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith(".parquet")
             else pd.read_csv(sys.argv[1]))
    cv = leave_site_out_cv(table)
    print("pooled:", {k: round(v, 3) if isinstance(v, float) else v for k, v in cv["pooled"].items()})
    print("\ncoefficients (standardised):\n", coefficients(fit(table)).round(3).to_string())
