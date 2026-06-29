"""model1 -- Random Forest downscaling regressor (the baseline approach).

A Random Forest learns the root-zone soil-moisture target from the coarse SMIPS
value plus fine terrain covariates and seasonality::

    sm_rootzone_pct ~ smips_totalbucket + terrain(...) + doy_sin + doy_cos

Applied per 30 m pixel (see :mod:`emt.downscale`) it sharpens the ~1 km SMIPS
field using the terrain structure within each coarse cell.

``lat``/``lon`` are deliberately NOT features -- they would let the forest
memorise station location and inflate skill under leave-site-out CV. Station id
is the CV grouping only. Skill is the leave-site-out estimate (train on all
sites but one, predict the held-out site).

Random Forests cannot extrapolate beyond the training range (each leaf predicts
an average), so this model shrinks predictions toward the training mean -- the
limitation :mod:`emt.model2` was built to probe.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from emt.covariates import TERRAIN_VARS
from emt.features import SMIPS_COL
from emt.evaluation import metrics, leave_site_out_cv as _cv, TARGET  # noqa: F401

# SLGA soil covariates were tested as features and reverted: they act as a
# near-unique per-station identifier and degraded leave-site-out skill
# (pooled NSE +0.15 -> +0.03). The loader emt/slga.py is kept for a future
# larger-network attempt. See the handout.
FEATURES = [SMIPS_COL, *TERRAIN_VARS, "doy_sin", "doy_cos"]


def build_estimator(**kwargs) -> RandomForestRegressor:
    """A sensible default Random Forest for this problem."""
    params = dict(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0)
    params.update(kwargs)
    return RandomForestRegressor(**params)


def fit(table: pd.DataFrame, estimator=None):
    """Fit the model on the full training table."""
    est = estimator if estimator is not None else build_estimator()
    sub = table.dropna(subset=FEATURES + [TARGET])
    est.fit(sub[FEATURES], sub[TARGET])
    return est


def leave_site_out_cv(table: pd.DataFrame, group_col: str = "station") -> dict:
    """Leave-site-out CV for this model (delegates to :mod:`emt.evaluation`)."""
    return _cv(table, FEATURES, build_estimator, group_col=group_col)


def feature_importance(model: RandomForestRegressor) -> pd.Series:
    """Random Forest impurity-based feature importances, sorted."""
    return pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)


if __name__ == "__main__":
    import sys
    table = (pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith(".parquet")
             else pd.read_csv(sys.argv[1]))
    cv = leave_site_out_cv(table)
    print("pooled:", {k: round(v, 3) if isinstance(v, float) else v for k, v in cv["pooled"].items()})
    print("\nfeature importance:\n", feature_importance(fit(table)).round(3).to_string())
