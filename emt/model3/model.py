"""model3 -- gradient-boosted trees (the between-RF-and-linear middle ground).

Same target and features as :mod:`emt.model1` / :mod:`emt.model2`::

    sm_rootzone_pct ~ smips_totalbucket + terrain(...) + doy_sin + doy_cos

Motivation (from the handout's future-work list): model1 (Random Forest) is a
*bagging* ensemble -- independent trees whose predictions are averaged, which
shrinks toward the training mean. A linear model (model2) can extrapolate but
cannot capture the nonlinear SMIPS x terrain interaction, so it scores lower.
Gradient boosting is the untested middle ground: an *additive* ensemble that fits
each tree to the residual of the ones before it, reducing bias for a given amount
of averaging while still modelling nonlinearity. It is still tree-based, so like
model1 it cannot predict outside the training range.

``HistGradientBoostingRegressor`` (histogram-binned boosting) is used for speed on
the ~40k-row table; its defaults are strong here (see the handout module note) so
no manual tuning is applied. ``lat``/``lon`` are excluded for the same reason as
the other models -- they would let the trees memorise station identity and
inflate leave-site-out skill; ``station`` is the CV grouping only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from emt.covariates import TERRAIN_VARS
from emt.features import SMIPS_COL
from emt.evaluation import metrics, leave_site_out_cv as _cv, TARGET  # noqa: F401

# Same predictor set as model1/model2 (see model1 for the lat/lon exclusion).
FEATURES = [SMIPS_COL, *TERRAIN_VARS, "doy_sin", "doy_cos"]


def build_estimator(**kwargs) -> HistGradientBoostingRegressor:
    """Histogram gradient boosting. Stock defaults; ``random_state`` fixed for
    reproducibility. Pass e.g. ``learning_rate=``/``max_iter=`` to override."""
    params = dict(random_state=0)
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


def fit(table: pd.DataFrame, estimator=None):
    """Fit the model on the full training table."""
    est = estimator if estimator is not None else build_estimator()
    sub = table.dropna(subset=FEATURES + [TARGET])
    est.fit(sub[FEATURES], sub[TARGET])
    return est


def leave_site_out_cv(table: pd.DataFrame, group_col: str = "station") -> dict:
    """Leave-site-out CV for this model (delegates to :mod:`emt.evaluation`)."""
    return _cv(table, FEATURES, build_estimator, group_col=group_col)


def feature_importance(model, table: pd.DataFrame | None = None,
                       n_repeats: int = 10) -> pd.Series:
    """Permutation feature importance, sorted.

    Histogram gradient boosting has no impurity-based importances (unlike the
    Random Forest in model1), so importance is measured by permutation: the drop
    in score when each feature is shuffled. This requires data, so pass the
    ``table`` the model was fit on. Returns mean importance per feature.
    """
    if table is None:
        raise ValueError("feature_importance for model3 needs the training table "
                         "(permutation importance); pass table=...")
    sub = table.dropna(subset=FEATURES + [TARGET])
    r = permutation_importance(model, sub[FEATURES], sub[TARGET],
                               n_repeats=n_repeats, random_state=0)
    return pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)


if __name__ == "__main__":
    import sys
    table = (pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith(".parquet")
             else pd.read_csv(sys.argv[1]))
    cv = leave_site_out_cv(table)
    print("pooled:", {k: round(v, 3) if isinstance(v, float) else v for k, v in cv["pooled"].items()})
    print("\nfeature importance (permutation):\n",
          feature_importance(fit(table), table).round(3).to_string())
