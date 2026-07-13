"""model6 -- model4 plus antecedent-meteorology features.

Same estimator as :mod:`emt.model4` (regularised histogram gradient boosting),
with the feature set extended by SILO trailing-window meteorology
(:mod:`emt.antecedent`): how much rain has fallen and how the water balance
(P - PET) and evaporative demand have run over the **last week, month and year**::

    model4 features + rain_7/30/365 + ppet_30/365 + vpd_30 + rain_365_anom

Motivation: soil moisture depends on recent weather history that the current
SMIPS value and the pixel climatology do not fully separate -- chiefly the
*month-scale water balance* (how much rain minus evaporative demand a site has
banked recently). Adding these dynamic, national-grid, leakage-safe features
lowers the per-station level bias (median |bias| 4.0 -> ~3.5 %) at a small
cross-site gain (pooled NSE 0.354 -> 0.363 on the 30-station table).

Note on the windows: importance concentrates in the 30-day features
(``ppet_30``, ``vpd_30``); the 365-day features rank low because the SMIPS pixel
climatology already carries the long-term/drought state. They are retained here
(the full set scored marginally best) but contribute little.
"""
from __future__ import annotations

import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

from emt.antecedent import ANTECEDENT_VARS, add_antecedent
from emt.model4 import model as m4
from emt.model4.model import TARGET  # noqa: F401
from emt.evaluation import metrics, leave_site_out_cv as _cv  # noqa: F401

FEATURES = [*m4.FEATURES, *ANTECEDENT_VARS]


def build_estimator(**kwargs) -> HistGradientBoostingRegressor:
    """Boosting config for model6's (larger) feature set. Tuned by
    GroupKFold-on-station on the corrected lookback features: with more, weaker
    predictors the model wants **expressive** trees whose variance is controlled
    by per-split feature subsampling rather than by tiny leaves (the opposite of
    model4). Leave-site-out NSE 0.35 on 36 stations.

    Note the contrast with model4's ``max_leaf_nodes=3``: that extreme-
    regularisation optimum was an artefact of the earlier look-ahead leak; with
    leak-free features the tuned optimum is unlimited trees whose variance is
    controlled by aggressive per-split feature subsampling. A single-parameter
    grouped-CV sweep found skill flat for ``max_features`` in 0.15-0.3 and
    collapsing above 0.3; ``0.15`` (4 of 25 features per split) is the grouped-CV
    and leave-site-out peak (LOSO NSE 0.40; grouped-CV, which holds out more
    stations, is a more conservative 0.25).
    """
    params = dict(learning_rate=0.03, max_iter=200, max_leaf_nodes=None,
                  min_samples_leaf=20, max_features=0.15,
                  l2_regularization=1.0, random_state=0)
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


def ensure_features(table: pd.DataFrame) -> pd.DataFrame:
    """model4's features (SMIPS climatology + soil) plus antecedent meteorology."""
    table = m4.ensure_features(table)
    if any(v not in table.columns for v in ANTECEDENT_VARS):
        from emt.insitu.coordinates import COORDS_CACHE
        coords = pd.read_csv(COORDS_CACHE)
        times = pd.to_datetime(table["time"])
        table = add_antecedent(table, coords, times.min().date(), times.max().date())
    return table


def fit(table: pd.DataFrame, estimator=None):
    est = estimator if estimator is not None else build_estimator()
    sub = ensure_features(table).dropna(subset=FEATURES + [TARGET])
    est.fit(sub[FEATURES], sub[TARGET])
    return est


def leave_site_out_cv(table: pd.DataFrame, group_col: str = "station") -> dict:
    return _cv(ensure_features(table), FEATURES, build_estimator, group_col=group_col)


def feature_importance(model, table: pd.DataFrame, n_repeats: int = 10) -> pd.Series:
    from sklearn.inspection import permutation_importance
    sub = ensure_features(table).dropna(subset=FEATURES + [TARGET])
    r = permutation_importance(model, sub[FEATURES], sub[TARGET],
                               n_repeats=n_repeats, random_state=0)
    return pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)


if __name__ == "__main__":
    import sys
    table = (pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith(".parquet")
             else pd.read_csv(sys.argv[1]))
    cv = leave_site_out_cv(table)
    p, ps = cv["pooled"], cv["per_site"]
    print("pooled:", {k: round(v, 3) if isinstance(v, float) else v for k, v in p.items()})
    print(f"per-station NSE>0: {(ps['nse'] > 0).sum()}/{len(ps)} "
          f"(median {ps['nse'].median():.2f}); median |bias| {ps['bias'].abs().median():.2f}")
