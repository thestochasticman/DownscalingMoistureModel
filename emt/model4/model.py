"""model4 -- regularised gradient boosting + SMIPS climatology + soil (the
improved model).

Same target as models 1-3, but with two feature additions and a heavily
regularised estimator, found by a systematic leave-site-out search::

    sm_rootzone_pct ~ smips_totalbucket + terrain(...) + doy_sin + doy_cos
                      + smips_mean_px + smips_std_px + smips_anom + smips_z
                      + soil_clay + soil_sand + soil_awc + soil_bdw

The three ingredients (each validated independently; see the handout):

* **SMIPS pixel climatology** (:func:`emt.features.add_smips_climatology`):
  the pixel's long-term SMIPS mean/std plus the day's anomaly/z-score. This
  factors the coarse predictor into a static *level* and a dynamic *departure*
  -- supplying the local-baseline signal whose absence caused the per-station
  bias in models 1-3. Derived from SMIPS alone: available at every pixel at
  inference, cannot memorise stations. Largest single lever (+0.08 pooled NSE).
* **Extreme regularisation**: ``max_leaf_nodes=3`` with a slow learning rate.
  Skill rises monotonically as trees shrink (31 -> 3), peaking at 3 -- tiny
  trees cannot memorise site quirks, forcing transferable structure.
* **SLGA soil** (:mod:`emt.slga`): with the climatology anchoring the level,
  soil texture adds real signal (best per-station profile). NOTE: soil HURT
  when added to models 1-3 -- that negative result was conditional on the
  missing level feature, not absolute.

Skill (30 stations, 2006-2010): pooled LOSO NSE 0.35 vs 0.15 (model1), 14/30
stations at positive per-station NSE (vs 7/30), median per-station NSE -0.07
(vs -0.56). Leave-region-out: -0.72 (model1) -> +0.12, all regional biases
shrink. ``lat``/``lon`` remain excluded; ``station`` is the CV grouping only.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from emt.covariates import TERRAIN_VARS
from emt.features import SMIPS_COL, CLIM_VARS, add_smips_climatology, add_soil_covariates
from emt.slga import SOIL_VARS
from emt.evaluation import metrics, leave_site_out_cv as _cv, TARGET  # noqa: F401

FEATURES = [SMIPS_COL, *TERRAIN_VARS, "doy_sin", "doy_cos",
            *CLIM_VARS, *SOIL_VARS]


def build_estimator(**kwargs) -> HistGradientBoostingRegressor:
    """Heavily regularised boosting: tiny trees (3 leaves), slow learning."""
    params = dict(learning_rate=0.03, max_iter=800, max_leaf_nodes=3,
                  l2_regularization=1.0, random_state=0)
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


def ensure_features(table: pd.DataFrame) -> pd.DataFrame:
    """Derive any missing model4 feature columns on a standard training table.

    Climatology comes from the table's own SMIPS series; soil is sampled from
    the (cached) per-station SLGA rasters using the repo's station-coordinates
    cache. A table that already has the columns passes through unchanged.
    """
    if any(v not in table.columns for v in CLIM_VARS):
        table = add_smips_climatology(table)
    if any(v not in table.columns for v in SOIL_VARS):
        from emt.insitu.coordinates import COORDS_CACHE
        coords = pd.read_csv(COORDS_CACHE)
        times = pd.to_datetime(table["time"])
        table = add_soil_covariates(table, coords,
                                    times.min().date(), times.max().date())
    return table


def fit(table: pd.DataFrame, estimator=None):
    """Fit the model on the full training table (derives features if missing)."""
    est = estimator if estimator is not None else build_estimator()
    sub = ensure_features(table).dropna(subset=FEATURES + [TARGET])
    est.fit(sub[FEATURES], sub[TARGET])
    return est


def leave_site_out_cv(table: pd.DataFrame, group_col: str = "station") -> dict:
    """Leave-site-out CV for this model (delegates to :mod:`emt.evaluation`)."""
    return _cv(ensure_features(table), FEATURES, build_estimator, group_col=group_col)


def feature_importance(model, table: pd.DataFrame,
                       n_repeats: int = 10) -> pd.Series:
    """Permutation importance (no impurity importances for HGB; needs data)."""
    sub = ensure_features(table).dropna(subset=FEATURES + [TARGET])
    r = permutation_importance(model, sub[FEATURES], sub[TARGET],
                               n_repeats=n_repeats, random_state=0)
    return pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)


if __name__ == "__main__":
    import sys
    table = (pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith(".parquet")
             else pd.read_csv(sys.argv[1]))
    cv = leave_site_out_cv(table)
    print("pooled:", {k: round(v, 3) if isinstance(v, float) else v for k, v in cv["pooled"].items()})
    per = cv["per_site"]
    print(f"per-station NSE>0: {(per['nse'] > 0).sum()}/{len(per)} "
          f"(median {per['nse'].median():.2f})")
