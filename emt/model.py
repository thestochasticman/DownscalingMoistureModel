"""Stage 5 -- the downscaling regression model.

A Random Forest learns the root-zone soil-moisture target from the coarse SMIPS
value plus fine terrain covariates and seasonality::

    sm_rootzone_pct ~ smips_totalbucket + terrain(...) + doy_sin + doy_cos

Applied per 30 m pixel (stage 6) it sharpens the ~1 km SMIPS field using the
terrain structure within each coarse cell.

``lat``/``lon`` are deliberately NOT features -- they would let the forest
memorise station location and inflate skill under leave-site-out CV. Station id
is used only as the CV grouping. The honest skill estimate is
:func:`leave_site_out_cv` (train on all sites but one, predict the held-out
site), since at inference the model meets terrain/SMIPS combinations from
unseen locations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

from emt.covariates import TERRAIN_VARS
from emt.features import SMIPS_COL

TARGET = "sm_rootzone_pct"
FEATURES = [SMIPS_COL, *TERRAIN_VARS, "doy_sin", "doy_cos"]


def build_estimator(**kwargs) -> RandomForestRegressor:
    """A sensible default Random Forest for this problem."""
    params = dict(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0)
    params.update(kwargs)
    return RandomForestRegressor(**params)


def metrics(y_true, y_pred) -> dict:
    """Standard soil-moisture validation metrics.

    Returns rmse, ubrmse (bias-removed RMSE), bias (pred-obs), r (Pearson),
    nse (Nash-Sutcliffe efficiency, 1 - SS_res/SS_tot), and n. NSE > 0 means the
    prediction is more skilful than the observed mean; NSE = 1 is perfect. NSE is
    identical to the coefficient of determination, also returned as ``r2``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    n = y_true.size
    if n < 2:
        return dict(rmse=np.nan, ubrmse=np.nan, bias=np.nan, r=np.nan,
                    nse=np.nan, r2=np.nan, n=n)
    err = y_pred - y_true
    bias = float(err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    ubrmse = float(np.sqrt(max(rmse ** 2 - bias ** 2, 0.0)))
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan  # Nash-Sutcliffe efficiency
    return dict(rmse=rmse, ubrmse=ubrmse, bias=bias, r=r, nse=nse, r2=nse, n=n)


def fit(table: pd.DataFrame, estimator: RandomForestRegressor | None = None
        ) -> RandomForestRegressor:
    """Fit the model on the full training table."""
    est = estimator if estimator is not None else build_estimator()
    sub = table.dropna(subset=FEATURES + [TARGET])
    est.fit(sub[FEATURES], sub[TARGET])
    return est


def leave_site_out_cv(table: pd.DataFrame, estimator_factory=build_estimator,
                      group_col: str = "station") -> dict:
    """Leave-one-site-out spatial cross-validation.

    For each station, train on every other station and predict the held-out
    one. Returns pooled metrics, per-station metrics, and the out-of-fold
    predictions.

    Returns:
        ``{"pooled": dict, "per_site": DataFrame, "predictions": DataFrame}``.
    """
    sub = table.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    groups = sub[group_col].values
    if len(np.unique(groups)) < 2:
        raise ValueError("leave-site-out CV needs >= 2 stations in the table.")

    logo = LeaveOneGroupOut()
    preds = np.full(len(sub), np.nan)
    for train_idx, test_idx in logo.split(sub[FEATURES], sub[TARGET], groups):
        est = estimator_factory()
        est.fit(sub.iloc[train_idx][FEATURES], sub.iloc[train_idx][TARGET])
        preds[test_idx] = est.predict(sub.iloc[test_idx][FEATURES])

    out = sub[[group_col, "time", TARGET]].copy()
    out["pred"] = preds

    per_site = (out.groupby(group_col)
                   .apply(lambda g: pd.Series(metrics(g[TARGET], g["pred"])),
                          include_groups=False)
                   .reset_index())
    pooled = metrics(out[TARGET], out["pred"])
    return {"pooled": pooled, "per_site": per_site, "predictions": out}


def feature_importance(model: RandomForestRegressor) -> pd.Series:
    """Random Forest feature importances as a sorted Series."""
    return pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)


if __name__ == "__main__":
    # Expects a prebuilt training table parquet/csv path as argv[1].
    import sys
    table = pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith(".parquet") else pd.read_csv(sys.argv[1])
    cv = leave_site_out_cv(table)
    print("pooled:", {k: round(v, 3) if isinstance(v, float) else v for k, v in cv["pooled"].items()})
    print(cv["per_site"].round(3).to_string(index=False))
    print("\nfeature importance:\n", feature_importance(fit(table)).round(3).to_string())
