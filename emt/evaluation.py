"""Shared evaluation harness: metrics and leave-site-out cross-validation.

Estimator-agnostic, so every model under ``emt/model1``, ``emt/model2``, ... is
scored identically. A model package supplies its feature list and an estimator
factory; everything here is the same across models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

TARGET = "sm_rootzone_pct"


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


def leave_site_out_cv(table: pd.DataFrame, features: list[str], estimator_factory,
                      group_col: str = "station", target: str = TARGET) -> dict:
    """Leave-one-site-out spatial cross-validation for any estimator.

    For each group (station), train a fresh ``estimator_factory()`` on every
    other group and predict the held-out one.

    Returns:
        ``{"pooled": dict, "per_site": DataFrame, "predictions": DataFrame}``.
    """
    sub = table.dropna(subset=features + [target]).reset_index(drop=True)
    groups = sub[group_col].values
    if len(np.unique(groups)) < 2:
        raise ValueError("leave-site-out CV needs >= 2 groups in the table.")

    logo = LeaveOneGroupOut()
    preds = np.full(len(sub), np.nan)
    for train_idx, test_idx in logo.split(sub[features], sub[target], groups):
        est = estimator_factory()
        est.fit(sub.iloc[train_idx][features], sub.iloc[train_idx][target])
        preds[test_idx] = est.predict(sub.iloc[test_idx][features])

    out = sub[[group_col, "time", target]].copy()
    out["pred"] = preds
    per_site = (out.groupby(group_col)
                   .apply(lambda g: pd.Series(metrics(g[target], g["pred"])),
                          include_groups=False)
                   .reset_index())
    pooled = metrics(out[target], out["pred"])
    return {"pooled": pooled, "per_site": per_site, "predictions": out}
