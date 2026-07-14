"""Fit-once model persistence.

Every model in the handout is expensive to fit (model6's boosting especially),
yet figures, feature importance and inference each need a fitted estimator. This
caches a fitted model to disk so it is trained **once** and loaded thereafter —
for reuse across figures and for property/AOI inference runs (where you never
want to retrain).

    est = fit_cached(model6, table, "model6")   # fits + saves first time, loads after
    est = load_model("model6")                    # None if not cached
"""
from __future__ import annotations

from pathlib import Path

import joblib

MODELS_DIR = Path("data/models")


def _path(name: str) -> Path:
    return MODELS_DIR / f"{name}.joblib"


def save_model(estimator, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(name)
    joblib.dump(estimator, p)
    return p


def load_model(name: str):
    """Return the cached estimator, or ``None`` if it hasn't been saved."""
    p = _path(name)
    return joblib.load(p) if p.exists() else None


def loso_cached(module, table, name: str, reload: bool = False, verbose: bool = True):
    """Leave-site-out out-of-fold predictions for ``module``, computed once.

    Returns a DataFrame ``[station, time, <target>, pred]`` and caches it to
    ``data/<name>_loso_predictions.csv`` so figures never re-run the CV. Prints
    per-station progress on the first (uncached) run.
    """
    import numpy as np
    import pandas as pd

    path = Path(f"data/{name}_loso_predictions.csv")
    if not reload and path.exists():
        return pd.read_csv(path, parse_dates=["time"])

    target = module.TARGET
    sub = (module.ensure_features(table)
           .dropna(subset=list(module.FEATURES) + [target]).reset_index(drop=True))
    sub["time"] = pd.to_datetime(sub["time"])
    out = sub[["station", "time", target]].copy()
    out["pred"] = np.nan
    stations = list(sub["station"].unique())
    for i, stn in enumerate(stations, 1):
        te = (sub["station"] == stn).values
        est = module.build_estimator()
        est.fit(sub.loc[~te, module.FEATURES], sub.loc[~te, target])
        out.loc[te, "pred"] = est.predict(sub.loc[te, module.FEATURES])
        if verbose:
            print(f"  {name} LOSO [{i}/{len(stations)}] {stn}", flush=True)
    out.to_csv(path, index=False)
    return out


def fit_cached(module, table, name: str, reload: bool = False):
    """Load the cached fit of ``module`` named ``name``; fit + save if absent.

    ``module`` is a model package (``emt.model6.model`` etc.) exposing ``fit``.
    Pass ``reload=True`` to force a refit (e.g. after changing the estimator).
    """
    if not reload:
        est = load_model(name)
        if est is not None:
            return est
    est = module.fit(table)
    save_model(est, name)
    return est
