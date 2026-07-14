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
