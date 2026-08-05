"""model10 -- the hybrid: model6's features plus the process model's state.

The two tracks fail in *different places*. Under blocked validation
[model6](../../handout/modules/model6.md) collapses at M2 (bias +6.7 %) where
[model8](../../handout/modules/model8.md) scores +0.74; model8 fails at Adelong
where model6 nearly survives on its rainfall features. That complementarity is
the standing argument, flagged since model7, for feeding one into the other.

model10 is the cheap direction of that: take model6's feature set and add
**bucket storage** -- the model8 water balance's state on the day, in mm --
as one more predictor::

    model6 features + bucket_storage

The storage is produced by running the *fitted* model8 bucket on SILO
rain/PET. It is therefore a national, backward-looking covariate computable at
any pixel, exactly like the others: it carries no in-situ information and
cannot memorise station identity. What it adds is a *physically integrated*
summary of the recent water balance -- something model6's trailing rain and
P-PET windows approximate linearly, and the bucket does with a nonlinear
store that saturates and drains.

Measured against the target it is a comparable predictor to SMIPS itself
(r 0.45 vs 0.52), while being available on days and in places SMIPS is not.

**Direction of the hybrid.** This is ML-consumes-physics. The other direction
-- assimilating SMIPS into the bucket as an observation -- is the harder and
probably more interesting one, and remains open.

**Inference note.** Applying model10 to a map needs bucket storage per pixel,
which :mod:`emt.model8.predict` already computes on the SILO forcing grid;
wiring that into :mod:`emt.downscale` is not done, so model10 is evaluated but
not shipped for inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from emt.evaluation import TARGET, leave_site_out_cv as _cv, metrics  # noqa: F401
from emt.model6 import model as m6
from emt.model6.model import build_estimator  # noqa: F401  (same estimator)

BUCKET_VARS = ["bucket_storage"]
FEATURES = [*m6.FEATURES, *BUCKET_VARS]


def add_bucket_storage(table: pd.DataFrame, model_name: str = "model8") -> pd.DataFrame:
    """Attach ``bucket_storage`` (mm) from the fitted process model.

    Runs the shipped bucket over the training forcing store with each station's
    own capacity scaling, then indexes it by (station, day). Deterministic and
    leakage-free: the bucket saw only rain and PET.
    """
    if "bucket_storage" in table.columns:
        return table
    from emt.model7.model import load_forcing, _step_loop
    from emt.model8 import model as m8
    from emt.persist import load_model

    est = load_model(model_name)
    if est is None:
        raise FileNotFoundError(f"no fitted {model_name} at data/models/{model_name}.joblib")
    f = load_forcing()
    smax, alpha, k = est.bucket_params
    cap = m8.awc_capacity().reindex(f.stations).to_numpy(dtype=float)
    cap = cap / est.cap_train_mean_ if getattr(est, "cap_train_mean_", None) else 1.0
    storage = _step_loop(f.rain, f.pet, smax * cap, alpha, k)

    t = table.copy()
    t["time"] = pd.to_datetime(t["time"])
    col = {s: j for j, s in enumerate(f.stations)}
    rows = ((t["time"] - f.times[0]).dt.days).to_numpy()
    cols = t["station"].map(col).to_numpy()
    ok = pd.notna(cols) & (rows >= 0) & (rows < len(f.times))
    vals = np.full(len(t), np.nan)
    vals[ok] = storage[rows[ok], cols[ok].astype(int)]
    t["bucket_storage"] = vals
    return t


def ensure_features(table: pd.DataFrame) -> pd.DataFrame:
    """model6's features plus the bucket state."""
    return add_bucket_storage(m6.ensure_features(table))


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
