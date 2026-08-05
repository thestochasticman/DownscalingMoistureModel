"""model7 -- a process model: daily bucket water balance, no machine learning.

Every earlier model is a statistical estimator over covariates. model7 instead
*simulates* the root-zone store directly: a single-layer bucket over the 0-90 cm
profile, forced only by SILO daily rain and Morton potential ET, stepped daily::

    S' = clip(S + P - AET - k*S, 0, smax)        AET = PET * min(1, S/(alpha*smax))

and read out as volumetric moisture through a linear observation operator::

    vwc% = theta_r + dtheta * S/smax

Water enters as rain, leaves as evapotranspiration (at the potential rate while
the bucket is above fraction ``alpha`` of capacity, linearly stressed below) and
as a linear recession ``k*S`` (deep drainage + lateral flow); whatever exceeds
capacity runs off. The five global parameters (``smax``, ``alpha``, ``k``,
``theta_r``, ``dtheta``) are *calibrated* on the training stations inside
``fit`` -- the process-model analogue of estimator training -- so the model runs
through the **identical leave-site-out harness** as models 1-6.

This is the foundation of the repo's **process-model track** -- predicting by
simulation with parameters you can read, no training table, and no dependence
on SMIPS (:mod:`emt.model8`, the same bucket with SLGA soil in the offset
stage, is the track's recommended configuration). By construction:

* **No SMIPS, no soil, no terrain in the state equation, no ML**: forcing in,
  physics through, moisture out.
* Parameters are global, so between-station level differences can come only
  from the forcing (~5 km SILO); cross-site level ranking needs the offset
  stage below.
* Optional per-station static offsets (e.g. terrain TWI/slope, passed as
  ``static``) shift the readout linearly -- the seat where national covariates
  add between-site structure without touching the physics.

The forcing store is continuous daily rain/PET per station from
``data/process_forcing_2005_2010.csv`` (SILO, fetched one year before the study
start so 2006 scoring follows a full spin-up year; simulation starts 2005-01-01
at half capacity). ``FEATURES = ["station", "time"]`` are *keys* into that
store, not covariates -- the harness's ``X`` tells the estimator which
station-days to compare, never what the weather was.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from emt.evaluation import TARGET, leave_site_out_cv as _cv, metrics  # noqa: F401

# Keys into the forcing store (see module docstring) -- deliberately no
# covariates: the process model sees only its own forcing.
FEATURES = ["station", "time"]

FORCING_CSV = Path("data/process_forcing_2005_2010.csv")

PARAM_NAMES = ("smax", "alpha", "k", "theta_r", "dtheta")
#             capacity  ET-stress  recession  dry VWC   wet-dry VWC range
# alpha > 1 is allowed: AET = PET*min(1, S/(alpha*smax)) then never reaches the
# potential rate, so large alpha doubles as an evaporative-efficiency scaling.
BOUNDS = ((50.0, 400.0), (0.05, 2.0), (5e-4, 0.2), (2.0, 25.0), (5.0, 45.0))
X0 = (150.0, 0.6, 0.02, 8.0, 25.0)

_RAIN, _PET = "daily_rain", "et_morton_potential"


# --------------------------------------------------------------------------- #
# Simulation core
# --------------------------------------------------------------------------- #
def _step_loop(rain: np.ndarray, pet: np.ndarray, smax: np.ndarray, alpha: float,
               k: float) -> np.ndarray:
    """Daily bucket recurrence over a (days, stations) forcing block -> storage.

    ``smax`` is per-station (a constant vector in the base configuration).
    """
    n_days, n_st = rain.shape
    out = np.empty((n_days, n_st))
    s = 0.5 * smax
    denom = alpha * smax
    for t in range(n_days):
        s = s + rain[t]
        aet = pet[t] * np.minimum(1.0, s / denom)
        s = np.minimum(np.maximum(s - aet - k * s, 0.0), smax)
        out[t] = s
    return out


try:                                    # numba ships with the PaddockTS stack
    from numba import njit
    _step_loop = njit(cache=True)(_step_loop)  # type: ignore[assignment]
except ImportError:                     # pragma: no cover - plain-numpy fallback
    pass


class Forcing:
    """Continuous daily rain/PET matrices, (n_days x n_stations)."""

    def __init__(self, frame: pd.DataFrame):
        f = frame.copy()
        f["time"] = pd.to_datetime(f["time"])
        self.times = pd.date_range(f["time"].min(), f["time"].max(), freq="D")
        rain = f.pivot_table(index="time", columns="station", values=_RAIN)
        pet = f.pivot_table(index="time", columns="station", values=_PET)
        rain = rain.reindex(self.times)
        pet = pet.reindex(self.times)
        self.stations = list(rain.columns)
        self._col = {s: j for j, s in enumerate(self.stations)}
        # SILO is gap-free in practice; guard interpolation for robustness.
        self.rain = rain.interpolate(limit=3).fillna(0.0).to_numpy()
        self.pet = pet.interpolate(limit=3).ffill().bfill().to_numpy()
        self._t0 = self.times[0]

    @classmethod
    def load(cls, path: str | Path = FORCING_CSV) -> "Forcing":
        return cls(pd.read_csv(path))

    def index(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(row, col) indices for the (station, time) keys in ``X``."""
        t = pd.to_datetime(X["time"]).dt.normalize()
        rows = ((t - self._t0).dt.days).to_numpy()
        if (rows < 0).any() or (rows >= len(self.times)).any():
            raise ValueError("observation outside the forcing period")
        cols = X["station"].map(self._col)
        if cols.isna().any():
            missing = sorted(X.loc[cols.isna(), "station"].unique())
            raise KeyError(f"no forcing for station(s): {missing}")
        return rows, cols.to_numpy(dtype=int)

    def vwc(self, x: np.ndarray, cap_rel: np.ndarray | None = None) -> np.ndarray:
        """Simulate all stations and map storage -> volumetric %.

        ``cap_rel`` (optional, per-station, mean 1 over the training stations)
        scales the capacity: ``smax_i = smax * cap_rel_i``. The readout keeps
        the *global* ``smax`` as its mm->% denominator, so higher-capacity
        stations can genuinely sit wetter -- this is where a soil covariate
        (e.g. SLGA AWC) injects between-station level structure physically.
        """
        smax, alpha, k, theta_r, dtheta = x
        smax_i = np.full(self.rain.shape[1], smax) if cap_rel is None else smax * cap_rel
        storage = _step_loop(self.rain, self.pet, smax_i, alpha, k)
        return theta_r + dtheta * storage / smax


_FORCING: Forcing | None = None


def load_forcing(path: str | Path = FORCING_CSV, reload: bool = False) -> Forcing:
    """Module-cached forcing store (one parse of the CSV per process)."""
    global _FORCING
    if _FORCING is None or reload:
        _FORCING = Forcing.load(path)
    return _FORCING


# --------------------------------------------------------------------------- #
# Estimator (sklearn-shaped so the shared harness applies unchanged)
# --------------------------------------------------------------------------- #
class BucketEstimator:
    """Bucket model with ``fit`` = parameter calibration on the training rows.

    ``static`` (optional): per-station statics (index = station, e.g. ``twi``,
    ``slope``). Fitted in a second stage: the bucket parameters are calibrated
    first, then the *per-station mean residuals* (one sample per training
    station) are ridge-regressed on the standardised statics, with the ridge
    penalty chosen by leave-one-out over the training stations. The held-out
    station's level offset then comes from its *own* covariate values, never
    from its observations. (Plain least squares here overfits the handful of
    training-station levels and transfers badly -- the ridge is essential.)

    ``capacity`` (optional): per-station relative bucket capacity (index =
    station; units cancel -- e.g. SLGA available water capacity). Normalised to
    mean 1 over the training stations and multiplied onto ``smax``, while the
    readout keeps the global ``smax`` denominator, so higher-capacity soils
    genuinely sit wetter: the physical route for a soil covariate to carry
    between-station level structure.
    """

    def __init__(self, forcing: Forcing | None = None,
                 static: pd.DataFrame | None = None,
                 capacity: pd.Series | None = None,
                 weight_fn=None,
                 n_starts: int = 3, maxfev: int = 800, seed: int = 0):
        self.forcing = forcing
        self.static = static
        self.capacity = capacity
        # Optional callable(X) -> per-row sample weights, applied whenever fit
        # is called without explicit sample_weight -- lets a configuration own
        # its weighting (e.g. model8's stratified weights) while running
        # unchanged through the shared est.fit(X, y) harness.
        self.weight_fn = weight_fn
        self.n_starts = n_starts
        self.maxfev = maxfev
        self.seed = seed
        self.params_: pd.Series | None = None

    # -- internals ---------------------------------------------------------- #
    def _forcing(self) -> Forcing:
        return self.forcing if self.forcing is not None else load_forcing()

    def _static_matrix(self, stations: pd.Series) -> np.ndarray:
        """Per-row standardised statics (z by the training stations' mean/std)."""
        vals = self.static.loc[stations, self._static_vars].to_numpy(dtype=float)
        return (vals - self._static_mean) / self._static_std

    # -- sklearn surface ---------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "BucketEstimator":
        f = self._forcing()
        rows, cols = f.index(X)
        yv = np.asarray(y, dtype=float)
        if sample_weight is None and self.weight_fn is not None:
            sample_weight = self.weight_fn(X)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        # Per-station capacity (e.g. SLGA AWC): normalised to mean 1 over the
        # *training* stations, applied to every station from its own value.
        # The training mean is kept for inference at new locations (their
        # capacity ratio is their own value over this same normaliser).
        self._cap_rel = None
        self.cap_train_mean_ = None
        if self.capacity is not None:
            cap = self.capacity.reindex(f.stations).to_numpy(dtype=float)
            train_mean = self.capacity.loc[X["station"].unique()].mean()
            self.cap_train_mean_ = float(train_mean)
            self._cap_rel = cap / float(train_mean)

        # Stage 1: calibrate the 5 bucket parameters (statics play no part).
        lo = np.array([b[0] for b in BOUNDS])
        hi = np.array([b[1] for b in BOUNDS])

        def loss(x: np.ndarray) -> float:
            if (x < lo).any() or (x > hi).any():
                return 1e6 + float(np.abs(np.clip(x, lo, hi) - x).sum())
            err = f.vwc(x, self._cap_rel)[rows, cols] - yv
            if w is None:
                return float(np.sqrt(np.nanmean(err ** 2)))
            m = np.isfinite(err)
            return float(np.sqrt(np.sum(w[m] * err[m] ** 2) / np.sum(w[m])))

        rng = np.random.default_rng(self.seed)
        x0 = np.array(X0)
        best = None
        for i in range(self.n_starts):
            start = x0 if i == 0 else np.clip(
                x0 * rng.uniform(0.6, 1.5, size=x0.shape), lo, hi)
            res = minimize(loss, start, method="Nelder-Mead",
                           options=dict(maxfev=self.maxfev, xatol=1e-3, fatol=1e-4))
            if best is None or res.fun < best.fun:
                best = res
        x = best.x
        names, values = list(PARAM_NAMES), list(x)
        self.rmse_ = float(best.fun)

        # Stage 2: ridge the per-station mean residual on the statics. With
        # sample weights, both the per-station residual and each station's vote
        # in the ridge are weighted, so the two stages optimise the same loss.
        if self.static is not None:
            from sklearn.linear_model import RidgeCV
            self._static_vars = list(self.static.columns)
            rf = pd.DataFrame({
                "station": np.asarray(X["station"]),
                "err": yv - f.vwc(x, self._cap_rel)[rows, cols],
                "w": np.ones(len(yv)) if w is None else w,
            })
            g = rf.groupby("station")
            resid = g.apply(lambda d: np.average(d["err"], weights=d["w"]),
                            include_groups=False)           # one sample/station
            # Unweighted: every station keeps an equal ridge vote (as before).
            # Weighted: a station's vote is its total sample weight.
            stn_w = (None if w is None else
                     (lambda s: (s / s.mean()).to_numpy(dtype=float))(
                         g["w"].sum().loc[resid.index]))
            ref = self.static.loc[resid.index, self._static_vars]
            self._static_mean = ref.mean().to_numpy(dtype=float)
            self._static_std = np.where(ref.std().to_numpy(dtype=float) > 0,
                                        ref.std().to_numpy(dtype=float), 1.0)
            Z = (ref.to_numpy(dtype=float) - self._static_mean) / self._static_std
            ridge = RidgeCV(alphas=np.logspace(-2, 3, 16)).fit(
                Z, resid.to_numpy(), sample_weight=stn_w)
            names += [f"c_{v}" for v in self._static_vars] + ["c_intercept"]
            values += list(ridge.coef_) + [float(ridge.intercept_)]
            self._ridge_alpha_ = float(ridge.alpha_)

        self.params_ = pd.Series(values, index=names)
        return self

    @property
    def bucket_params(self) -> tuple[float, float, float]:
        """``(smax, alpha, k)`` -- the state-equation parameters."""
        x = self.params_.to_numpy()
        return float(x[0]), float(x[1]), float(x[2])

    def readout(self, storage: np.ndarray, statics: np.ndarray | None = None) -> np.ndarray:
        """Map bucket storage (mm) to volumetric %, with the fitted offsets.

        The inference counterpart of :meth:`predict`: it takes *storage the
        caller simulated* (at a new location, over any period, on any grid)
        instead of indexing the training forcing store. ``statics`` is an
        ``(n, n_statics)`` array in the fitted ``_static_vars`` order,
        standardised here by the training mean/std.
        """
        x = self.params_.to_numpy()
        smax, theta_r, dtheta = x[0], x[3], x[4]
        vwc = theta_r + dtheta * np.asarray(storage, dtype=float) / smax
        if statics is not None:
            z = (np.asarray(statics, dtype=float) - self._static_mean) / self._static_std
            vwc = vwc + z @ x[5:-1] + x[-1]
        return vwc

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("fit before predict")
        rows, cols = self._forcing().index(X)
        x = self.params_.to_numpy()
        pred = self._forcing().vwc(x[:5], self._cap_rel)[rows, cols]
        if self.static is not None:
            pred = pred + self._static_matrix(X["station"]) @ x[5:-1] + x[-1]
        return pred


# --------------------------------------------------------------------------- #
# Model-package surface (same shape as model1..model6)
# --------------------------------------------------------------------------- #
def build_estimator(**kwargs) -> BucketEstimator:
    """Calibratable bucket model. ``static=DataFrame`` adds per-station terrain
    offsets; all other kwargs pass through to :class:`BucketEstimator`."""
    return BucketEstimator(**kwargs)


def ensure_features(table: pd.DataFrame) -> pd.DataFrame:
    """model7 needs only (station, time) keys -- present in every EMT table."""
    return table


def fit(table: pd.DataFrame, estimator: BucketEstimator | None = None) -> BucketEstimator:
    est = estimator if estimator is not None else build_estimator()
    sub = table.dropna(subset=FEATURES + [TARGET])
    est.fit(sub[FEATURES], sub[TARGET])
    return est


def leave_site_out_cv(table: pd.DataFrame, group_col: str = "station",
                      **est_kwargs) -> dict:
    factory = (lambda: build_estimator(**est_kwargs)) if est_kwargs else build_estimator
    return _cv(table, FEATURES, factory, group_col=group_col)


def parameters(model: BucketEstimator) -> pd.Series:
    """Fitted parameters (the process-model analogue of coefficients)."""
    if model.params_ is None:
        raise RuntimeError("fit before reading parameters")
    return model.params_


if __name__ == "__main__":
    import sys
    table = (pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith(".parquet")
             else pd.read_csv(sys.argv[1]))
    cv = leave_site_out_cv(table)
    p, ps = cv["pooled"], cv["per_site"]
    print("pooled:", {k: round(v, 3) if isinstance(v, float) else v for k, v in p.items()})
    print(f"per-station NSE>0: {(ps['nse'] > 0).sum()}/{len(ps)} "
          f"(median {ps['nse'].median():.2f}); median |bias| {ps['bias'].abs().median():.2f}")
