"""Tabular data container: the one place that knows about DataFrames.

``TabularData`` holds a feature matrix, target, station labels and timestamps as
numpy arrays, plus a ``Scaler`` fitted on whatever rows it was built from. The
network code downstream sees only tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from emt.nn.config import DataConfig


@dataclass(frozen=True)
class Scaler:
    """log1p on the chosen columns, then z-score; target z-scored."""
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    log_idx: tuple[int, ...] = ()

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, log_idx: tuple[int, ...] = ()) -> "Scaler":
        Xl = cls._log(X, log_idx)
        return cls(Xl.mean(0), Xl.std(0) + 1e-6, float(y.mean()), float(y.std() + 1e-6), log_idx)

    @staticmethod
    def _log(X, idx):
        if not idx:
            return X
        X = X.copy()
        X[:, list(idx)] = np.log1p(np.clip(X[:, list(idx)], 0, None))
        return X

    def x(self, X): return (self._log(X, self.log_idx) - self.x_mean) / self.x_std
    def y(self, y): return (y - self.y_mean) / self.y_std
    def y_inv(self, ys): return ys * self.y_std + self.y_mean


@dataclass
class TabularData:
    X: np.ndarray                 # (n, p) float32, raw units
    y: np.ndarray                 # (n,)   float32, raw units (nan allowed at predict time)
    station: np.ndarray           # (n,)   str
    time: np.ndarray              # (n,)   datetime64
    weight: np.ndarray            # (n,)   float32, mean 1
    cfg: DataConfig

    @classmethod
    def from_frame(cls, df: pd.DataFrame, cfg: DataConfig = DataConfig(),
                   weight: np.ndarray | None = None, require_target: bool = True) -> "TabularData":
        cols = list(cfg.features) + ([cfg.target] if require_target else [])
        df = df.dropna(subset=cols).reset_index(drop=True)
        w = np.ones(len(df), np.float32) if weight is None else np.asarray(weight, np.float32)
        w = w * (len(w) / w.sum())
        y = (df[cfg.target].to_numpy(np.float32) if cfg.target in df
             else np.full(len(df), np.nan, np.float32))
        return cls(df[list(cfg.features)].to_numpy(np.float32), y,
                   df[cfg.group].to_numpy(str) if cfg.group in df else np.array(["?"] * len(df)),
                   pd.to_datetime(df[cfg.time]).to_numpy() if cfg.time in df
                   else np.full(len(df), np.datetime64("NaT")),
                   w, cfg)

    def __len__(self): return len(self.y)

    def subset(self, mask: np.ndarray) -> "TabularData":
        return TabularData(self.X[mask], self.y[mask], self.station[mask], self.time[mask],
                           self.weight[mask], self.cfg)

    @property
    def station_codes(self) -> tuple[np.ndarray, np.ndarray]:
        """(integer code per row, unique station names)."""
        codes, uniq = pd.factorize(self.station)
        return codes, np.asarray(uniq)

    def station_sigma(self, scaler: Scaler) -> np.ndarray:
        """Per-row training std of the standardised target at that row's station."""
        codes, uniq = self.station_codes
        ys = scaler.y(self.y)
        sig = pd.Series(ys).groupby(codes).std().reindex(range(len(uniq))).fillna(1.0)
        return sig.to_numpy(np.float32)[codes]

    def grouped_split(self, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """(train_mask, val_mask) holding out whole stations."""
        if frac <= 0:
            return np.ones(len(self), bool), np.zeros(len(self), bool)
        codes, uniq = self.station_codes
        if len(uniq) < 4:                       # too few stations: fall back to rows
            va = rng.random(len(self)) < frac
            return ~va, va
        n_val = max(1, int(round(frac * len(uniq))))
        va = np.isin(codes, rng.choice(len(uniq), n_val, replace=False))
        return ~va, va

    def frame(self, pred: np.ndarray | None = None) -> pd.DataFrame:
        out = pd.DataFrame({self.cfg.group: self.station, self.cfg.time: self.time,
                            self.cfg.target: self.y})
        if pred is not None:
            out["pred"] = pred
        return out
