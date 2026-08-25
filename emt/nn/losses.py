"""Training losses, all in standardised-target units.

* ``mse``   -- plain squared error.
* ``huber`` -- robust to the occasional bad in-situ day.
* ``nse``   -- per-station NSE* (Kratzert et al. 2019, HESS 23:5089):
               ``err^2 / (sigma_station + eps)^2``. Pooled NSE is a rescaled
               MSE with the same optimum; this per-station form is the one that
               changes it, matching the per-station NSE we report.

Every loss takes ``(pred, target, weight, sigma)`` and returns a weighted mean.
"""
from __future__ import annotations

import torch


def mse(pred, target, weight, sigma, *, eps=0.0, delta=1.0):
    return _wmean(0.5 * (pred - target) ** 2, weight)


def huber(pred, target, weight, sigma, *, eps=0.0, delta=1.0):
    err = pred - target
    a = err.abs()
    e = torch.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta))
    return _wmean(e, weight)


def nse(pred, target, weight, sigma, *, eps=0.1, delta=1.0):
    return _wmean(0.5 * (pred - target) ** 2 / (sigma + eps) ** 2, weight)


def _wmean(e, w):
    return (w * e).sum() / w.sum()


LOSSES = {"mse": mse, "huber": huber, "nse": nse}


def get(name: str):
    try:
        return LOSSES[name]
    except KeyError:
        raise ValueError(f"unknown loss {name!r}; choose from {sorted(LOSSES)}") from None
