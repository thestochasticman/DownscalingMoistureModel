"""Two cheap learned combiners over the base models, per fold, both designs.

The gated stack (emt/nn/stack.py) lost to the plain mean. Before concluding
"learning loses", test the two simplest learned combiners:

* ``global``  -- ONE convex weight vector per fold (k numbers on the simplex),
                 fitted on the training rows' out-of-fold base predictions.
                 If this beats the mean, per-sample gating was the problem;
                 if not, equal weights are genuinely near-optimal.
* ``affine``  -- ridge regression on the base predictions plus an intercept,
                 shrunk toward the plain mean (penalising deviation from
                 weights 1/k, bias 0). The combiner that IS allowed to correct
                 level -- the test of whether the convexity constraint costs.

No networks; each fold is a tiny least-squares problem. Run from repo root::

    PYTHONPATH=. python handout/run_stack_variants.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from emt.evaluation import metrics
from emt.nn import cv
from emt.nn.stack import BASES, TARGET, build_table


def global_convex(P_tr, y_tr, P_te):
    """min ||P w - y||^2 over the simplex (softmax parametrisation)."""
    k = P_tr.shape[1]

    def loss(z):
        w = np.exp(z - z.max()); w /= w.sum()
        return float(((P_tr @ w - y_tr) ** 2).mean())

    res = minimize(loss, np.zeros(k), method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-8, maxfev=2000))
    w = np.exp(res.x - res.x.max()); w /= w.sum()
    return P_te @ w, w


def affine_ridge(P_tr, y_tr, P_te, lam=10.0):
    """Ridge toward the plain mean: pred = P w + b, penalty lam*(||w-1/k||^2 + b^2).

    lam is in units of (training rows / 1000) so the shrinkage does not vanish
    as n grows; lam=10 keeps the fit close to the mean unless the data insist.
    """
    n, k = P_tr.shape
    X = np.column_stack([P_tr, np.ones(n)])
    w0 = np.append(np.full(k, 1.0 / k), 0.0)
    L = lam * n / 1000.0
    A = X.T @ X + L * np.eye(k + 1)
    b = X.T @ y_tr + L * w0
    w = np.linalg.solve(A, b)
    return np.column_stack([P_te, np.ones(len(P_te))]) @ w, w


def run(design: str) -> None:
    tab, pcols = build_table(design)
    P = tab[pcols].to_numpy(float)
    y = tab[TARGET].to_numpy(float)

    class D:                                   # minimal dataset for fold_labels
        station = tab["station"].to_numpy(str)
        time = pd.to_datetime(tab["time"]).to_numpy()
    labels = cv.fold_labels(D, design)

    out = {name: np.full(len(tab), np.nan) for name in ("global", "affine")}
    weights = {}
    for held in sorted(np.unique(labels)):
        te = labels == held
        tr = cv.train_mask(labels, held, design, D)
        out["global"][te], wg = global_convex(P[tr], y[tr], P[te])
        out["affine"][te], wa = affine_ridge(P[tr], y[tr], P[te])
        weights[held] = np.round(wg, 2)
    res = tab[["station", "time", TARGET]].copy()
    res["mean"] = P.mean(1)
    for name, p in out.items():
        res[name] = p

    print(f"\n== {design}: bases {pcols}")
    for name in ("mean", "global", "affine"):
        o = res.rename(columns={name: "pred"})
        cv.print_summary(f"{name:>7}", o)
    print("  per-fold global weights:", {k: list(v) for k, v in weights.items()})


if __name__ == "__main__":
    for design in BASES:
        run(design)
