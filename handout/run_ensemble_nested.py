"""Nested validation of the ensemble's selection rule.

The reported ensemble (blocked pooled NSE +0.42) had its MEMBERSHIP and its
aggregation (mean vs median) chosen by looking at blocked results -- and is
then reported on those same blocked folds. That is selection pressure, and it
is the most attackable number on the branch.

This runs the honest version. For each held-out block B:

  1. choose the configuration -- which bases, mean or median -- using only the
     OTHER eight blocks' out-of-fold rows;
  2. apply that configuration to B, which played no part in choosing it.

The resulting predictions are fully out-of-fold with respect to the selection,
so scoring them answers: does +0.42 survive when the recipe is not allowed to
see the fold it is judged on?

Reference points, both selection-free:
  * every base averaged (median of all six) -- a rule that needs no data;
  * the reported fixed pick, median(hyb, hybA, m8, m6, m9).

Caveat recorded in the output: the inner selection scores configurations on
other blocks' OOF predictions, which come from base models that did see B
during their own training. Removing that too would require re-running every
base under a double hold-out; this check removes the primary pressure (picking
the recipe on the fold you report), not that residual one.

Run from repo root::  PYTHONPATH=. python handout/run_ensemble_nested.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from emt.evaluation import metrics
from emt.nn.cv import block_of

REPO = Path(__file__).resolve().parent.parent
TARGET = "sm_rootzone_pct"
BASES = {
    "hyb":  "nn_hybrid_q_blockcv_predictions.csv",
    "hybA": "nn_hybrid_qa_blockcv_predictions.csv",
    "m8":   "model8_blockcv_capacity_aridity_weighted_predictions.csv",
    "m6":   "model6_blockcv_predictions.csv",
    "m9":   "model9_blockcv_predictions.csv",
    "seqB": "nn_seq_big_blockcv_predictions.csv",
}
REPORTED = (("hyb", "hybA", "m8", "m6", "m9"), "median")


def load() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    frames = {}
    for k, f in BASES.items():
        o = pd.read_csv(REPO / "data" / f, parse_dates=["time"]).set_index(["station", "time"])
        frames[k] = o["pred"]
        y = o[TARGET]
    P = pd.DataFrame(frames).dropna()
    y = y.loc[P.index]
    blocks = np.array([block_of(s) for s, _ in P.index])
    return P, y, blocks


def configurations(names) -> list[tuple[tuple[str, ...], str]]:
    out = []
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            out += [(combo, "mean"), (combo, "median")]
    return out


def apply(P: pd.DataFrame, cfg) -> pd.Series:
    members, agg = cfg
    return getattr(P[list(members)], agg)(axis=1)


def score(y, pred, blocks) -> dict:
    o = pd.DataFrame({"y": y, "p": pred, "b": blocks,
                      "s": [s for s, _ in y.index]})
    stn = o.groupby("s").apply(lambda g: metrics(g["y"], g["p"])["nse"], include_groups=False)
    blk = o.groupby("b").apply(lambda g: metrics(g["y"], g["p"])["nse"], include_groups=False)
    m = metrics(o["y"], o["p"])
    return dict(pooled=round(m["nse"], 3), r=round(m["r"], 3),
                stn_pos=f"{int((stn > 0).sum())}/{len(stn)}",
                blk_pos=f"{int((blk > 0).sum())}/{len(blk)}",
                blk_med=round(float(blk.median()), 3))


def main() -> None:
    P, y, blocks = load()
    cfgs = configurations(list(BASES))
    print(f"{len(P):,} rows, {len(BASES)} bases, {len(cfgs)} candidate configurations, "
          f"{len(set(blocks))} outer folds\n")

    for criterion in ("pooled", "block-median"):
        nested = pd.Series(np.nan, index=P.index)
        chosen = {}
        for B in sorted(set(blocks)):
            inner = blocks != B
            best, best_v = None, -np.inf
            for cfg in cfgs:
                p = apply(P[inner], cfg)
                yi, bi = y[inner], blocks[inner]
                if criterion == "pooled":
                    v = metrics(yi, p)["nse"]
                else:
                    v = float(pd.DataFrame({"y": yi, "p": p, "b": bi})
                              .groupby("b").apply(lambda g: metrics(g["y"], g["p"])["nse"],
                                                  include_groups=False).median())
                if v > best_v:
                    best, best_v = cfg, v
            nested[blocks == B] = apply(P[blocks == B], best).values
            chosen[B] = f"{best[1]}({'+'.join(best[0])})"
        print(f"inner criterion = {criterion}")
        print(f"  NESTED (selection never saw the scored fold): {score(y, nested, blocks)}")
        picks = pd.Series(chosen)
        print(f"  configurations chosen: {picks.nunique()} distinct across 9 folds")
        for cfg, n in picks.value_counts().items():
            print(f"    {n}x  {cfg}")
        print()

    print("selection-free reference points (same rows):")
    print(f"  median(all six bases)          {score(y, P.median(axis=1), blocks)}")
    print(f"  reported fixed pick            {score(y, apply(P, REPORTED), blocks)}")
    print("\nNote: the inner selection scores candidates on other blocks' out-of-fold rows,")
    print("whose base models did see the held-out block during their own training. This")
    print("check removes the primary selection pressure, not that residual one.")


if __name__ == "__main__":
    main()
