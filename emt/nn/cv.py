"""The validation ladder for the NN track.

Four fold designs of increasing strictness, the same ones the handout uses:

    station    leave-one-station-out (interpolation next to instrumented sites)
    block      leave-one-block-out: YANCO / KYEAMBA / ADELONG / each M-site (transfer)
    year       leave-one-year-out (regime transfer)
    blockyear  hold out one (block, year); train on neither that block nor that year

``run`` returns out-of-fold predictions ``[station, time, target, pred, fold]``;
``summarise`` scores them pooled, per block and per station.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from emt.evaluation import metrics
from emt.nn.config import DataConfig, MLPConfig, TrainConfig
from emt.nn.data import TabularData
from emt.nn.model import MLPModel

DESIGNS = ("station", "block", "year", "blockyear")


def block_of(station: str) -> str:
    return {"Y": "YANCO", "K": "KYEAMBA", "A": "ADELONG"}.get(station[0], station)


def fold_labels(d: TabularData, design: str) -> np.ndarray:
    block = np.array([block_of(s) for s in d.station])
    year = pd.DatetimeIndex(d.time).year.astype(str).to_numpy()
    return {"station": d.station, "block": block, "year": year,
            "blockyear": np.char.add(np.char.add(block, "|"), year)}[design]


def train_mask(labels: np.ndarray, held: str, design: str, d: TabularData) -> np.ndarray:
    if design != "blockyear":
        return labels != held
    blk, yr = held.split("|")
    return (fold_labels(d, "block") != blk) & (fold_labels(d, "year") != yr)


def _fold(args):
    """One fold: fit on the training rows, predict the held-out rows."""
    d, tr, te, data, mlp, train = args
    model = MLPModel(data, mlp, train).fit_data(d.subset(tr))
    return model.predict_X(d.X[te])


def run(df: pd.DataFrame, design: str = "station", data: DataConfig = DataConfig(),
        mlp: MLPConfig = MLPConfig(), train: TrainConfig = TrainConfig(),
        weight: np.ndarray | None = None, workers: int = 1, verbose: bool = True) -> pd.DataFrame:
    """Out-of-fold predictions under ``design``. ``workers > 1`` runs folds in
    parallel processes sharing the GPU -- the nets are tiny, so several folds
    fit comfortably and the wall-clock is dominated by per-step overhead that
    parallelism hides."""
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    d = TabularData.from_frame(df, data, weight)
    labels = fold_labels(d, design)
    pred = np.full(len(d), np.nan)
    folds = sorted(np.unique(labels))
    masks = [(labels == held, train_mask(labels, held, design, d)) for held in folds]
    jobs = [(d, tr, te, data, mlp, train) for te, tr in masks]

    def report(i, held, te, tr):
        if verbose:
            m = metrics(d.y[te], pred[te])
            print(f"  [{design} {i}/{len(folds)}] held out {held:<14} "
                  f"trained on {len(np.unique(d.station[tr]))} stations   "
                  f"NSE {m['nse']:+.3f}  r {m['r']:.2f}  bias {m['bias']:+.2f}", flush=True)

    if workers <= 1:
        for i, (held, (te, tr), job) in enumerate(zip(folds, masks, jobs), 1):
            pred[te] = _fold(job)
            report(i, held, te, tr)
    else:
        import torch.multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, (held, (te, tr), p) in enumerate(zip(folds, masks, pool.imap(_fold, jobs)), 1):
                pred[te] = p
                report(i, held, te, tr)
    out = d.frame(pred)
    out["fold"] = labels
    return out


def summarise(out: pd.DataFrame, target: str = DataConfig.target) -> dict:
    stn = (out.groupby("station")
              .apply(lambda g: pd.Series(metrics(g[target], g["pred"])), include_groups=False))
    blk = (out.assign(block=out["station"].map(block_of)).groupby("block")
              .apply(lambda g: pd.Series(metrics(g[target], g["pred"])), include_groups=False))
    return {"pooled": metrics(out[target], out["pred"]), "per_station": stn, "per_block": blk}


def print_summary(name: str, out: pd.DataFrame) -> None:
    s = summarise(out)
    p, stn, blk = s["pooled"], s["per_station"], s["per_block"]
    print(f"\n{name}: pooled NSE {p['nse']:+.3f}  r {p['r']:.3f}  RMSE {p['rmse']:.2f}  "
          f"ubRMSE {p['ubrmse']:.2f}  bias {p['bias']:+.2f}  (n={p['n']})")
    print(f"  stations: NSE>0 {(stn['nse'] > 0).sum()}/{len(stn)}  median NSE {stn['nse'].median():+.2f}  "
          f"median r {stn['r'].median():.2f}  median |bias| {stn['bias'].abs().median():.2f}")
    print(f"  blocks:   median NSE {blk['nse'].median():+.2f}   "
          + "  ".join(f"{b} {v:+.2f}" for b, v in blk['nse'].items()))
