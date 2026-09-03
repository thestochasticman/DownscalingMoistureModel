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

import functools

import numpy as np
import pandas as pd

from emt.evaluation import metrics
from emt.nn.config import DataConfig, MLPConfig, TrainConfig
from emt.nn.data import TabularData
from emt.nn.model import MLPModel

DESIGNS = ("station", "block", "year", "blockyear")


def block_of(station: str) -> str:
    return {"Y": "YANCO", "K": "KYEAMBA", "A": "ADELONG"}.get(station[0], station)


def fold_labels(d, design: str) -> np.ndarray:
    block = np.array([block_of(s) for s in d.station])
    year = pd.DatetimeIndex(d.time).year.astype(str).to_numpy()
    return {"station": d.station, "block": block, "year": year,
            "blockyear": np.char.add(np.char.add(block, "|"), year)}[design]


def train_mask(labels: np.ndarray, held: str, design: str, d) -> np.ndarray:
    if design != "blockyear":
        return labels != held
    blk, yr = held.split("|")
    return (fold_labels(d, "block") != blk) & (fold_labels(d, "year") != yr)


def _fold(args, retries: int = 3):
    """One fold: fit on the training rows, predict the held-out rows.

    ``weight_fn(subset)`` -- computed on each fold's own training rows, exactly
    like run_blocked_cv -- replaces the subset's sample weights before the fit.

    A CUDA OOM (several workers peaking at once on one card) is retried after
    freeing the cache and waiting, rather than killing the whole run."""
    import time
    import torch
    d, tr, te, factory, weight_fn = args
    for attempt in range(retries + 1):
        try:
            sub = d.subset(tr)
            if weight_fn is not None:
                w = np.asarray(weight_fn(sub), np.float32)
                sub.weight = w * (len(w) / w.sum())
            model = factory().fit_data(sub)
            return model.predict_data(d.subset(te))
        except torch.OutOfMemoryError:
            if attempt == retries:
                raise
            torch.cuda.empty_cache()
            time.sleep(30 * (attempt + 1))


def run_dataset(d, factory, design: str = "station", workers: int = 1,
                verbose: bool = True, weight_fn=None) -> pd.DataFrame:
    """Out-of-fold predictions under ``design`` for any dataset exposing
    ``station``, ``time``, ``y``, ``subset(mask)``, ``frame(pred)`` and any
    model from ``factory()`` exposing ``fit_data`` / ``predict_data``.

    ``workers > 1`` runs folds in parallel processes sharing the GPU -- the
    nets are small, so several folds fit comfortably and the wall-clock is
    dominated by per-step overhead that parallelism hides."""
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    labels = fold_labels(d, design)
    pred = np.full(len(d), np.nan)
    folds = sorted(np.unique(labels))
    masks = [(labels == held, train_mask(labels, held, design, d)) for held in folds]
    jobs = [(d, tr, te, factory, weight_fn) for te, tr in masks]

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
        import os
        import torch.multiprocessing as mp
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, (held, (te, tr), p) in enumerate(zip(folds, masks, pool.imap(_fold, jobs)), 1):
                pred[te] = p
                report(i, held, te, tr)
    out = d.frame(pred)
    out["fold"] = labels
    return out


def run(df: pd.DataFrame, design: str = "station", data: DataConfig = DataConfig(),
        mlp: MLPConfig = MLPConfig(), train: TrainConfig = TrainConfig(),
        weight: np.ndarray | None = None, workers: int = 1, verbose: bool = True) -> pd.DataFrame:
    """The tabular MLP on the ladder."""
    return run_dataset(TabularData.from_frame(df, data, weight),
                       functools.partial(MLPModel, data, mlp, train), design, workers, verbose)


class StratifiedWeights:
    """model8's stratified training weights (see handout/run_blocked_cv.py):
    aridity (P/PET) tertile x block cells, each stratum equal total weight,
    split equally over its blocks, then its samples; tempered by ``**temper``.
    A picklable callable(subset) for ``run_dataset(weight_fn=...)`` -- computed
    on each fold's own training rows."""

    def __init__(self, temper: float = 0.5):
        f = pd.read_csv("data/process_forcing_2005_2010.csv")
        g = f.groupby("station").agg(rain=("daily_rain", "mean"), pet=("et_morton_potential", "mean"))
        self.strata = pd.qcut(g["rain"] / g["pet"], 3, labels=["dry", "mid", "wet"])
        self.temper = temper

    def __call__(self, sub):
        stn = pd.Series(sub.station)
        df = pd.DataFrame({"stratum": stn.map(self.strata), "block": stn.map(block_of)})
        cell = list(zip(df["stratum"], df["block"]))
        cell_n = pd.Series(cell).value_counts()
        blocks_in = df.drop_duplicates().groupby("stratum", observed=True).size()
        n_strata = df["stratum"].nunique()
        w = np.array([1.0 / (n_strata * blocks_in[s] * cell_n[(s, b)]) for s, b in cell])
        w = w ** self.temper
        return w / w.mean()


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
