"""Random hyper-parameter search, scored on the validation ladder.

Samples configs from ``SPACE``, runs ``cv.run`` for each under the chosen fold
design (``block`` is the cheap, honest default: 9 folds and it is the number
that matters for a national product), and appends one row per trial to a CSV
so the search is resumable and inspectable while it runs.

    PYTHONPATH=. python -m emt.nn.search --trials 30 --design block --workers 9
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from emt.nn import cv
from emt.nn.config import MLPConfig, TrainConfig

SPACE = {
    "hidden": [(128, 128), (256, 256), (256, 256, 128), (512, 256, 128), (128,) * 4, (256,) * 4],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
    "residual": [True, False],
    "loss": ["mse", "nse", "huber"],
    "lr": [5e-4, 1e-3, 2e-3, 4e-3],
    "weight_decay": [1e-4, 1e-3, 1e-2, 5e-2],
    "batch_size": [256, 512, 1024, 2048],
    "input_noise": [0.0, 0.05, 0.1, 0.2],
    "epochs": [60, 100, 150],
}


def sample(rng: np.random.Generator) -> tuple[MLPConfig, TrainConfig]:
    pick = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
    mlp = MLPConfig(**{f.name: pick[f.name] for f in dataclasses.fields(MLPConfig)})
    train = TrainConfig(**{f.name: pick[f.name] for f in dataclasses.fields(TrainConfig)
                           if f.name in pick})
    return mlp, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/model6_features_2006_2010.csv")
    ap.add_argument("--design", default="block", choices=cv.DESIGNS)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/nn_search.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.table, parse_dates=["time"])
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    done = len(pd.read_csv(out)) if out.exists() else 0
    for t in range(a.trials):
        mlp, train = sample(rng)
        if t < done:
            continue                      # resume: replay the RNG, skip finished trials
        print(f"\n=== trial {t}: {mlp} {train}", flush=True)
        preds = cv.run(df, a.design, mlp=mlp, train=train, workers=a.workers, verbose=False)
        s = cv.summarise(preds)
        row = {"trial": t, "design": a.design,
               "pooled_nse": s["pooled"]["nse"], "pooled_r": s["pooled"]["r"],
               "stn_median_nse": s["per_station"]["nse"].median(),
               "stn_pos": int((s["per_station"]["nse"] > 0).sum()),
               "stn_median_abs_bias": s["per_station"]["bias"].abs().median(),
               "blk_median_nse": s["per_block"]["nse"].median(),
               "mlp": json.dumps(dataclasses.asdict(mlp)), "train": json.dumps(train.to_dict())}
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
               if k not in ("mlp", "train")}, flush=True)
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
