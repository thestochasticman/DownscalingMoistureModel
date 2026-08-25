"""CLI for the NN track.

    PYTHONPATH=. python -m emt.nn cv   --design station --loss nse
    PYTHONPATH=. python -m emt.nn fit  --out data/models/nn_mlp.pt

Defaults to the cached model6 feature table (data/model6_features_2006_2010.csv),
so results are like-for-like with model6.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from emt.nn import cv
from emt.nn.config import MLPConfig, TrainConfig
from emt.nn.model import MLPModel

DEFAULT_TABLE = "data/model6_features_2006_2010.csv"


def add_config_args(ap: argparse.ArgumentParser) -> None:
    for dc, prefix in ((TrainConfig, ""), (MLPConfig, "")):
        for f in dataclasses.fields(dc):
            if f.name == "device":
                ap.add_argument("--device", default=None)
                continue
            typ = f.type if isinstance(f.type, type) else str
            if f.default is True or f.default is False:
                ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"),
                                default=f.default)
            elif isinstance(f.default, tuple):
                ap.add_argument(f"--{f.name}", type=lambda s: tuple(int(x) for x in s.split(",")),
                                default=f.default)
            elif f.default is None:
                ap.add_argument(f"--{f.name}", type=float, default=None)
            else:
                ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)


def configs(a) -> tuple[MLPConfig, TrainConfig]:
    d = vars(a)
    mlp = MLPConfig(**{f.name: d[f.name] for f in dataclasses.fields(MLPConfig)})
    train = TrainConfig(**{f.name: d[f.name] for f in dataclasses.fields(TrainConfig)})
    return mlp, train


def main() -> None:
    ap = argparse.ArgumentParser(prog="emt.nn", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["cv", "fit"])
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--design", default="station", choices=cv.DESIGNS)
    ap.add_argument("--out", default=None, help="cv: predictions CSV; fit: model .pt")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="cv: parallel folds on the GPU")
    add_config_args(ap)
    a = ap.parse_args()
    mlp, train = configs(a)
    df = pd.read_csv(a.table, parse_dates=["time"])
    print(f"table {a.table}: {len(df)} rows, {df['station'].nunique()} stations")
    print(f"mlp   {mlp}\ntrain {train}")

    if a.cmd == "cv":
        out = cv.run(df, a.design, mlp=mlp, train=train, workers=a.workers, verbose=True)
        cv.print_summary(f"nn-mlp[{train.loss}] @{a.design}", out)
        path = a.out or f"data/nn_mlp_{train.loss}_{a.design}cv_predictions.csv"
        out.to_csv(path, index=False)
        print(f"wrote {path}")
    else:
        model = MLPModel(mlp=mlp, train=train, verbose=a.verbose).fit(df)
        path = model.save(a.out or "data/models/nn_mlp.pt")
        best = [max(r.get("val_nse", float("nan")) for r in h) for h in model.history]
        print(f"saved {path}; member best val NSE {['%.3f' % b for b in best]}")


if __name__ == "__main__":
    main()
