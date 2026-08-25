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
from emt.nn.config import DataConfig, MLPConfig, TrainConfig
from emt.nn.model import MLPModel

DEFAULT_TABLE = "data/model6_features_2006_2010.csv"


def add_config_args(ap: argparse.ArgumentParser, dcs=(TrainConfig, MLPConfig)) -> None:
    """One CLI flag per field of each dataclass in ``dcs``."""
    for dc in dcs:
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


def config_from_args(dc, a):
    return dc(**{f.name: getattr(a, f.name) for f in dataclasses.fields(dc)})


def configs(a) -> tuple[MLPConfig, TrainConfig]:
    return config_from_args(MLPConfig, a), config_from_args(TrainConfig, a)


def main() -> None:
    ap = argparse.ArgumentParser(prog="emt.nn", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["cv", "fit"])
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--design", default="station", choices=cv.DESIGNS)
    ap.add_argument("--out", default=None, help="cv: predictions CSV; fit: model .pt")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="cv: parallel folds on the GPU")
    ap.add_argument("--no-log1p", action="store_true", help="disable the log1p transforms")
    ap.add_argument("--no-static", action="store_true",
                    help="treat every feature as dynamic (no static branch / static noise)")
    ap.add_argument("--tag", default=None, help="name for the output files")
    add_config_args(ap)
    a = ap.parse_args()
    mlp, train = configs(a)
    data = DataConfig(log1p=() if a.no_log1p else DataConfig.log1p,
                      static=() if a.no_static else DataConfig.static)
    df = pd.read_csv(a.table, parse_dates=["time"])
    print(f"table {a.table}: {len(df)} rows, {df['station'].nunique()} stations")
    print(f"data  log1p={data.log1p} static={len(data.static)} cols\nmlp   {mlp}\ntrain {train}")

    if a.cmd == "cv":
        out = cv.run(df, a.design, data=data, mlp=mlp, train=train, workers=a.workers, verbose=True)
        tag = a.tag or f"nn_mlp_{train.loss}"
        cv.print_summary(f"{tag} @{a.design}", out)
        path = a.out or f"data/{tag}_{a.design}cv_predictions.csv"
        out.to_csv(path, index=False)
        print(f"wrote {path}")
    else:
        model = MLPModel(data, mlp, train, verbose=a.verbose).fit(df)
        path = model.save(a.out or "data/models/nn_mlp.pt")
        best = [max(r.get("val_nse", float("nan")) for r in h) for h in model.history]
        print(f"saved {path}; member best val NSE {['%.3f' % b for b in best]}")


if __name__ == "__main__":
    main()
