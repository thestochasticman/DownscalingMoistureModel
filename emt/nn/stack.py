"""The stack: a learned combiner over the repo's validated models.

``mean(hybrid, model8)`` already beats every single model under both designs
(see ``emt/nn/README.md``). The stack replaces that fixed 50/50 with weights
that DEPEND ON THE SITE: a small net looks at a sample's statics and at how
much the base models disagree, and outputs mixing weights.

How a prediction is made, in full::

    inputs   statics(x)               soil, terrain, aridity (quantile-scaled)
             p_1(x) .. p_k(x)         the base models' predictions (raw %)
    gate     w = softmax(MLP(statics, p_centred, p_spread))     k weights, sum 1
    output   pred(x) = sum_i w_i * p_i(x)

Two properties keep it honest:

* **Convex, no intercept, no free scale.** The output is always a weighted
  average of the base predictions, so the gate can only *choose between*
  models -- it cannot shift a site's level, which is how every over-flexible
  model here has failed. The gate's last layer is zero-initialised: at step 0
  the stack IS the plain mean, and any departure from it must survive the
  station-grouped early-stopping split.
* **Fold discipline, same ladder.** The stack is evaluated with the same
  station / block folds as every model: to predict a held-out site, the gate
  is trained on the OTHER sites only. Its training data are the bases'
  out-of-fold predictions (each made by a model never trained on its own
  site), so no observation of a held-out site ever influences its prediction
  -- standard stacked generalisation.

Bases (all have cached out-of-fold predictions for the design used):

    station   hybrid-q, model8, nn-mlp, nn-seq
    block     hybrid-q, model8, model6

Run::

    PYTHONPATH=. python -m emt.nn.stack cv --design block
    PYTHONPATH=. python -m emt.nn.stack cv --design station
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from emt.nn.config import STATIC_FEATURES, TrainConfig
from emt.nn.data import DataConfig, Scaler, TabularData, TabularView
from emt.nn.train import Trainer

DATA = Path("data")
TARGET = "sm_rootzone_pct"

#: out-of-fold prediction files per design; every file is leakage-safe for
#: that design's folds
BASES = {
    "station": {
        "hybrid": "nn_hybrid_q_stationcv_predictions.csv",
        "model8": "model8_losocv_capacity_aridity_weighted_predictions.csv",
        "mlp": "nn_B_log_snoise_stationcv_predictions.csv",
        "seq": "nn_seq_nse_stationcv_predictions.csv",
    },
    "block": {
        "hybrid": "nn_hybrid_q_blockcv_predictions.csv",
        "model8": "model8_blockcv_capacity_aridity_weighted_predictions.csv",
        "model6": "model6_blockcv_predictions.csv",
    },
}
STATICS_CSVS = ("process_soil_statics.csv", "process_terrain_statics.csv",
                "process_climate_statics.csv")
GATE_STATICS = tuple(f for f in STATIC_FEATURES if f != "aridity") + ("aridity",)
#: regime conditioning: WHEN to trust a base, not WHERE -- day-of-year and the
#: recent forcing state, from the cached model6 feature table. Unlike statics
#: these vary within a site, so the gate is not a purely spatial model and has
#: ~50k samples (not 37 sites) to learn from.
REGIME_FEATURES = ("doy_sin", "doy_cos", "rain_7", "rain_30", "ppet_30", "vpd_30")


def build_table(design: str, gate_on: str = "statics") -> tuple[pd.DataFrame, list[str]]:
    """One row per (station, day): target, gate features (site statics or
    regime state), and every base's out-of-fold prediction (``p_<name>``)."""
    tab = None
    for name, f in BASES[design].items():
        o = pd.read_csv(DATA / f, parse_dates=["time"])
        o = o[["station", "time", TARGET, "pred"]].rename(columns={"pred": f"p_{name}"})
        tab = o if tab is None else tab.merge(o.drop(columns=[TARGET]), on=["station", "time"])
    if gate_on == "regime":
        reg = pd.read_csv(DATA / "model6_features_2006_2010.csv", parse_dates=["time"])
        tab = tab.merge(reg[["station", "time", *REGIME_FEATURES]], on=["station", "time"])
    else:
        statics = None
        for f in STATICS_CSVS:
            s = pd.read_csv(DATA / f)
            statics = s if statics is None else statics.merge(s, on="station")
        tab = tab.merge(statics, on="station")
    return tab.dropna(), [f"p_{n}" for n in BASES[design]]


@dataclass(frozen=True)
class StackScaler:
    """Statics are FEATURES: quantile-scaled (the method that won for the
    hybrid). Base predictions are CANDIDATE ANSWERS in %, not features: all k
    are scaled identically by the target's train mean/std, so their convex
    combination is still a prediction on the target scale. Presents the
    ``Scaler`` interface (x / y / y_inv) the shared ``TabularView`` expects."""
    statics: Scaler          # quantile scaler over the static columns
    n_base: int
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, n_base: int) -> "StackScaler":
        return cls(Scaler.fit(X[:, :-n_base], y, (), "quantile"), n_base,
                   float(y.mean()), float(y.std() + 1e-6))

    def x(self, X: np.ndarray) -> np.ndarray:
        s = self.statics.x(X[:, :-self.n_base])
        p = (X[:, -self.n_base:] - self.y_mean) / self.y_std
        return np.concatenate([s, p], 1).astype(np.float32)

    def y(self, y): return (y - self.y_mean) / self.y_std
    def y_inv(self, ys): return ys * self.y_std + self.y_mean


class GateNet(nn.Module):
    """softmax gate -> convex combination of the base predictions.

    Input layout (after StackScaler): ``[statics | p_1..p_k]``. The gate sees
    the statics plus two disagreement signals -- the centred predictions and
    their spread -- but the OUTPUT uses the raw (standardised) predictions.
    Zero-initialised last layer: step 0 = the plain mean of the bases.
    """

    def __init__(self, n_static: int, n_base: int, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.n_base = n_base
        self.gate = nn.Sequential(nn.Linear(n_static + n_base + 1, hidden), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_base))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x):
        statics, preds = x[:, :-self.n_base], x[:, -self.n_base:]
        centred = preds - preds.mean(1, keepdim=True)
        spread = preds.std(1, keepdim=True)
        w = torch.softmax(self.gate(torch.cat([statics, centred, spread], 1)), 1)
        return (w * preds).sum(1)

    def weights(self, x):
        statics, preds = x[:, :-self.n_base], x[:, -self.n_base:]
        centred = preds - preds.mean(1, keepdim=True)
        return torch.softmax(self.gate(torch.cat([statics, centred,
                                                  preds.std(1, keepdim=True)], 1)), 1)


class StackModel:
    """fit/predict wrapper with the shared Trainer, ensemble and early
    stopping; drops into ``cv.run_dataset`` like every other model here."""

    def __init__(self, data_cfg: DataConfig, n_base: int, train: TrainConfig = TrainConfig(),
                 hidden: int = 32, verbose: bool = False):
        self.data_cfg, self.n_base = data_cfg, n_base
        self.train_cfg, self.hidden, self.verbose = train, hidden, verbose
        self.scaler: StackScaler | None = None
        self.nets: list[GateNet] = []

    def fit_data(self, d: TabularData) -> "StackModel":
        self.scaler = StackScaler.fit(d.X, d.y, self.n_base)
        trainer = Trainer(self.train_cfg, self.verbose)
        view = lambda sub: TabularView(sub, self.scaler, trainer.device)  # noqa: E731
        self.nets = []
        for k in range(self.train_cfg.n_ensemble):
            seed = self.train_cfg.seed + 1000 * k
            tr, va = d.grouped_split(self.train_cfg.val_frac, np.random.default_rng(seed))
            net = GateNet(d.X.shape[1] - self.n_base, self.n_base, self.hidden)
            trainer.fit(net, view(d.subset(tr)), view(d.subset(va)) if va.any() else None, seed)
            self.nets.append(net.cpu())
        return self

    def predict_data(self, d: TabularData) -> np.ndarray:
        trainer = Trainer(self.train_cfg)
        view = TabularView(d, self.scaler, trainer.device)
        ys = torch.stack([trainer.predict_view(net, view) for net in self.nets]).mean(0)
        return self.scaler.y_inv(ys.cpu().numpy())

    def station_weights(self, d: TabularData, base_names: list[str]) -> pd.DataFrame:
        """Mean gate weight per station (ensemble mean) -- who gets trusted where."""
        view = TabularView(d, self.scaler, torch.device("cpu"))
        with torch.no_grad():
            W = torch.stack([net.weights(view.X) for net in self.nets]).mean(0).numpy()
        return (pd.DataFrame(W, columns=base_names).assign(station=d.station)
                  .groupby("station").mean())


def main() -> None:
    import argparse
    from emt.evaluation import metrics
    from emt.nn import cv

    ap = argparse.ArgumentParser(prog="emt.nn.stack",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["cv"])
    ap.add_argument("--design", default="block", choices=list(BASES))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--ensemble", type=int, default=4)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="nn_stack")
    ap.add_argument("--gate-on", default="statics", choices=["statics", "regime"],
                    help="condition the gate on WHERE (site statics) or WHEN (regime state)")
    a = ap.parse_args()

    tab, pcols = build_table(a.design, a.gate_on)
    gate_feats = REGIME_FEATURES if a.gate_on == "regime" else GATE_STATICS
    feats = [*(f for f in gate_feats if f in tab.columns), *pcols]
    dcfg = DataConfig(features=tuple(feats), scale="quantile", static=(), log1p=())
    train = TrainConfig(loss="nse", epochs=a.epochs, n_ensemble=a.ensemble, batch_size=4096,
                        lr=5e-3, input_noise=0.0, static_noise=0.0, device="cpu", amp=False,
                        patience=25)
    d = TabularData.from_frame(tab, dcfg)
    print(f"stack[{a.design}|{a.gate_on}]: {len(d)} rows, bases {pcols}, "
          f"{len(feats) - len(pcols)} gate features")
    for name in pcols:                                   # baselines, same rows
        print(f"  base {name:<9} pooled NSE {metrics(tab[TARGET], tab[name])['nse']:+.3f}")
    print(f"  plain mean     pooled NSE {metrics(tab[TARGET], tab[pcols].mean(1))['nse']:+.3f}")

    out = cv.run_dataset(d, functools.partial(StackModel, dcfg, len(pcols), train, a.hidden),
                         a.design, a.workers, True)
    cv.print_summary(f"{a.tag} @{a.design}", out)
    path = f"data/{a.tag}_{a.design}cv_predictions.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
