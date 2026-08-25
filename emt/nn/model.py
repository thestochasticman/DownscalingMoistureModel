"""MLPModel: the user-facing object. Fit on a DataFrame, predict on a DataFrame,
save/load. Internally an ensemble of ``n_ensemble`` ResidualMLPs, each trained
from its own seed on its own station-grouped early-stopping split, averaged at
predict time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from emt.nn.config import DataConfig, MLPConfig, TrainConfig
from emt.nn.data import Scaler, TabularData
from emt.nn.mlp import ResidualMLP
from emt.nn.train import Trainer


class MLPModel:
    def __init__(self, data: DataConfig = DataConfig(), mlp: MLPConfig = MLPConfig(),
                 train: TrainConfig = TrainConfig(), verbose: bool = False):
        self.data, self.mlp, self.train_cfg, self.verbose = data, mlp, train, verbose
        self.scaler: Scaler | None = None
        self.nets: list[ResidualMLP] = []
        self.history: list[list[dict]] = []

    # -- training -----------------------------------------------------------
    def fit(self, df: pd.DataFrame, weight: np.ndarray | None = None) -> "MLPModel":
        return self.fit_data(TabularData.from_frame(df, self.data, weight))

    def fit_data(self, d: TabularData) -> "MLPModel":
        self.scaler = Scaler.fit(d.X, d.y)
        trainer = Trainer(self.train_cfg, self.scaler, self.verbose)
        self.nets, self.history = [], []
        for k in range(self.train_cfg.n_ensemble):
            seed = self.train_cfg.seed + 1000 * k
            tr, va = d.grouped_split(self.train_cfg.val_frac, np.random.default_rng(seed))
            net = ResidualMLP(d.X.shape[1], self.mlp)
            if self.verbose:
                print(f"[member {k + 1}/{self.train_cfg.n_ensemble}] "
                      f"train {tr.sum()} rows / val {va.sum()} rows", flush=True)
            self.history.append(trainer.fit(net, d.subset(tr), d.subset(va) if va.any() else None, seed))
            self.nets.append(net.cpu())
        return self

    # -- inference ----------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_X(df[list(self.data.features)].to_numpy(np.float32))

    def predict_X(self, X: np.ndarray) -> np.ndarray:
        if not self.nets:
            raise RuntimeError("MLPModel is not fitted")
        trainer = Trainer(self.train_cfg, self.scaler)
        return np.mean([trainer.predict(net, X) for net in self.nets], axis=0)

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"data": self.data, "mlp": self.mlp, "train": self.train_cfg,
                    "scaler": self.scaler, "history": self.history,
                    "state_dicts": [n.state_dict() for n in self.nets]}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MLPModel":
        ck = torch.load(path, weights_only=False, map_location="cpu")
        m = cls(ck["data"], ck["mlp"], ck["train"])
        m.scaler, m.history = ck["scaler"], ck["history"]
        for sd in ck["state_dicts"]:
            net = ResidualMLP(len(m.data.features), m.mlp)
            net.load_state_dict(sd)
            m.nets.append(net)
        return m
