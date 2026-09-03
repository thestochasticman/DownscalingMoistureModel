"""The hybrid: a differentiable model7 bucket whose parameters come from a net.

model7 calibrates five *global* bucket parameters and model8 then ridge-regresses
a per-station level offset on soil/terrain/aridity. Here the same bucket is
written in torch and run for every station over the full forcing panel, and a
small MLP maps the station statics to per-station **deviations** of the five
parameters from a learned global set::

    theta_i = lo + (hi - lo) * sigmoid( g + dev_scale * MLP(statics_i) )
    S_i(t)  = clip(S + P - PET * min(1, S/(alpha_i smax_i)) - k_i S, 0, smax_i)
    vwc_i(t)= theta_r_i + dtheta_i * S_i(t)/smax_i  [+ w . statics_i + b]

so soil can change capacity, ET stress and recession *inside the physics*, not
just shift the readout -- while the water balance stays enforced. The
deviation MLP's last layer is zero-initialised: at step 0 this IS model7 with
its default parameters, and every departure has to earn its place against the
station-grouped validation split. The optional offset head is model8's ridge
stage trained jointly.

Training is full-batch: one forward pass simulates the whole panel, so a
mini-batch would recompute it for nothing. ``SeqData`` (with lookback=1) supplies
the panel, statics and samples; the Trainer, losses, ensemble and CV ladder are
shared with the other models.

    PYTHONPATH=. python -m emt.nn.hybrid cv --design station --loss nse --workers 6
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from emt.nn.config import (BUCKET_BOUNDS, BUCKET_PARAMS, BUCKET_X0, HybridConfig, SeqDataConfig,
                           TrainConfig)
from emt.nn.data import Scaler
from emt.nn.seq import SeqData, load_frames, DATA
from emt.nn.train import Trainer


def _logit(p):
    return float(np.log(p / (1 - p)))


class DiffBucket(nn.Module):
    """Parameters only; the forcing panel and statics arrive with every call."""

    def __init__(self, n_static: int, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        lo = torch.tensor([b[0] for b in BUCKET_BOUNDS])
        hi = torch.tensor([b[1] for b in BUCKET_BOUNDS])
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        x0 = torch.tensor(BUCKET_X0)
        self.g = nn.Parameter(torch.tensor([_logit(v) for v in ((x0 - lo) / (hi - lo)).tolist()]))
        mask = torch.tensor([1.0 if p in cfg.learn else 0.0 for p in BUCKET_PARAMS])
        self.register_buffer("learn_mask", mask)
        if cfg.hidden > 0 and cfg.dev_scale > 0:
            self.dev = nn.Sequential(nn.Linear(n_static, cfg.hidden), nn.SiLU(), nn.Dropout(cfg.dropout),
                                     nn.Linear(cfg.hidden, len(BUCKET_PARAMS)))
            nn.init.zeros_(self.dev[-1].weight)
            nn.init.zeros_(self.dev[-1].bias)
        else:
            self.dev = None
        if cfg.offset:
            self.offset = nn.Linear(n_static, 1)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
        else:
            self.offset = None

    def params(self, sta: torch.Tensor) -> torch.Tensor:
        """(n_stations, 5) bucket parameters in physical units."""
        z = self.g[None, :].expand(sta.shape[0], -1)
        if self.dev is not None:
            z = z + self.cfg.dev_scale * self.learn_mask * self.dev(sta)
        return self.lo + (self.hi - self.lo) * torch.sigmoid(z)

    @staticmethod
    def simulate(rain: torch.Tensor, pet: torch.Tensor, smax, alpha, k) -> torch.Tensor:
        """model7's recurrence over a (T, n_stations) panel -> storage (T, n_stations)."""
        s = 0.5 * smax
        denom = alpha * smax
        out = []
        for t in range(rain.shape[0]):
            s = s + rain[t]
            aet = pet[t] * torch.clamp(s / denom, max=1.0)
            s = torch.clamp(s - aet - k * s, min=0.0)
            s = torch.minimum(s, smax)
            out.append(s)
        return torch.stack(out)

    def forward(self, inputs):
        rain, pet, sta, stn, pos, noisy, static_noise = inputs
        if noisy and static_noise > 0:
            sta = sta + static_noise * torch.randn_like(sta)
        p = self.params(sta)
        smax, alpha, k, theta_r, dtheta = p.unbind(1)
        storage = self.simulate(rain, pet, smax, alpha, k)           # (T, n_stn)
        vwc = theta_r + dtheta * storage / smax                       # (T, n_stn), %
        if self.offset is not None:
            vwc = vwc + self.offset(sta).squeeze(-1)
        return vwc[pos, stn]                                          # raw %, standardised by the view


class HybridView:
    """``SeqData`` on a device for the bucket (see ``train.DeviceView``).

    The net returns raw volumetric %; the view exposes the target on the same
    scale (``scaler.y``) by wrapping predictions in ``batch``? No -- simpler:
    ``batch`` hands the net a ``scale`` so the loss compares standardised units.
    """

    def __init__(self, d: SeqData, scaler: Scaler, device: torch.device, static_noise: float = 0.0):
        T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        i_rain = d.cfg.forcing.index("daily_rain")
        i_pet = d.cfg.forcing.index("et_morton_potential")
        self.n = len(d)
        self.rain = T(d.F[:, :, i_rain].T.copy())                   # (T, n_stn)
        self.pet = T(d.F[:, :, i_pet].T.copy())
        self.S = T(scaler.x(d.S))
        self.stn, self.pos = T(d.stn_idx, torch.long), T(d.pos, torch.long)
        self.y, self.w = T(scaler.y(d.y)), T(d.weight)
        self.sigma = T(d.station_sigma(scaler))
        self.static_noise = static_noise
        self.y_mean, self.y_std = scaler.y_mean, scaler.y_std

    def batch(self, idx: torch.Tensor, noisy: bool):
        return (self.rain, self.pet, self.S, self.stn[idx], self.pos[idx], noisy, self.static_noise)


class Standardised(nn.Module):
    """Wrap the bucket so the Trainer sees standardised-target outputs."""

    def __init__(self, net: DiffBucket, y_mean: float, y_std: float):
        super().__init__()
        self.net = net
        self.register_buffer("y_mean", torch.tensor(float(y_mean)))
        self.register_buffer("y_std", torch.tensor(float(y_std)))

    def forward(self, inputs):
        return (self.net(inputs) - self.y_mean) / self.y_std


class HybridModel:
    def __init__(self, data: SeqDataConfig = SeqDataConfig(lookback=1), net: HybridConfig = HybridConfig(),
                 train: TrainConfig = TrainConfig(), verbose: bool = False):
        self.data, self.net_cfg, self.train_cfg, self.verbose = data, net, train, verbose
        self.scaler: Scaler | None = None
        self.nets: list[Standardised] = []
        self.history: list[list[dict]] = []

    def fit_data(self, d: SeqData) -> "HybridModel":
        S_train = d.S[np.unique(d.stn_idx)]
        self.scaler = Scaler.fit(S_train, d.y, d.cfg.static_log_idx, d.cfg.scale)
        trainer = Trainer(self.train_cfg, self.verbose)
        view = lambda sub: HybridView(sub, self.scaler, trainer.device, self.train_cfg.static_noise)  # noqa: E731
        self.nets, self.history = [], []
        for k in range(self.train_cfg.n_ensemble):
            seed = self.train_cfg.seed + 1000 * k
            tr, va = d.grouped_split(self.train_cfg.val_frac, np.random.default_rng(seed))
            net = Standardised(DiffBucket(d.S.shape[1], self.net_cfg), self.scaler.y_mean, self.scaler.y_std)
            if self.verbose:
                print(f"[member {k + 1}/{self.train_cfg.n_ensemble}] "
                      f"train {tr.sum()} rows / val {va.sum()} rows", flush=True)
            self.history.append(trainer.fit(net, view(d.subset(tr)),
                                            view(d.subset(va)) if va.any() else None, seed))
            self.nets.append(net.cpu())
        return self

    def predict_data(self, d: SeqData) -> np.ndarray:
        trainer = Trainer(self.train_cfg)
        view = HybridView(d, self.scaler, trainer.device)
        ys = torch.stack([trainer.predict_view(net, view, chunk=len(d)) for net in self.nets]).mean(0)
        return self.scaler.y_inv(ys.cpu().numpy())

    def station_params(self, d: SeqData) -> pd.DataFrame:
        """Fitted per-station bucket parameters (ensemble mean), physical units."""
        dev = torch.device("cpu")
        S = torch.as_tensor(self.scaler.x(d.S), dtype=torch.float32, device=dev)
        with torch.no_grad():
            P = torch.stack([n.net.to(dev).eval().params(S) for n in self.nets]).mean(0).numpy()
        out = pd.DataFrame(P, columns=BUCKET_PARAMS, index=d.stations)
        if self.net_cfg.offset:
            with torch.no_grad():
                out["offset"] = torch.stack([n.net.offset(S).squeeze(-1) for n in self.nets]).mean(0).numpy()
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"data": self.data, "net": self.net_cfg, "train": self.train_cfg, "scaler": self.scaler,
                    "history": self.history, "n_static": len(self.scaler.x_mean),
                    "state_dicts": [n.state_dict() for n in self.nets]}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "HybridModel":
        ck = torch.load(path, weights_only=False, map_location="cpu")
        m = cls(ck["data"], ck["net"], ck["train"])
        m.scaler, m.history = ck["scaler"], ck["history"]
        for sd in ck["state_dicts"]:
            net = Standardised(DiffBucket(ck["n_static"], m.net_cfg), m.scaler.y_mean, m.scaler.y_std)
            net.load_state_dict(sd)
            m.nets.append(net)
        return m


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
HYBRID_TRAIN = dict(batch_size=1 << 20, epochs=400, patience=40, lr=0.02, weight_decay=1e-3,
                    warmup_frac=0.05, input_noise=0.0, n_ensemble=3)


def main() -> None:
    import argparse
    from emt.nn import cv
    from emt.nn.__main__ import add_config_args, config_from_args

    ap = argparse.ArgumentParser(prog="emt.nn.hybrid", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["cv", "fit"])
    ap.add_argument("--design", default="station", choices=cv.DESIGNS)
    ap.add_argument("--forcing", default=str(DATA / "process_forcing_2005_2010.csv"))
    ap.add_argument("--target", default=str(DATA / "process_target_2006_2010.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--scale", default="zscore", choices=["zscore", "robust", "quantile"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--weighted", action="store_true",
                    help="model8's stratified training weights (per training fold)")
    ap.add_argument("-v", "--verbose", action="store_true")
    add_config_args(ap, (TrainConfig, HybridConfig))
    ap.set_defaults(**HYBRID_TRAIN)
    a = ap.parse_args()
    train = config_from_args(TrainConfig, a)
    net = config_from_args(HybridConfig, a)
    dcfg = SeqDataConfig(lookback=1, scale=a.scale)
    forcing, target, statics = load_frames(a.forcing, a.target)
    d = SeqData.build(forcing, target, statics, dcfg)
    print(f"hybrid: {len(d)} samples, {len(d.stations)} stations, panel {d.F.shape[:2]}, statics {d.S.shape[1]}")
    print(f"net   {net}\ntrain {train}")
    tag = a.tag or f"nn_hybrid_{train.loss}"
    if a.cmd == "cv":
        wf = cv.StratifiedWeights() if a.weighted else None
        out = cv.run_dataset(d, functools.partial(HybridModel, dcfg, net, train), a.design, a.workers,
                             True, weight_fn=wf)
        cv.print_summary(f"{tag} @{a.design}", out)
        path = a.out or f"data/{tag}_{a.design}cv_predictions.csv"
        out.to_csv(path, index=False)
        print(f"wrote {path}")
    else:
        m = HybridModel(dcfg, net, train, verbose=a.verbose).fit_data(d)
        print(m.station_params(d).round(3).to_string())
        print("saved", m.save(a.out or f"data/models/{tag}.pt"))


if __name__ == "__main__":
    main()
