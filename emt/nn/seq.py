"""The sequence model: a Transformer over the SILO forcing window.

Inputs for one sample (station *s*, day *t*): the previous ``lookback`` days of
forcing at *s* -- rain, PET, VPD, plus day-of-year sin/cos -- and the station
statics (soil, terrain, aridity). No SMIPS: this is the neural analogue of the
model7/8 bucket, learning the water balance from forcing instead of simulating
it, and it can run for any date the forcing exists.

    tokens = [static token] + [forcing token(t-L+1) ... forcing token(t)]
    pre-norm Transformer encoder, learned positional embedding
    readout at the last token (day t) -> linear -> soil moisture (standardised)

Windows are gathered on the fly from a per-station forcing tensor resident on
the GPU (``SeqView.batch``), so 50k samples x 365 days is never materialised.

``SeqData`` / ``SeqView`` implement the same protocol as the tabular classes,
so the Trainer, the ensemble, and the CV ladder are shared.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from emt.nn.config import SeqDataConfig, TrainConfig, TransformerConfig
from emt.nn.data import Scaler
from emt.nn.train import Trainer

DATA = Path("data")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_frames(forcing_csv=DATA / "process_forcing_2005_2010.csv",
                target_csv=DATA / "process_target_2006_2010.csv",
                statics_csvs=(DATA / "process_soil_statics.csv", DATA / "process_terrain_statics.csv",
                              DATA / "process_climate_statics.csv")):
    forcing = pd.read_csv(forcing_csv, parse_dates=["time"])
    target = pd.read_csv(target_csv, parse_dates=["time"])
    statics = None
    for p in statics_csvs:
        s = pd.read_csv(p)
        statics = s if statics is None else statics.merge(s, on="station")
    return forcing, target, statics


@dataclass
class SeqData:
    """Per-station forcing panels plus (station, day) target samples."""
    F: np.ndarray            # (n_stations, T, n_forcing + 2)  forcing + doy sin/cos, raw units
    S: np.ndarray            # (n_stations, n_static)           raw units
    stations: np.ndarray     # (n_stations,) names; row order of F and S
    t0: np.datetime64        # date of column 0 of F
    # samples
    stn_idx: np.ndarray      # (n,) row into F/S
    pos: np.ndarray          # (n,) column into F = the target day
    y: np.ndarray            # (n,) float32
    weight: np.ndarray       # (n,) float32, mean 1
    cfg: SeqDataConfig

    @classmethod
    def build(cls, forcing: pd.DataFrame, target: pd.DataFrame, statics: pd.DataFrame,
              cfg: SeqDataConfig = SeqDataConfig(), weight: np.ndarray | None = None) -> "SeqData":
        stations = np.array(sorted(set(target[cfg.group]) & set(forcing[cfg.group]) & set(statics[cfg.group])))
        t0, t1 = forcing[cfg.time].min(), forcing[cfg.time].max()
        days = pd.date_range(t0, t1, freq="D")
        T = len(days)
        F = np.full((len(stations), T, len(cfg.forcing) + 2), np.nan, np.float32)
        doy = days.dayofyear.to_numpy() / 365.25 * 2 * np.pi
        for i, s in enumerate(stations):
            f = forcing[forcing[cfg.group] == s].set_index(cfg.time).reindex(days)
            F[i, :, :len(cfg.forcing)] = f[list(cfg.forcing)].to_numpy(np.float32)
            F[i, :, -2], F[i, :, -1] = np.sin(doy), np.cos(doy)
        # forward-fill short forcing gaps, then zero-fill (rain) / column-mean (others)
        for i in range(len(stations)):
            df = pd.DataFrame(F[i]).ffill(limit=7)
            F[i] = df.fillna(df.mean()).fillna(0).to_numpy(np.float32)
        S = statics.set_index(cfg.group).loc[stations, list(cfg.statics)].to_numpy(np.float32)

        tgt = target.dropna(subset=[cfg.target])
        tgt = tgt[tgt[cfg.group].isin(stations)]
        row = pd.Series(np.arange(len(stations)), index=stations)
        stn_idx = row[tgt[cfg.group].to_numpy()].to_numpy()
        pos = ((tgt[cfg.time] - t0).dt.days).to_numpy()
        ok = pos >= cfg.lookback - 1                      # full window available
        w = np.ones(len(tgt), np.float32) if weight is None else np.asarray(weight, np.float32)
        d = cls(F, S, stations, np.datetime64(t0, "D"), stn_idx[ok], pos[ok],
                tgt[cfg.target].to_numpy(np.float32)[ok], w[ok], cfg)
        d.weight = d.weight * (len(d.weight) / d.weight.sum())
        return d

    # protocol shared with TabularData ---------------------------------------
    def __len__(self): return len(self.y)

    @property
    def station(self) -> np.ndarray: return self.stations[self.stn_idx]

    @property
    def time(self) -> np.ndarray: return self.t0 + self.pos.astype("timedelta64[D]")

    def subset(self, mask: np.ndarray) -> "SeqData":
        return SeqData(self.F, self.S, self.stations, self.t0, self.stn_idx[mask], self.pos[mask],
                       self.y[mask], self.weight[mask], self.cfg)

    @property
    def station_codes(self):
        codes, uniq = pd.factorize(self.station)
        return codes, np.asarray(uniq)

    def station_sigma(self, scaler: Scaler) -> np.ndarray:
        codes, uniq = self.station_codes
        sig = pd.Series(scaler.y(self.y)).groupby(codes).std().reindex(range(len(uniq))).fillna(1.0)
        return sig.to_numpy(np.float32)[codes]

    def grouped_split(self, frac: float, rng: np.random.Generator):
        if frac <= 0:
            return np.ones(len(self), bool), np.zeros(len(self), bool)
        codes, uniq = self.station_codes
        n_val = max(1, int(round(frac * len(uniq))))
        va = np.isin(codes, rng.choice(len(uniq), n_val, replace=False))
        return ~va, va

    def frame(self, pred: np.ndarray | None = None) -> pd.DataFrame:
        out = pd.DataFrame({self.cfg.group: self.station, self.cfg.time: self.time,
                            self.cfg.target: self.y})
        if pred is not None:
            out["pred"] = pred
        return out


@dataclass(frozen=True)
class SeqScaler:
    """Forcing: log1p(rain) then z-score per channel (doy channels untouched);
    statics: log1p on chosen columns then z-score; target z-score."""
    f_mean: np.ndarray
    f_std: np.ndarray
    s: Scaler                    # statics + target

    @classmethod
    def fit(cls, d: SeqData) -> "SeqScaler":
        n_f = len(d.cfg.forcing)
        Fl = cls._lograin(d.F[:, :, :n_f], d.cfg)
        flat = Fl.reshape(-1, n_f)
        f_mean, f_std = flat.mean(0), flat.std(0) + 1e-6
        # statics scaler is fitted on the stations present in the SAMPLES (training fold)
        S_train = d.S[np.unique(d.stn_idx)]
        return cls(f_mean, f_std, Scaler.fit(S_train, d.y, d.cfg.static_log_idx, d.cfg.scale))

    @staticmethod
    def _lograin(F, cfg):
        F = F.copy()
        j = cfg.forcing.index("daily_rain") if "daily_rain" in cfg.forcing else None
        if j is not None:
            F[..., j] = np.log1p(np.clip(F[..., j], 0, None))
        return F

    def forcing(self, F: np.ndarray, cfg: SeqDataConfig) -> np.ndarray:
        n_f = len(cfg.forcing)
        out = F.copy()
        out[..., :n_f] = (self._lograin(F[..., :n_f], cfg) - self.f_mean) / self.f_std
        return out

    def y(self, y): return self.s.y(y)
    def y_inv(self, ys): return self.s.y_inv(ys)


class SeqView:
    """``SeqData`` standardised and resident on a device (see ``train.DeviceView``)."""

    def __init__(self, d: SeqData, scaler: SeqScaler, device: torch.device,
                 input_noise: float = 0.0, static_noise: float = 0.0):
        T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        self.n, self.L = len(d), d.cfg.lookback
        self.F = T(scaler.forcing(d.F, d.cfg))                 # (n_stn, T, c)
        self.S = T(scaler.s.x(d.S))                            # (n_stn, p)
        self.stn = T(d.stn_idx, torch.long)
        self.pos = T(d.pos, torch.long)
        self.y, self.w = T(scaler.y(d.y)), T(d.weight)
        self.sigma = T(d.station_sigma(scaler.s))
        self.offsets = torch.arange(-self.L + 1, 1, device=device)
        self.n_forcing = len(d.cfg.forcing)
        self.input_noise, self.static_noise = input_noise, static_noise

    def batch(self, idx: torch.Tensor, noisy: bool):
        cols = self.pos[idx, None] + self.offsets                # (B, L)
        seq = self.F[self.stn[idx, None], cols]                  # (B, L, c)
        sta = self.S[self.stn[idx]]                              # (B, p)
        if noisy:
            if self.input_noise > 0:
                noise = torch.randn_like(seq)
                noise[..., self.n_forcing:] = 0                  # leave doy channels alone
                seq = seq + self.input_noise * noise
            if self.static_noise > 0:
                sta = sta + self.static_noise * torch.randn_like(sta)
        return seq, sta


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
class SeqTransformer(nn.Module):
    def __init__(self, n_forcing_ch: int, n_static: int, lookback: int, cfg: TransformerConfig):
        super().__init__()
        d = cfg.d_model
        self.readout = cfg.readout
        self.in_proj = nn.Linear(n_forcing_ch, d)
        self.static_proj = nn.Sequential(nn.Linear(n_static, d), nn.SiLU(), nn.Dropout(cfg.static_dropout))
        self.pos_emb = nn.Parameter(torch.zeros(1, lookback + 1, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.d_ff, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, inputs):
        seq, sta = inputs
        x = torch.cat([self.static_proj(sta)[:, None, :], self.in_proj(seq)], dim=1)
        x = self.encoder(x + self.pos_emb[:, :x.shape[1]])
        h = x[:, -1] if self.readout == "last" else x.mean(1)
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class SeqModel:
    def __init__(self, data: SeqDataConfig = SeqDataConfig(), net: TransformerConfig = TransformerConfig(),
                 train: TrainConfig = TrainConfig(), verbose: bool = False):
        self.data, self.net_cfg, self.train_cfg, self.verbose = data, net, train, verbose
        self.scaler: SeqScaler | None = None
        self.nets: list[SeqTransformer] = []
        self.history: list[list[dict]] = []

    def _new_net(self, d: SeqData) -> SeqTransformer:
        return SeqTransformer(d.F.shape[-1], d.S.shape[-1], d.cfg.lookback, self.net_cfg)

    def fit_data(self, d: SeqData) -> "SeqModel":
        self.scaler = SeqScaler.fit(d)
        trainer = Trainer(self.train_cfg, self.verbose)
        view = lambda sub: SeqView(sub, self.scaler, trainer.device,  # noqa: E731
                                   self.train_cfg.input_noise, self.train_cfg.static_noise)
        self.nets, self.history = [], []
        for k in range(self.train_cfg.n_ensemble):
            seed = self.train_cfg.seed + 1000 * k
            tr, va = d.grouped_split(self.train_cfg.val_frac, np.random.default_rng(seed))
            net = self._new_net(d)
            if self.verbose:
                print(f"[member {k + 1}/{self.train_cfg.n_ensemble}] "
                      f"train {tr.sum()} rows / val {va.sum()} rows", flush=True)
            self.history.append(trainer.fit(net, view(d.subset(tr)),
                                            view(d.subset(va)) if va.any() else None, seed))
            self.nets.append(net.cpu())
        return self

    def predict_data(self, d: SeqData) -> np.ndarray:
        trainer = Trainer(self.train_cfg)
        view = SeqView(d, self.scaler, trainer.device)
        ys = torch.stack([trainer.predict_view(net, view) for net in self.nets]).mean(0)
        return self.scaler.y_inv(ys.cpu().numpy())

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"data": self.data, "net": self.net_cfg, "train": self.train_cfg,
                    "scaler": self.scaler, "history": self.history,
                    "shapes": (self.nets[0].in_proj.in_features, self.nets[0].static_proj[0].in_features),
                    "state_dicts": [n.state_dict() for n in self.nets]}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SeqModel":
        ck = torch.load(path, weights_only=False, map_location="cpu")
        m = cls(ck["data"], ck["net"], ck["train"])
        m.scaler, m.history = ck["scaler"], ck["history"]
        n_ch, n_static = ck["shapes"]
        for sd in ck["state_dicts"]:
            net = SeqTransformer(n_ch, n_static, m.data.lookback, m.net_cfg)
            net.load_state_dict(sd)
            m.nets.append(net)
        return m


# --------------------------------------------------------------------------- #
# CLI:  PYTHONPATH=. python -m emt.nn.seq cv --design station --loss nse --workers 8
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse
    import functools
    from emt.nn import cv
    from emt.nn.__main__ import add_config_args, config_from_args

    ap = argparse.ArgumentParser(prog="emt.nn.seq", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["cv", "fit"])
    ap.add_argument("--design", default="station", choices=cv.DESIGNS)
    ap.add_argument("--forcing", default=str(DATA / "process_forcing_2005_2010.csv"))
    ap.add_argument("--target", default=str(DATA / "process_target_2006_2010.csv"))
    ap.add_argument("--lookback", type=int, default=SeqDataConfig.lookback)
    ap.add_argument("--scale", default="zscore", choices=["zscore", "robust", "quantile"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    add_config_args(ap, (TrainConfig, TransformerConfig))
    a = ap.parse_args()
    train = config_from_args(TrainConfig, a)
    net = config_from_args(TransformerConfig, a)
    dcfg = SeqDataConfig(lookback=a.lookback, scale=a.scale)
    forcing, target, statics = load_frames(a.forcing, a.target)
    d = SeqData.build(forcing, target, statics, dcfg)
    print(f"seq: {len(d)} samples, {len(d.stations)} stations, lookback {dcfg.lookback}, "
          f"forcing {d.F.shape}, statics {d.S.shape[1]}")
    print(f"net   {net}\ntrain {train}")
    tag = a.tag or f"nn_seq_{train.loss}"
    if a.cmd == "cv":
        out = cv.run_dataset(d, functools.partial(SeqModel, dcfg, net, train), a.design, a.workers, True)
        cv.print_summary(f"{tag} @{a.design}", out)
        path = a.out or f"data/{tag}_{a.design}cv_predictions.csv"
        out.to_csv(path, index=False)
        print(f"wrote {path}")
    else:
        m = SeqModel(dcfg, net, train, verbose=a.verbose).fit_data(d)
        print("saved", m.save(a.out or f"data/models/{tag}.pt"))


if __name__ == "__main__":
    main()
