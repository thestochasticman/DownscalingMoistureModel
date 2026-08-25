"""One training run: a net, a TrainConfig, a training set and an optional
validation set -> trained net + epoch history.

The whole training set lives on the device and mini-batches are index slices
(the tables are tens of thousands of rows; a DataLoader would be the bottleneck).
AdamW + one-cycle LR (warm-up, cosine anneal), gradient clipping, bf16 autocast,
optional Gaussian input noise, early stopping on validation NSE with best-weight
restore.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from emt.nn import losses
from emt.nn.config import TrainConfig
from emt.nn.data import Scaler, TabularData


def device_for(cfg: TrainConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, cfg: TrainConfig, scaler: Scaler, verbose: bool = False):
        self.cfg, self.scaler, self.verbose = cfg, scaler, verbose
        self.device = device_for(cfg)
        self.loss_fn = losses.get(cfg.loss)

    def _tensors(self, d: TabularData):
        T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)  # noqa: E731
        return (T(self.scaler.x(d.X)), T(self.scaler.y(d.y)), T(d.weight),
                T(d.station_sigma(self.scaler)))

    def fit(self, net: torch.nn.Module, train: TabularData, val: TabularData | None,
            seed: int) -> list[dict]:
        cfg = self.cfg
        torch.manual_seed(seed)
        # per-column noise scale: statics get cfg.static_noise, the rest cfg.input_noise
        noise = torch.full((train.X.shape[1],), cfg.input_noise, device=self.device)
        if train.cfg.static_idx:
            noise[list(train.cfg.static_idx)] = cfg.static_noise
        use_noise = bool((noise > 0).any())
        net = net.to(self.device)
        Xtr, ytr, wtr, str_ = self._tensors(train)
        Xva, yva = self._tensors(val)[:2] if val is not None and len(val) else (None, None)
        use_amp = cfg.amp and self.device.type == "cuda"
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp)  # noqa: E731

        n = len(train)
        steps = max(1, math.ceil(n / cfg.batch_size))
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps, pct_start=cfg.warmup_frac,
            anneal_strategy="cos", div_factor=25.0, final_div_factor=1e3)
        gen = torch.Generator().manual_seed(seed)

        best, best_state, bad, history = -np.inf, None, 0, []
        for epoch in range(cfg.epochs):
            net.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            total = torch.zeros((), device=self.device)
            for i in range(steps):
                idx = perm[i * cfg.batch_size:(i + 1) * cfg.batch_size]
                xb = Xtr[idx]
                if use_noise:
                    xb = xb + noise * torch.randn_like(xb)
                with autocast():
                    pred = net(xb)
                loss = self.loss_fn(pred.float(), ytr[idx], wtr[idx], str_[idx],
                                    eps=cfg.nse_eps, delta=cfg.huber_delta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad)
                opt.step()
                sched.step()
                total += loss.detach()
            rec = dict(epoch=epoch, train_loss=total.item() / steps, lr=sched.get_last_lr()[0])

            if Xva is not None:
                net.eval()
                with torch.no_grad(), autocast():
                    pv = net(Xva).float()
                rec["val_nse"] = float(1 - ((pv - yva) ** 2).sum() / ((yva - yva.mean()) ** 2).sum())
                if rec["val_nse"] > best + 1e-4:
                    best, bad, best_state = rec["val_nse"], 0, copy.deepcopy(net.state_dict())
                else:
                    bad += 1
            history.append(rec)
            if self.verbose:
                print("  " + " ".join(f"{k}={v:.4g}" for k, v in rec.items()), flush=True)
            if Xva is not None and bad >= cfg.patience:
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def predict(self, net: torch.nn.Module, X: np.ndarray, chunk: int = 65536) -> np.ndarray:
        net = net.to(self.device).eval()
        xt = torch.as_tensor(self.scaler.x(X), dtype=torch.float32, device=self.device)
        out = [net(xt[i:i + chunk]).float().cpu().numpy() for i in range(0, len(xt), chunk)]
        return self.scaler.y_inv(np.concatenate(out) if out else np.zeros(0))
