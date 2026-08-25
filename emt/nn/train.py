"""One training run: a net, a TrainConfig, a training view and an optional
validation view -> trained net + epoch history.

A *view* is data already resident on the device (see ``DeviceView``): it knows
its length, its standardised target / weight / per-station sigma, and how to
produce the net's input for a set of row indices (optionally with training
noise). Tabular and sequence data both implement it, so this loop is shared.

AdamW + one-cycle LR (warm-up, cosine anneal), gradient clipping, bf16 autocast,
early stopping on validation NSE with best-weight restore.
"""
from __future__ import annotations

import copy
import math
from typing import Protocol

import numpy as np
import torch

from emt.nn import losses
from emt.nn.config import TrainConfig


class DeviceView(Protocol):
    n: int
    y: torch.Tensor          # (n,) standardised target
    w: torch.Tensor          # (n,) sample weight
    sigma: torch.Tensor      # (n,) per-station std of standardised target

    def batch(self, idx: torch.Tensor, noisy: bool): ...   # -> net input(s)


def device_for(cfg: TrainConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, cfg: TrainConfig, verbose: bool = False):
        self.cfg, self.verbose = cfg, verbose
        self.device = device_for(cfg)
        self.loss_fn = losses.get(cfg.loss)
        self.use_amp = cfg.amp and self.device.type == "cuda"

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp)

    def fit(self, net: torch.nn.Module, train: DeviceView, val: DeviceView | None,
            seed: int) -> list[dict]:
        cfg = self.cfg
        torch.manual_seed(seed)
        net = net.to(self.device)
        n = train.n
        steps = max(1, math.ceil(n / cfg.batch_size))
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        total = cfg.epochs * steps
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=total,
            pct_start=min(0.9, max(cfg.warmup_frac, 1.5 / total)),   # >= 1 warm-up step
            anneal_strategy="cos", div_factor=25.0, final_div_factor=1e3)
        gen = torch.Generator().manual_seed(seed)

        best, best_state, bad, history = -np.inf, None, 0, []
        for epoch in range(cfg.epochs):
            net.train()
            perm = torch.randperm(n, generator=gen).to(self.device)
            total = torch.zeros((), device=self.device)
            for i in range(steps):
                idx = perm[i * cfg.batch_size:(i + 1) * cfg.batch_size]
                with self.autocast():
                    pred = net(train.batch(idx, noisy=True))
                loss = self.loss_fn(pred.float(), train.y[idx], train.w[idx], train.sigma[idx],
                                    eps=cfg.nse_eps, delta=cfg.huber_delta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad)
                opt.step()
                sched.step()
                total += loss.detach()
            rec = dict(epoch=epoch, train_loss=total.item() / steps, lr=sched.get_last_lr()[0])

            if val is not None and val.n:
                pv = self.predict_view(net, val)
                rec["val_nse"] = float(1 - ((pv - val.y) ** 2).sum() / ((val.y - val.y.mean()) ** 2).sum())
                if rec["val_nse"] > best + 1e-4:
                    best, bad, best_state = rec["val_nse"], 0, copy.deepcopy(net.state_dict())
                else:
                    bad += 1
            history.append(rec)
            if self.verbose:
                print("  " + " ".join(f"{k}={v:.4g}" for k, v in rec.items()), flush=True)
            if val is not None and bad >= cfg.patience:
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def predict_view(self, net: torch.nn.Module, view: DeviceView, chunk: int = 4096) -> torch.Tensor:
        """Standardised predictions for every row of ``view`` (on device)."""
        net = net.to(self.device).eval()
        out = []
        for i in range(0, view.n, chunk):
            idx = torch.arange(i, min(i + chunk, view.n), device=self.device)
            with self.autocast():
                out.append(net(view.batch(idx, noisy=False)).float())
        return torch.cat(out) if out else torch.zeros(0, device=self.device)
