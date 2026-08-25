"""ResidualMLP: pre-norm residual blocks (LayerNorm -> Linear -> SiLU -> Dropout)."""
from __future__ import annotations

import torch
import torch.nn as nn

from emt.nn.config import MLPConfig


class Block(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float, residual: bool):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.lin = nn.Linear(d_in, d_out)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        if not residual:
            self.skip = None
        elif d_in == d_out:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        h = self.drop(self.act(self.lin(self.norm(x))))
        return h if self.skip is None else h + self.skip(x)


class ResidualMLP(nn.Module):
    """Residual MLP; optionally the static columns enter through a narrow,
    heavily-dropped bottleneck so the net cannot read station identity off them.

    ``static_idx`` are column positions of the static features in the input.
    With ``cfg.static_bottleneck == 0`` (or no statics) the input is used whole.
    """
    def __init__(self, n_in: int, cfg: MLPConfig, static_idx: tuple[int, ...] = ()):
        super().__init__()
        self.static_idx = list(static_idx) if cfg.static_bottleneck > 0 else []
        self.dynamic_idx = [i for i in range(n_in) if i not in self.static_idx]
        if self.static_idx:
            self.static = nn.Sequential(nn.Linear(len(self.static_idx), cfg.static_bottleneck),
                                        nn.SiLU(), nn.Dropout(cfg.static_dropout))
            d = len(self.dynamic_idx) + cfg.static_bottleneck
        else:
            self.static, d = None, n_in
        blocks = []
        for h in cfg.hidden:
            blocks.append(Block(d, h, cfg.dropout, cfg.residual))
            d = h
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.register_buffer("_dyn", torch.tensor(self.dynamic_idx, dtype=torch.long))
        self.register_buffer("_sta", torch.tensor(self.static_idx, dtype=torch.long))

    def forward(self, x):
        if self.static is not None:
            x = torch.cat([x[:, self._dyn], self.static(x[:, self._sta])], dim=1)
        return self.head(self.blocks(x)).squeeze(-1)
