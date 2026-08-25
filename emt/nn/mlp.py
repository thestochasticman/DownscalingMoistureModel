"""ResidualMLP: pre-norm residual blocks (LayerNorm -> Linear -> SiLU -> Dropout)."""
from __future__ import annotations

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
    def __init__(self, n_in: int, cfg: MLPConfig):
        super().__init__()
        blocks, d = [], n_in
        for h in cfg.hidden:
            blocks.append(Block(d, h, cfg.dropout, cfg.residual))
            d = h
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        return self.head(self.blocks(x)).squeeze(-1)
