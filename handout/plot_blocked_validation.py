"""Blocked-validation figure, from cached block-CV predictions.

(a) NSE per held-out block, blocks ordered dry -> wet by aridity: model6
against model8 before and after its climate fix -- the edge-of-envelope
failure and the two models' complementary strengths.
(b) per-block bias against block aridity for model8 base -> final, showing
what the aridity static + weights recover and what only new data can.

Reads data/model{6,8}_blockcv*_predictions.csv (written by run_blocked_cv.py)
and data/process_climate_statics.csv; nothing re-fits.

Run from repo root::  PYTHONPATH=. python handout/plot_blocked_validation.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emt.evaluation import metrics

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "blocked_validation.png"
TARGET = "sm_rootzone_pct"

C_M6, C_M8B, C_M8F = "#1f77b4", "#999999", "#2ca02c"
CLIP = -4.0                      # bar floor; true values annotated

runs = {
    "m6":  pd.read_csv(REPO / "data" / "model6_blockcv_predictions.csv"),
    "m8b": pd.read_csv(REPO / "data" / "model8_blockcv_predictions.csv"),
    "m8f": pd.read_csv(REPO / "data" / "model8_blockcv_aridity_weighted_predictions.csv"),
}
per_block = {k: v.groupby("block").apply(
                 lambda g: pd.Series(metrics(g[TARGET], g["pred"])),
                 include_groups=False)
             for k, v in runs.items()}

# Block aridity (station mean) orders the x-axis dry -> wet.
clim = pd.read_csv(REPO / "data" / "process_climate_statics.csv")
clim["block"] = clim["station"].map(
    lambda s: {"Y": "YANCO", "K": "KYEAMBA", "A": "ADELONG"}.get(s[0], s))
aridity = clim.groupby("block")["aridity"].mean().sort_values()
blocks = [b for b in aridity.index if b in per_block["m8b"].index]

fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), width_ratios=[1.5, 1])

# (a) per-block NSE, dry -> wet
x = np.arange(len(blocks))
series = [("model6 (ML)", "m6", C_M6),
          ("model8", "m8b", C_M8B),
          ("model8 + aridity + weights", "m8f", C_M8F)]
for i, (label, key, color) in enumerate(series):
    vals = per_block[key].loc[blocks, "nse"]
    clipped = vals.clip(lower=CLIP)
    bars = ax[0].bar(x + (i - 1) * 0.27, clipped, width=0.25, color=color,
                     label=label, zorder=2)
    for xi, (v, c) in zip(x, zip(vals, clipped)):
        if v < CLIP:             # annotate off-scale values instead of hiding them
            ax[0].annotate(f"{v:+.1f}", (xi + (i - 1) * 0.27, CLIP + 0.1),
                           ha="center", va="bottom", fontsize=7, rotation=90,
                           color="k")
ax[0].axhline(0, color="k", lw=.8)
ax[0].set_xticks(x)
ax[0].set_xticklabels([f"{b}\n{aridity[b]:.2f}" for b in blocks], fontsize=8)
ax[0].set(ylim=[CLIP - 0.15, 1.0], ylabel="NSE of the held-out block",
          xlabel="held-out block, ordered dry → wet (block mean P/PET below name)",
          title="(a) Leave-one-block-out NSE — failure sits at the climate-envelope edges")
ax[0].legend(fontsize=9, loc="lower center")
ax[0].grid(alpha=.3, axis="y", zorder=0)

# (b) per-block bias vs aridity: model8 base -> final
b0 = per_block["m8b"].loc[blocks, "bias"]
b1 = per_block["m8f"].loc[blocks, "bias"]
ar = aridity[blocks]
for blk in blocks:
    ax[1].annotate("", xy=(ar[blk], b1[blk]), xytext=(ar[blk], b0[blk]),
                   arrowprops=dict(arrowstyle="->", color="#bbb", lw=1.2))
ax[1].scatter(ar, b0, s=42, color=C_M8B, label="model8", zorder=3)
ax[1].scatter(ar, b1, s=46, color=C_M8F, edgecolor="k", linewidth=.4,
              label="model8 + aridity + weights", zorder=4)
for blk in ("M7", "ADELONG", "M2"):
    ax[1].annotate(blk, (ar[blk], b1[blk]), textcoords="offset points",
                   xytext=(6, -3), fontsize=8)
ax[1].axhline(0, color="k", lw=.8)
ax[1].set(xlabel="block mean aridity (P/PET)", ylabel="held-out block bias (pred−obs, %)",
          title="(b) model8 block bias vs climate — the fix helps, the edges remain")
ax[1].legend(fontsize=9, loc="lower left")
ax[1].grid(alpha=.3)

fig.suptitle("Blocked validation (9 spatially independent blocks, 2006–2010): "
             "transfer skill, not interpolation", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG, dpi=130)
plt.close(fig)
print(f"wrote {FIG.relative_to(REPO)}")
