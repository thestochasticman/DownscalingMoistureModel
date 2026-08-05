"""model9 figure: what a per-site readout changes, and where it pays.

(a) The output range each model can produce per station, against what was
observed -- model8's single global band versus model9's per-site bands, with
the stations whose observations sit above their ceiling marked.
(b) Per-station blocked NSE change, model8 -> model9, coloured by block: the
gain is concentrated in the wet Adelong cluster, not spread evenly.

Reads data/process_pedotransfer_statics.csv, data/process_target_2006_2010.csv
and the two blocked prediction tables; loads the fitted models for their
readout constants. Nothing re-fits.

Run from repo root::  PYTHONPATH=. python handout/plot_model9_readout.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emt.evaluation import metrics
from emt.model8 import model as m8
from emt.model9 import model as m9
from emt.persist import load_model

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "model9_readout.png"
TARGET = "sm_rootzone_pct"
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728",
              "MURRUMBIDGEE": "#9467bd"}

tab = pd.read_csv(REPO / "data" / "process_target_2006_2010.csv")
site_of = tab.drop_duplicates("station").set_index("station")["site"]
obs = tab.groupby("station")[TARGET].agg(
    p5=lambda s: s.quantile(.05), p95=lambda s: s.quantile(.95), mean="mean")

e8, e9 = load_model("model8"), load_model("model9")
lim = m9.load_readout_limits()
st8 = m8.load_statics()

# per-station offsets (same standardisation the fits used)
def offsets(est, statics):
    z = (statics[est._static_vars] - est._static_mean) / est._static_std
    x = est.params_.to_numpy()
    n = getattr(est, "_n_process", 5)
    return pd.Series(z.to_numpy() @ x[n:-1] + x[-1], index=statics.index)

off8, off9 = offsets(e8, st8), offsets(e9, st8)
p8 = e8.params_
lo8 = p8["theta_r"] + off8
hi8 = p8["theta_r"] + p8["dtheta"] + off8
lo9 = lim["theta_r"] + off9
hi9 = lim["theta_r"] + e9.params_["gamma"] * lim["dtheta"] + off9

order = obs["mean"].sort_values().index
x = np.arange(len(order))

fig, ax = plt.subplots(1, 2, figsize=(15, 5.8), width_ratios=[1.35, 1])

# (a) achievable range vs observed
for i, s in enumerate(order):
    ax[0].plot([i - .22, i - .22], [lo8[s], hi8[s]], lw=4, color="#bbb",
               solid_capstyle="butt", zorder=2,
               label="model8 achievable range" if i == 0 else None)
    ax[0].plot([i + .22, i + .22], [lo9[s], hi9[s]], lw=4, color="#2ca02c",
               solid_capstyle="butt", zorder=2,
               label="model9 achievable range" if i == 0 else None)
    ax[0].plot([i, i], [obs.p5[s], obs.p95[s]], lw=1.1, color="k", zorder=3)
    ax[0].plot(i, obs["mean"][s], "o", ms=3.5, color="k", zorder=4,
               label="observed mean (bar = p5–p95)" if i == 0 else None)
over = [s for s in order if obs["mean"][s] > hi9[s]]
for s in over:
    ax[0].annotate(s, (list(order).index(s), obs["mean"][s]),
                   textcoords="offset points", xytext=(0, 7), fontsize=8,
                   ha="center", color="#A6392F", fontweight="bold")
ax[0].set_xticks(x)
ax[0].set_xticklabels(order, rotation=90, fontsize=6.5)
ax[0].set(ylabel="root-zone soil moisture (%)",
          title="(a) What each model can output, per station "
                "(stations ordered dry → wet)")
ax[0].legend(fontsize=8.5, loc="upper left")
ax[0].grid(alpha=.3, axis="y")

# (b) per-station blocked NSE change
b8 = pd.read_csv(REPO / "data" /
                 "model8_blockcv_capacity_aridity_weighted_predictions.csv")
b9 = pd.read_csv(REPO / "data" / "model9_blockcv_predictions.csv")
n8 = b8.groupby("station").apply(lambda g: metrics(g[TARGET], g["pred"])["nse"],
                                 include_groups=False)
n9 = b9.groupby("station").apply(lambda g: metrics(g[TARGET], g["pred"])["nse"],
                                 include_groups=False)
d = (n9 - n8).sort_values()
cols = [SITE_COLOR[site_of[s]] for s in d.index]
ax[1].barh(np.arange(len(d)), d.values, color=cols)
ax[1].axvline(0, color="k", lw=.8)
ax[1].set_yticks(np.arange(len(d)))
ax[1].set_yticklabels(d.index, fontsize=6.5)
ax[1].set(xlabel="change in blocked NSE, model8 → model9",
          title=f"(b) Where the gain is  ({int((d > 0).sum())}/{len(d)} stations improved)")
ax[1].grid(alpha=.3, axis="x")
handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=s)
           for s, c in SITE_COLOR.items()]
ax[1].legend(handles=handles, fontsize=8, loc="lower right")

fig.suptitle("model9: replacing two global readout constants with per-site "
             "soil hydraulic limits", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG, dpi=130)
plt.close(fig)
print(f"wrote {FIG.relative_to(REPO)}")
print(f"stations whose mean observation still exceeds the model9 ceiling: {over}")
