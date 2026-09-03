"""Held-out per-station series for each NN-track model, from cached CV
predictions (one figure per model; nothing retrains here).

Run from repo root::  PYTHONPATH=. python handout/plot_nn_per_station.py [mlp|seq|hybrid ...]
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from emt.evaluation import metrics

REPO = Path(__file__).resolve().parent.parent
TARGET = "sm_rootzone_pct"
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728",
              "MURRUMBIDGEE": "#9467bd"}
SITES = ["ADELONG", "KYEAMBA", "MURRUMBIDGEE", "YANCO"]

MODELS = {  # key: (predictions csv, figure, title prefix)
    "mlp": ("nn_B_log_snoise_stationcv_predictions.csv", "nn_mlp_per_station.png",
            "nn-mlp (residual MLP on model6 features)"),
    "seq": ("nn_seq_nse_stationcv_predictions.csv", "nn_seq_per_station.png",
            "nn-transformer (365-day SILO forcing window, no SMIPS)"),
    "hybrid": ("nn_hybrid_q_stationcv_predictions.csv", "nn_hybrid_per_station.png",
               "nn-hybrid (differentiable bucket, quantile statics)"),
}

tab = pd.read_csv(REPO / "data" / "process_target_2006_2010.csv")
site_of = tab.drop_duplicates("station").set_index("station")["site"]


def make(key: str) -> None:
    csv, figname, title = MODELS[key]
    pred = pd.read_csv(REPO / "data" / csv, parse_dates=["time"])
    pooled = metrics(pred[TARGET], pred["pred"])
    ps = (pred.groupby("station").apply(lambda g: pd.Series(metrics(g[TARGET], g["pred"])),
                                        include_groups=False))
    stations = sorted(pred.station.unique(), key=lambda s: (SITES.index(site_of[s]), s))
    ncols = 6
    nrows = (len(stations) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.4), sharex=True)
    axes = axes.ravel()
    for ax, stn in zip(axes, stations):
        g = pred[pred.station == stn].sort_values("time")
        c = SITE_COLOR[site_of[stn]]
        ax.plot(g["time"], g[TARGET], color="k", lw=0.7, label="observed")
        ax.plot(g["time"], g["pred"], color=c, lw=0.9, label="held-out prediction")
        m = ps.loc[stn]
        ax.set_title(f"{stn}   r={m.r:.2f}  NSE={m.nse:.2f}  bias={m.bias:+.1f}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=.25)
    for ax in axes[len(stations):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{title} leave-site-out prediction vs observation, all {len(stations)} "
                 f"stations (2006-2010).  Pooled NSE = {pooled['nse']:+.2f}, r = {pooled['r']:.2f}  "
                 f"(purple = regional M-sites)", fontsize=13, y=0.995)
    fig.supylabel("root-zone soil moisture (%)", fontsize=10)
    fig.tight_layout(rect=[0.01, 0, 1, 0.985])
    fig.savefig(REPO / "handout" / "figures" / figname, dpi=110)
    plt.close(fig)
    print(f"wrote handout/figures/{figname}")


if __name__ == "__main__":
    import sys
    for key in (sys.argv[1:] or list(MODELS)):
        make(key)
