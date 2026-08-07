"""Where the remaining error actually lives: level, not dynamics.

Six stations score below −2 NSE under every configuration from model7 to
model10, and K12 sits at −14.6. The natural reading is that the model fails
there. It does not — those same stations carry the *lowest* absolute errors in
the network (A4's ubRMSE is 0.91 pp, the best of 37, at NSE −2.08).

The resolution is that NSE is variance-normalised and charges the full cost of
a constant offset, so decomposing MSE into its two parts separates the two
failures cleanly:

    MSE = bias² + ubRMSE²
          level   dynamics

Run from repo root::  PYTHONPATH=. python handout/plot_error_decomposition.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emt.evaluation import metrics

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "error_decomposition.png"
PRED = REPO / "data" / "model8_losocv_capacity_aridity_weighted_predictions.csv"
TARGET = "sm_rootzone_pct"
C_LEVEL, C_DYN = "#c44e52", "#4c72b0"


def main() -> None:
    p = pd.read_csv(PRED, parse_dates=["time"])
    rows = []
    for s, g in p.groupby("station"):
        m = metrics(g[TARGET], g["pred"])
        # Oracle de-biasing: remove each station's mean offset using the
        # held-out truth. NOT achievable in deployment -- it is an upper bound
        # on what a perfect level correction could recover, nothing more.
        deb = g["pred"] - (g["pred"].mean() - g[TARGET].mean())
        rows.append(dict(
            station=s, nse=m["nse"], bias=m["bias"], ubrmse=m["ubrmse"], r=m["r"],
            bias_sq=m["bias"] ** 2, ubrmse_sq=m["ubrmse"] ** 2,
            bias_frac=m["bias"] ** 2 / m["rmse"] ** 2,
            nse_debiased=metrics(g[TARGET], deb)["nse"], obs_sd=g[TARGET].std()))
    d = pd.DataFrame(rows).set_index("station").sort_values("bias_frac")

    fig, axes = plt.subplots(1, 3, figsize=(18, 10),
                             gridspec_kw={"width_ratios": [1.5, 1, 1], "wspace": 0.28})
    fig.suptitle("The residual error is a level offset, not a dynamics failure",
                 fontsize=16, fontweight="bold", y=0.965)
    fig.text(0.5, 0.935, "model8, leave-one-station-out, 2006–2010   ·   "
             "MSE = bias² + ubRMSE²", ha="center", fontsize=10.5,
             style="italic", color="#444")

    # --- 1: stacked MSE decomposition -------------------------------------
    ax = axes[0]
    y = np.arange(len(d))
    ax.barh(y, d.bias_sq, color=C_LEVEL, label="bias² (level)")
    ax.barh(y, d.ubrmse_sq, left=d.bias_sq, color=C_DYN, label="ubRMSE² (dynamics)")
    ax.set_yticks(y, d.index, fontsize=8.5)
    ax.set_xlabel("mean squared error (%²)")
    ax.set_title(f"Per-station MSE split\nlevel is {d.bias_frac.median()*100:.0f}% "
                 f"of the median station's error", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.7, len(d) - 0.3)

    # --- 2: dynamics are fine everywhere ----------------------------------
    ax = axes[1]
    colours = [C_LEVEL if v < 0 else C_DYN for v in d.nse]
    ax.scatter(d.r, d.nse.clip(-6, 1), s=60, c=colours, edgecolor="white", lw=0.7)
    ax.axhline(0, color="#333", ls=":", lw=1.1)
    ax.axvline(0.5, color="#999", ls="--", lw=1.0)
    n_bad = int(((d.r > 0.5) & (d.nse < 0)).sum())
    ax.fill_between([0.5, 1.0], -6, 0, color=C_LEVEL, alpha=0.07)
    ax.text(0.75, -3.0, f"{n_bad} stations\ngood dynamics,\nwrong level",
            ha="center", fontsize=10.5, color=C_LEVEL, fontweight="bold")
    for s in ("K12", "A4", "K2", "A5"):
        ax.annotate(s, (d.loc[s, "r"], max(d.loc[s, "nse"], -6)),
                    textcoords="offset points", xytext=(6, 4), fontsize=8.5)
    ax.set(xlabel="correlation r (dynamics)", ylabel="station NSE (clipped at −6)",
           xlim=(0, 1.02), ylim=(-6.3, 1.05))
    ax.set_title(f"Dynamics vs total skill\nmedian r = {d.r.median():.2f}",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)

    # --- 3: the headroom ---------------------------------------------------
    ax = axes[2]
    o = d.sort_values("nse")
    yy = np.arange(len(o))
    ax.plot([o.nse.clip(-6), o.nse_debiased.clip(-6)], [yy, yy],
            color="#bbb", lw=1.4, zorder=1)
    ax.scatter(o.nse.clip(-6), yy, s=34, color=C_LEVEL, zorder=3, label="as fitted")
    ax.scatter(o.nse_debiased.clip(-6), yy, s=34, color="#55a868", zorder=3,
               label="level offset removed (oracle)")
    ax.axvline(0, color="#333", ls=":", lw=1.1)
    ax.set_yticks(yy, o.index, fontsize=8)
    ax.set(xlabel="station NSE (clipped at −6)", xlim=(-6.3, 1.05))
    ax.set_title(f"Headroom if level were solved\nmedian {o.nse.median():+.2f} → "
                 f"{o.nse_debiased.median():+.2f}   ·   "
                 f"{int((o.nse > 0).sum())}/37 → {int((o.nse_debiased > 0).sum())}/37 "
                 f"positive", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.7, len(o) - 0.3)

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {FIG}")
    print(f"median bias share of MSE   {d.bias_frac.median()*100:.0f}%")
    print(f"bias >80% of MSE           {int((d.bias_frac > 0.8).sum())}/37 stations")
    print(f"median station r           {d.r.median():.2f}")
    print(f"good dynamics, NSE<0       {n_bad} stations")
    print(f"median NSE  {d.nse.median():+.3f} -> {d.nse_debiased.median():+.3f} (oracle)")
    print(f"NSE>0       {int((d.nse > 0).sum())}/37 -> "
          f"{int((d.nse_debiased > 0).sum())}/37")


if __name__ == "__main__":
    main()
