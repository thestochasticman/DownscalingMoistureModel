"""Station-out comparison: the legacy 2006-2010 model against the 2002-2013 refit.

Both models are scored **leave-one-station-out** -- the held-out station's own
observations are never in training, though its cluster neighbours are, so this
is the interpolation estimate rather than the transfer one.

The two runs cover different periods, so a naive comparison would confound the
model with the sample. Every panel except the last is therefore computed on the
**50,608 station-days both runs predict** (an inner join on station and time;
the target values are bit-identical across the two files). That makes the
comparison like-for-like: same observations, same stations, same folds --
only the training record differs.

The last panel shows what the matched subset cannot: how the refit performs
across the eight years the legacy model never saw.

Run from repo root::  PYTHONPATH=. python handout/plot_station_out_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from emt.evaluation import metrics

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "station_out_comparison.png"
TARGET = "sm_rootzone_pct"
LEGACY = REPO / "data" / "model8_losocv_capacity_aridity_weighted_predictions.csv"
EXTENDED = REPO / "data" / "model8_extended_stationcv_predictions.csv"

C_LEG, C_EXT = "#8c8c8c", "#2ca02c"
C_UP, C_DOWN = "#2ca02c", "#c44e52"


def per_station(df: pd.DataFrame, pred: str) -> pd.DataFrame:
    return df.groupby("station").apply(
        lambda g: pd.Series(metrics(g[TARGET], g[pred])), include_groups=False)


def main() -> None:
    leg = pd.read_csv(LEGACY, parse_dates=["time"])
    ext = pd.read_csv(EXTENDED, parse_dates=["time"])
    m = leg.merge(ext[["station", "time", "pred"]], on=["station", "time"],
                  suffixes=("_leg", "_ext"))
    m = m.rename(columns={"sm_rootzone_pct": TARGET})
    sl, se = per_station(m, "pred_leg"), per_station(m, "pred_ext")
    pl, pe = metrics(m[TARGET], m["pred_leg"]), metrics(m[TARGET], m["pred_ext"])

    fig = plt.figure(figsize=(14, 19))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[0.85, 2.5, 0.85, 0.85],
                  hspace=0.30, wspace=0.22)
    fig.suptitle("Leave-one-station-out: legacy 2006–2010 model vs 2002–2013 refit",
                 fontsize=15, fontweight="bold", y=0.975)
    fig.text(0.5, 0.955, f"panels 1–3 on the {len(m):,} station-days both runs "
             f"predict — identical observations, so only the training record differs",
             ha="center", fontsize=9.5, style="italic", color="#444")

    # --- row 1: held-out prediction against observation -------------------
    lims = (0, 60)
    for ax, pred, ttl, p, c in ((fig.add_subplot(gs[0, 0]), "pred_leg",
                                 "legacy: trained on 2006–2010", pl, C_LEG),
                                (fig.add_subplot(gs[0, 1]), "pred_ext",
                                 "refit: trained on 2002–2013", pe, C_EXT)):
        ax.hexbin(m[TARGET], m[pred], gridsize=55, extent=(*lims, *lims),
                  cmap="viridis", bins="log", mincnt=1, linewidths=0)
        ax.plot(lims, lims, color="white", lw=1.4, ls="--", zorder=3)
        ax.set(xlim=lims, ylim=lims, xlabel="observed VWC (%)",
               ylabel="predicted VWC (%)", title=ttl, aspect="equal")
        ax.set_title(ttl, fontsize=11, color=c, fontweight="bold")
        ax.text(0.04, 0.96, f"NSE {p['nse']:+.3f}\nr {p['r']:.2f}\n"
                f"ubRMSE {p['ubrmse']:.2f}\nbias {p['bias']:+.2f}",
                transform=ax.transAxes, va="top", fontsize=9.5, color="white",
                bbox=dict(boxstyle="round,pad=0.4", fc="black", alpha=0.55, ec="none"))

    # --- row 2: per-station NSE, legacy -> refit ---------------------------
    ax = fig.add_subplot(gs[1, :])
    d = pd.DataFrame({"leg": sl["nse"], "ext": se["nse"]}).dropna()
    d["delta"] = d.ext - d.leg
    d = d.sort_values("delta")
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.plot([r.leg, r.ext], [yi, yi], color=C_UP if r.delta > 0 else C_DOWN,
                lw=2.0, alpha=0.65, zorder=1, solid_capstyle="round")
    ax.scatter(d.leg, y, s=34, color=C_LEG, zorder=3, label="legacy 2006–2010")
    ax.scatter(d.ext, y, s=34, color=C_EXT, zorder=3, label="refit 2002–2013")
    ax.axvline(0, color="#333", lw=1.0, ls=":", zorder=0)
    ax.set_yticks(y, d.index, fontsize=8.5)
    XMIN = -2.0
    ax.set_xlim(XMIN, 1.0)
    ax.set_xlabel("held-out station NSE  (x-axis clipped at −2)")

    # Stations worse than the clip would otherwise vanish from the panel.
    # Silently dropping them would read as "no data"; print the real values.
    for yi, (stn, r) in zip(y, d.iterrows()):
        if min(r.leg, r.ext) >= XMIN:
            continue
        ax.annotate("", xy=(XMIN + 0.015, yi), xytext=(XMIN + 0.16, yi),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2))
        ax.text(XMIN + 0.19, yi, f"{r.leg:.1f} → {r.ext:.1f}", va="center",
                fontsize=7.5, color="#444", style="italic")
    n_up = int((d.delta > 0).sum())
    ax.set_title(f"Per-station skill: {n_up} of {len(d)} stations improve   "
                 f"(median {d.leg.median():+.3f} → {d.ext.median():+.3f})",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)

    # --- row 3 left: bias ---------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(sl["bias"].abs(), se["bias"].abs(), s=42, color=C_EXT,
               edgecolor="white", lw=0.6, zorder=3)
    hi = float(np.nanmax([sl["bias"].abs().max(), se["bias"].abs().max()])) * 1.1
    ax.plot([0, hi], [0, hi], color="#333", ls="--", lw=1.1)
    ax.fill_between([0, hi], [0, 0], [0, hi], color=C_UP, alpha=0.07)
    ax.text(hi * 0.62, hi * 0.16, "refit closer\nto zero bias", fontsize=9,
            color="#2a6", ha="center")
    better = int((se["bias"].abs() < sl["bias"].abs()).sum())
    ax.set(xlim=(0, hi), ylim=(0, hi), xlabel="legacy |bias| (pp)",
           ylabel="refit |bias| (pp)", aspect="equal")
    ax.set_title(f"Station bias magnitude: {better}/{len(sl)} improve",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)

    # --- row 3 right: NSE distribution -------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    bins = np.linspace(-1.0, 1.0, 25)
    ax.hist(sl["nse"].clip(-1, 1), bins=bins, alpha=0.55, color=C_LEG,
            label=f"legacy (median {sl['nse'].median():+.2f})")
    ax.hist(se["nse"].clip(-1, 1), bins=bins, alpha=0.55, color=C_EXT,
            label=f"refit (median {se['nse'].median():+.2f})")
    ax.axvline(0, color="#333", ls=":", lw=1.0)
    ax.set(xlabel="station NSE (clipped to ±1)", ylabel="stations")
    ax.set_title("Distribution of station skill", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    # --- row 4: the years the legacy model never saw ------------------------
    ax = fig.add_subplot(gs[3, :])
    ey = ext.copy()
    ey["year"] = ey.time.dt.year
    yr = ey.groupby("year").apply(
        lambda g: pd.Series(metrics(g[TARGET], g["pred"])), include_groups=False)
    nstn = ey.groupby("year").station.nunique()
    seen = set(range(2006, 2011))
    cols = [C_LEG if y in seen else C_EXT for y in yr.index]
    ax.bar(yr.index, yr["nse"], color=cols, width=0.72, edgecolor="white")
    for x, v, n in zip(yr.index, yr["nse"], nstn):
        ax.text(x, v + (0.012 if v >= 0 else -0.03), f"{n}", ha="center",
                fontsize=8, color="#555", va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="#333", lw=1.0)
    ax.set(xlabel="year", ylabel="pooled NSE", xticks=list(yr.index))
    ax.set_title("Refit, station-out skill by year — grey = years the legacy model "
                 "was trained on, green = the eight it never saw\n"
                 "number above each bar is the count of reporting stations: "
                 "year folds are NOT like-for-like",
                 fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=9)

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {FIG}")
    print(f"matched {len(m):,} rows / {m.station.nunique()} stations")
    print(f"  legacy  NSE {pl['nse']:+.3f}  r {pl['r']:.2f}  bias {pl['bias']:+.2f}")
    print(f"  refit   NSE {pe['nse']:+.3f}  r {pe['r']:.2f}  bias {pe['bias']:+.2f}")
    print(f"  stations improved {n_up}/{len(d)}; |bias| improved {better}/{len(sl)}")
    print(f"  median station NSE {sl['nse'].median():+.3f} -> {se['nse'].median():+.3f}")


if __name__ == "__main__":
    main()
