"""Reproduce the three handout figures from the EMT pipeline.

Run from the repo root so the ``emt`` package and ``PaddockTS`` are importable::

    PYTHONPATH=. python handout/plot_results.py

What it does:
  1. Loads (or rebuilds) the Kyeamba 2020 Jun-Jul training table.
  2. Rebuilds the *pre-fix* SMIPS values by disabling the native-grid snap, to
     show what the resampling bug did (Figure 1).
  3. Runs the Stage 5 leave-site-out cross-validation (Figures 2-3).

Outputs (overwritten each run): ``handout/figures/*.png``.

The figures are committed so the README renders on GitHub; this script
regenerates them whenever the pipeline or training data change.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import emt.smips as smips
from emt.features import build_training_table, SMIPS_COL
from emt.model import leave_site_out_cv, fit, feature_importance, TARGET
from emt.insitu.oznet import fetch_manifest, load_daily_rootzone
from emt.insitu.coordinates import COORDS_CACHE

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "handout" / "figures"
TABLE_CSV = REPO / "data" / "train_kyeamba_2020JJ.csv"

START, END = date(2020, 6, 1), date(2020, 7, 31)
STATIONS = ["K6", "K7", "K10", "K12"]
COLORS = {"K6": "#1f77b4", "K7": "#ff7f0e", "K10": "#2ca02c", "K12": "#d62728"}


def _station_inputs():
    """Coords + daily OzNet root-zone for the four Kyeamba stations."""
    coords = pd.read_csv(COORDS_CACHE)
    man = fetch_manifest()
    coords = coords.merge(man[["site", "station"]].drop_duplicates(), on="station")
    sub = coords[coords["station"].isin(STATIONS)].copy()
    m = man[man["station"].isin(STATIONS) & (man["year"] == 2020)]
    daily = load_daily_rootzone(manifest=m, verbose=False)
    return sub, daily


def load_table(sub, daily) -> pd.DataFrame:
    """Load the cached training table, or build (and cache) it if absent."""
    if TABLE_CSV.exists():
        t = pd.read_csv(TABLE_CSV)
    else:
        t = build_training_table(sub, daily, START, END, verbose=False)
        t.to_csv(TABLE_CSV, index=False)
    t["time"] = pd.to_datetime(t["time"])
    return t


def old_smips(sub, daily) -> pd.DataFrame:
    """Pre-fix SMIPS: temporarily replace ``snap_bbox`` with the identity so the
    request bbox is *not* aligned to the native grid (the old, window-dependent
    behaviour). Returns ``station, time, smips_old``."""
    orig = smips.snap_bbox
    smips.snap_bbox = lambda bbox, pad=1: list(bbox)
    try:
        t = build_training_table(sub, daily, START, END, verbose=False)
    finally:
        smips.snap_bbox = orig
    t["time"] = pd.to_datetime(t["time"])
    return t.rename(columns={SMIPS_COL: "smips_old"})[["station", "time", "smips_old"]]


def fig_data(new, old):
    """Figure 1 -- the SMIPS correction and target-vs-SMIPS relationship."""
    df = new.merge(old, on=["station", "time"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    for st in STATIONS:
        d = df[df.station == st]
        ax[0].scatter(d["smips_old"], d[SMIPS_COL], s=18, color=COLORS[st], label=st, alpha=.8)
    lim = [40, 115]
    ax[0].plot(lim, lim, "k--", lw=1, label="1:1 (no change)")
    ax[0].set(xlim=lim, ylim=lim, xlabel="SMIPS old (resampled, mm)",
              ylabel="SMIPS new (native grid, mm)",
              title="(a) SMIPS before vs after native-grid alignment")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    shift = df.groupby("station").apply(
        lambda g: (g[SMIPS_COL] - g["smips_old"]).mean(), include_groups=False)
    bars = ax[1].bar(STATIONS, [shift[s] for s in STATIONS],
                     color=[COLORS[s] for s in STATIONS])
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set(ylabel="mean Δ SMIPS  (new − old, mm)", title="(b) Per-station SMIPS shift")
    for b, s in zip(bars, STATIONS):
        ax[1].text(b.get_x()+b.get_width()/2, b.get_height(), f"{shift[s]:+.1f}",
                   ha="center", va="bottom" if b.get_height() >= 0 else "top", fontsize=9)
    ax[1].grid(alpha=.3, axis="y")

    for st in STATIONS:
        d = new[new.station == st]
        ax[2].scatter(d[SMIPS_COL], d[TARGET], s=18, color=COLORS[st], label=st, alpha=.8)
    r = new[[SMIPS_COL, TARGET]].corr().iloc[0, 1]
    ax[2].set(xlabel="SMIPS TotalBucket (mm)", ylabel="OzNet root-zone (%)",
              title=f"(c) Target vs SMIPS  (pooled r = {r:.2f})")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "smips_correction.png", dpi=130)
    plt.close(fig)


def fig_model(new, cv, per_site, imp):
    """Figure 2 -- leave-site-out CV degeneracy + feature importance."""
    pred = cv["predictions"].copy()
    pred["time"] = pd.to_datetime(pred["time"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    for st in STATIONS:
        d = pred[pred.station == st]
        ax[0].scatter(d[TARGET], d["pred"], s=20, color=COLORS[st], alpha=.8,
                      label=f"{st} (r={per_site.loc[st,'r']:.2f}, bias={per_site.loc[st,'bias']:+.1f})")
    lim = [pred[[TARGET, "pred"]].min().min()-1, pred[[TARGET, "pred"]].max().max()+1]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlim=lim, ylim=lim, xlabel="observed root-zone (%)", ylabel="LOSO predicted (%)",
              title="(a) Leave-site-out: predicted vs observed by station")
    ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(alpha=.3)

    x = np.arange(len(STATIONS)); w = 0.38
    ax[1].bar(x-w/2, [per_site.loc[s, "r"] for s in STATIONS], w, label="within-site r", color="#4c72b0")
    ax[1].bar(x+w/2, [per_site.loc[s, "bias"] for s in STATIONS], w, label="bias (pred−obs)", color="#c44e52")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(STATIONS)
    ax[1].set(title="(b) Per-station correlation and bias", ylabel="value")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")

    imp_s = imp.sort_values()
    ax[2].barh(imp_s.index, imp_s.values, color="#55a868")
    ax[2].set(title="(c) Random Forest feature importance", xlabel="importance")
    ax[2].grid(alpha=.3, axis="x")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "leave_site_out_cv.png", dpi=130)
    plt.close(fig)


def fig_timeseries(new, cv, per_site):
    """Figure 3 -- per-site LOSO prediction vs observed, with SMIPS overlaid."""
    pred = cv["predictions"].copy()
    pred["time"] = pd.to_datetime(pred["time"])
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    for stn, axx in zip(STATIONS, axes.ravel()):
        d = new[new.station == stn].sort_values("time")
        p = pred[pred.station == stn].sort_values("time")
        axx.plot(d["time"], d[TARGET], "-o", ms=3, color="k", label="observed")
        axx.plot(p["time"], p["pred"], "-", color=COLORS[stn], lw=2, label="LOSO prediction")
        ax2 = axx.twinx()
        ax2.plot(d["time"], d[SMIPS_COL], "--", color="grey", lw=1.2, label="SMIPS (mm)")
        ax2.set_ylabel("SMIPS (mm)", color="grey", fontsize=8)
        axx.set_title(f"{stn}  (within-site r={per_site.loc[stn,'r']:.2f}, "
                      f"bias={per_site.loc[stn,'bias']:+.1f}%)")
        axx.set_ylabel("root-zone (%)"); axx.grid(alpha=.3)
        axx.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        if stn == "K6":
            axx.legend(fontsize=8, loc="upper left")
    fig.suptitle("Leave-site-out prediction vs observed by station "
                 "(right axis: SMIPS)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_site_timeseries.png", dpi=130)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sub, daily = _station_inputs()
    new = load_table(sub, daily)
    old = old_smips(sub, daily)

    cv = leave_site_out_cv(new)
    per_site = cv["per_site"].set_index("station")
    imp = feature_importance(fit(new))

    fig_data(new, old)
    fig_model(new, cv, per_site, imp)
    fig_timeseries(new, cv, per_site)
    print(f"wrote 3 figures to {FIG_DIR.relative_to(REPO)}/")
    print("pooled LOSO:", {k: round(v, 3) if isinstance(v, float) else v
                           for k, v in cv["pooled"].items()})


if __name__ == "__main__":
    main()
