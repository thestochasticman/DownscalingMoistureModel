"""Gallery of generated 30 m soil-moisture fields across a seasonal cycle.

Trains model4 on the full 36-station table (the production model), then applies
it over one focus AOI on a series of dates through 2008, tiling the resulting
30 m fields into a grid with a shared colour scale so the seasonal wet -> dry
evolution is directly comparable.

Run from repo root::  PYTHONPATH=. python handout/plot_downscale_gallery.py
"""
from __future__ import annotations
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emt.queries import query_for_focus_area
from emt.downscale import downscale
from emt.smips import smips_climatology
from emt.slga import soil_covariates, SOIL_VARS
from emt.model4.model import build_estimator, ensure_features, FEATURES, TARGET

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "downscale_gallery.png"
FIG_SMIPS = REPO / "handout" / "figures" / "downscale_gallery_smips.png"
FIG_PAIR = REPO / "handout" / "figures" / "downscale_gallery_paired.png"
TABLE = REPO / "data" / "train_catchment_plus_m_2006_2010.csv"
AOI = "kyeamba"                       # terrain relief -> visible drainage structure
CLIM_PERIOD = (date(2006, 1, 1), date(2010, 12, 31))
DATES = [date(2008, m, d) for m, d in
         [(1, 15), (2, 25), (4, 5), (5, 15), (6, 25), (8, 5), (9, 15), (10, 25), (12, 5)]]
t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# --- production model: model4 on all 36 stations ---
tab = ensure_features(pd.read_csv(TABLE)).dropna(subset=FEATURES + [TARGET])
model = build_estimator().fit(tab[FEATURES], tab[TARGET])
stamp(f"trained model4 on {tab.station.nunique()} stations, {len(tab):,} rows")

# --- static AOI rasters (climatology + soil), fetched once ---
q_clim = query_for_focus_area(AOI, *CLIM_PERIOD)
clim = smips_climatology(q_clim, step_days=5)
soil = soil_covariates(q_clim)
extra = {"smips_mean_px": clim["smips_mean_px"], "smips_std_px": clim["smips_std_px"],
         **{v: soil[v] for v in SOIL_VARS}}
stamp(f"AOI climatology + soil for {AOI}")

# --- downscale each date (keep both the 30 m field and the coarse SMIPS input) ---
pred_fields, smips_fields = [], []
for d in DATES:
    ds = downscale(model, query_for_focus_area(AOI, d, d), d, FEATURES, extra_layers=extra)
    pred_fields.append((d, ds["sm_pred"]))
    smips_fields.append((d, ds["smips_native"]))
    stamp(f"downscaled {d}  (mean {float(ds['sm_pred'].mean()):.1f}%)")

g0 = pred_fields[0][1]
ext = [float(g0.x.min()), float(g0.x.max()), float(g0.y.min()), float(g0.y.max())]


def gallery(fields, cmap, label, title, out):
    """Tile a list of (date, DataArray) on a shared colour scale."""
    allvals = np.concatenate([f.values[np.isfinite(f.values)] for _, f in fields])
    vmin, vmax = np.percentile(allvals, [2, 98])
    ncol = 3
    nrow = int(np.ceil(len(fields) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.3, nrow * 4.0))
    axes = axes.ravel()
    im = None
    for ax, (d, f) in zip(axes, fields):
        im = ax.imshow(f.values, extent=ext, origin="upper", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_title(d.strftime("%d %b %Y"), fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(fields):]:
        ax.set_visible(False)
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label=label)
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)
    stamp(f"wrote {out.relative_to(REPO)}")


gallery(smips_fields, "YlGnBu", "SMIPS TotalBucket (mm)",
        f"Coarse SMIPS input (~1 km) over {AOI.title()}, 2008 (shared colour scale)",
        FIG_SMIPS)
gallery(pred_fields, "YlGnBu", "root-zone soil moisture (%)",
        f"Generated 30 m soil moisture over {AOI.title()}, 2008 "
        f"(model4; shared colour scale)", FIG)


def paired(smips_fields, pred_fields, out):
    """One image: coarse SMIPS (left) beside the 30 m field (right), per date."""
    def rng(fields):
        v = np.concatenate([f.values[np.isfinite(f.values)] for _, f in fields])
        return np.percentile(v, [2, 98])
    svmin, svmax = rng(smips_fields)
    pvmin, pvmax = rng(pred_fields)
    n = len(smips_fields)
    fig, axes = plt.subplots(n, 2, figsize=(6.4, n * 2.35))
    ims = [None, None]
    for i, ((d, sf), (_, pf)) in enumerate(zip(smips_fields, pred_fields)):
        ims[0] = axes[i, 0].imshow(sf.values, extent=ext, origin="upper",
                                   cmap="YlGnBu", vmin=svmin, vmax=svmax)
        ims[1] = axes[i, 1].imshow(pf.values, extent=ext, origin="upper",
                                   cmap="YlGnBu", vmin=pvmin, vmax=pvmax)
        axes[i, 0].set_ylabel(d.strftime("%d %b %Y"), fontsize=10)
        for j in (0, 1):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    axes[0, 0].set_title("Coarse SMIPS input\n(~1 km, mm)", fontsize=11)
    axes[0, 1].set_title("Generated 30 m\n(model4, %)", fontsize=11)
    fig.subplots_adjust(left=0.11, right=0.86, top=0.95, bottom=0.03,
                        wspace=0.05, hspace=0.06)
    fig.colorbar(ims[0], cax=fig.add_axes([0.885, 0.53, 0.02, 0.38]),
                 label="SMIPS (mm)")
    fig.colorbar(ims[1], cax=fig.add_axes([0.885, 0.08, 0.02, 0.38]),
                 label="soil moisture (%)")
    fig.suptitle(f"Coarse SMIPS → generated 30 m, {AOI.title()} 2008", fontsize=13, y=0.985)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    stamp(f"wrote {out.relative_to(REPO)}")


paired(smips_fields, pred_fields, FIG_PAIR)
