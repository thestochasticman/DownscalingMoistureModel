"""Stage 6 with model6: leave-Yanco-out downscale + validation at 12 stations.

Trains model6 (and model4 for reference) on Kyeamba + Adelong + M-sites, with
Yanco withheld entirely, downscales the Yanco AOI for one day using GRIDDED
antecedent meteorology, and validates against the 12 held-out Yanco stations
(in-situ used only as the scoring reference, never as an input).

NOTE: deliberately not referenced from the handout README -- single-date
downscale NSE is unstable and the leave-site-out / leave-region-out numbers are
the reported skill. Kept for a visual model6-vs-model4 transfer check.

Run from repo root::  PYTHONPATH=. python handout/plot_downscale_model6.py
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
from pyproj import Transformer

from emt.queries import query_for_focus_area
from emt.downscale import downscale
from emt.covariates import sample_points
from emt.smips import smips_lookback_day
from emt.slga import soil_covariates, SOIL_VARS
from emt.antecedent import antecedent_grid, antecedent_day_layers
from emt.model6 import model as m6
from emt.model4 import model as m4

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "downscale_yanco_model6.png"
TABLE = REPO / "data" / "train_catchment_plus_m_2006_2010.csv"
DAY = date(2008, 7, 31)
TARGET = m6.TARGET
t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# --- train model6 and model4, leave-Yanco-out ---
tab = pd.read_csv(TABLE)
train = tab[tab.site != "YANCO"]
model6 = m6.fit(train)
tr4 = m4.ensure_features(train).dropna(subset=m4.FEATURES + [TARGET])
model4 = m4.build_estimator().fit(tr4[m4.FEATURES], tr4[TARGET])
stamp(f"trained model6 & model4 on {train.station.nunique()} stns (Yanco held out)")

# --- extra AOI rasters (leak-free lookback + soil) + gridded antecedent ---
q = query_for_focus_area("yanco", DAY, DAY)
smips_l = {k: v for k, v in smips_lookback_day(q, DAY).items()
           if k != "smips_totalbucket"}
soil = soil_covariates(q)
static = {**smips_l, **{v: soil[v] for v in SOIL_VARS}}
ante = antecedent_day_layers(antecedent_grid(q, DAY, DAY), DAY)
stamp("AOI SMIPS lookback + soil + gridded antecedent")

# --- downscale both for the day ---
ds4 = downscale(model4, q, DAY, m4.FEATURES, extra_layers=static)
ds6 = downscale(model6, q, DAY, m6.FEATURES, extra_layers={**static, **ante})
stamp(f"downscaled model4 & model6 ({ds6['sm_pred'].size/1e6:.1f}M px)")

# --- validate at the 12 held-out Yanco stations ---
obs = tab[(tab.site == "YANCO") & (tab.time == DAY.isoformat())].copy()
def val(ds):
    p = [float(sample_points(ds["sm_pred"], r.lon, r.lat).values)
         for r in obs.itertuples(index=False)]
    o = obs.assign(pred=p).dropna(subset=["pred"])
    return o, m6.metrics(o[TARGET], o["pred"])
o4, M4 = val(ds4); o6, M6 = val(ds6)
stamp(f"model4: bias={M4['bias']:+.2f} ubRMSE={M4['ubrmse']:.2f} r={M4['r']:.2f}")
stamp(f"model6: bias={M6['bias']:+.2f} ubRMSE={M6['ubrmse']:.2f} r={M6['r']:.2f}")

# --- figure ---
crs = ds6.rio.crs
tr = Transformer.from_crs(4326, crs, always_xy=True)
sx, sy = tr.transform(o6["lon"].values, o6["lat"].values)
ext = [float(ds6.x.min()), float(ds6.x.max()), float(ds6.y.min()), float(ds6.y.max())]
vmin, vmax = np.nanpercentile(ds6["sm_pred"].values, [2, 98])
ny, nx = ds6["sm_pred"].shape

fig, ax = plt.subplots(2, 2, figsize=(15, 12))
im0 = ax[0,0].imshow(ds6["smips_native"].values, extent=ext, origin="upper", cmap="YlGnBu")
ax[0,0].set_title("(a) Coarse SMIPS input (~1 km, mm)")
fig.colorbar(im0, ax=ax[0,0], shrink=.8, label="SMIPS TotalBucket (mm)")

im1 = ax[0,1].imshow(ds6["sm_pred"].values, extent=ext, origin="upper",
                     cmap="YlGnBu", vmin=vmin, vmax=vmax)
ax[0,1].scatter(sx, sy, c=o6[TARGET], cmap="YlGnBu", vmin=vmin, vmax=vmax,
                s=90, edgecolor="red", linewidth=1.5, zorder=3)
ax[0,1].set_title("(b) model6 downscaled 30 m (%); dots = held-out stations")
fig.colorbar(im1, ax=ax[0,1], shrink=.8, label="root-zone SM (%)")

zy, zx = ny // 3, nx // 3
sub = ds6["sm_pred"].values[zy:zy+ny//3, zx:zx+nx//3]
sub_ext = [float(ds6.x[zx]), float(ds6.x[zx+nx//3-1]),
           float(ds6.y[zy+ny//3-1]), float(ds6.y[zy])]
im2 = ax[1,0].imshow(sub, extent=sub_ext, origin="upper", cmap="YlGnBu")
ax[1,0].set_title("(c) Detail: 30 m terrain structure")
fig.colorbar(im2, ax=ax[1,0], shrink=.8, label="root-zone SM (%)")

ax[1,1].scatter(o6[TARGET], o6["pred"], s=80, color="#9467bd", edgecolor="k")
for r in o6.itertuples(index=False):
    ax[1,1].annotate(r.station, (getattr(r, TARGET), r.pred), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
lim = [min(o6[TARGET].min(), o6["pred"].min())-2, max(o6[TARGET].max(), o6["pred"].max())+2]
ax[1,1].plot(lim, lim, "k--", lw=1)
ax[1,1].set(xlim=lim, ylim=lim, xlabel="OzNet observed (%)", ylabel="downscaled @ station (%)",
            title="(d) Held-out validation at the Yanco stations")
ax[1,1].text(.03, .97,
             f"model6:  bias={M6['bias']:+.2f}%  ubRMSE={M6['ubrmse']:.2f}%  r={M6['r']:.2f}\n"
             f"model4:  bias={M4['bias']:+.2f}%  ubRMSE={M4['ubrmse']:.2f}%  r={M4['r']:.2f}\n"
             f"(single-date NSE unstable; LOSO is the reported skill)",
             transform=ax[1,1].transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="w", alpha=.8))
ax[1,1].grid(alpha=.3)

fig.suptitle(f"Stage 6 (model6, gridded antecedent): SMIPS to 30 m over Yanco, {DAY} "
             f"(Yanco held out)", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG, dpi=130)
stamp(f"wrote {FIG.relative_to(REPO)}")
