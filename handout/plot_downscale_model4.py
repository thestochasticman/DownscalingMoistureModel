"""Stage 6 with model4: downscale SMIPS to 30 m over Yanco and validate.

Same out-of-sample protocol as ``plot_downscale.py`` (train on Kyeamba +
Adelong, Yanco withheld entirely), but with model4 -- which needs two static
AOI rasters at inference, passed via ``extra_layers``:

* SMIPS pixel climatology over the AOI (``emt.smips.smips_climatology``,
  thinned multi-year fetch, cached),
* the SLGA soil stack (``emt.slga.soil_covariates``).

Run from repo root::  PYTHONPATH=. python handout/plot_downscale_model4.py
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
from emt.smips import smips_climatology
from emt.slga import soil_covariates, SOIL_VARS
from emt.model4.model import (build_estimator, ensure_features, metrics,
                              FEATURES, TARGET)

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "downscale_yanco_model4.png"
TABLE = REPO / "data" / "train_catchment_2006_2010.csv"
DAY = date(2008, 7, 31)          # all 12 Yanco stations report this day
CLIM_PERIOD = (date(2006, 1, 1), date(2010, 12, 31))   # matches training period
# model1 reference values on this exact protocol (see plot_downscale.py run)
M1_REF = dict(rmse=11.5, ubrmse=2.4, bias=11.3, r=0.41, nse=-20.0)
t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# --- train leave-Yanco-out (Kyeamba + Adelong) ---
tab = pd.read_csv(TABLE)
train = ensure_features(tab[tab.site != "YANCO"]).dropna(subset=FEATURES + [TARGET])
model = build_estimator()
model.fit(train[FEATURES], train[TARGET])
stamp(f"trained model4 on {train.site.unique().tolist()} "
      f"({train.station.nunique()} stns, {len(train):,} rows)")

# --- static AOI rasters for inference ---
q_clim = query_for_focus_area("yanco", *CLIM_PERIOD)
clim = smips_climatology(q_clim, step_days=5)
stamp(f"SMIPS climatology over Yanco AOI ({int(clim.attrs.get('n_samples', 0))} sample days)")
soil = soil_covariates(q_clim)
stamp("SLGA soil stack over Yanco AOI")
extra = {"smips_mean_px": clim["smips_mean_px"],
         "smips_std_px": clim["smips_std_px"],
         **{v: soil[v] for v in SOIL_VARS}}

# --- downscale Yanco for the day ---
q = query_for_focus_area("yanco", DAY, DAY)
stamp(f"downscaling Yanco AOI {q.bbox} for {DAY} ...")
ds = downscale(model, q, DAY, FEATURES, extra_layers=extra)
ny, nx = ds["sm_pred"].shape
stamp(f"30 m grid: {ny} x {nx} = {ny*nx/1e6:.1f}M pixels")

# --- validate at the 12 Yanco stations (held out) ---
obs = tab[(tab.site == "YANCO") & (tab.time == DAY.isoformat())].copy()
pred_pts = [float(sample_points(ds["sm_pred"], r.lon, r.lat).values)
            for r in obs.itertuples(index=False)]
obs["pred"] = pred_pts
obs = obs.dropna(subset=["pred"])
M = metrics(obs[TARGET], obs["pred"])
stamp("validation @ Yanco stations: " +
      ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in M.items()))

# --- figures ---
crs = ds.rio.crs
tr = Transformer.from_crs(4326, crs, always_xy=True)
sx, sy = tr.transform(obs["lon"].values, obs["lat"].values)
ext = [float(ds.x.min()), float(ds.x.max()), float(ds.y.min()), float(ds.y.max())]
vmin, vmax = np.nanpercentile(ds["sm_pred"].values, [2, 98])

fig, ax = plt.subplots(2, 2, figsize=(15, 12))

# (a) coarse SMIPS input (mm), blocky ~1 km
im0 = ax[0,0].imshow(ds["smips_native"].values, extent=ext, origin="upper", cmap="YlGnBu")
ax[0,0].set_title("(a) Coarse SMIPS input (~1 km, mm)")
fig.colorbar(im0, ax=ax[0,0], shrink=.8, label="SMIPS TotalBucket (mm)")

# (b) 30 m downscaled product (%), with held-out stations coloured by observation
im1 = ax[0,1].imshow(ds["sm_pred"].values, extent=ext, origin="upper",
                     cmap="YlGnBu", vmin=vmin, vmax=vmax)
ax[0,1].scatter(sx, sy, c=obs[TARGET], cmap="YlGnBu", vmin=vmin, vmax=vmax,
                s=90, edgecolor="red", linewidth=1.5, zorder=3)
ax[0,1].set_title("(b) model4 downscaled 30 m product (%); dots = held-out stations")
fig.colorbar(im1, ax=ax[0,1], shrink=.8, label="root-zone SM (%)")

# (c) the terrain detail the downscaling adds (zoom on the field)
zy, zx = ny // 3, nx // 3
sub = ds["sm_pred"].values[zy:zy+ny//3, zx:zx+nx//3]
sub_ext = [float(ds.x[zx]), float(ds.x[zx+nx//3-1]),
           float(ds.y[zy+ny//3-1]), float(ds.y[zy])]
im2 = ax[1,0].imshow(sub, extent=sub_ext, origin="upper", cmap="YlGnBu")
ax[1,0].set_title("(c) Detail: 30 m terrain structure")
fig.colorbar(im2, ax=ax[1,0], shrink=.8, label="root-zone SM (%)")

# (d) validation scatter, with the model1 reference on the same protocol
ax[1,1].scatter(obs[TARGET], obs["pred"], s=80, color="#2ca02c", edgecolor="k")
for r in obs.itertuples(index=False):
    ax[1,1].annotate(r.station, (getattr(r, TARGET), r.pred), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
lim = [min(obs[TARGET].min(), obs["pred"].min())-2,
       max(obs[TARGET].max(), obs["pred"].max())+2]
ax[1,1].plot(lim, lim, "k--", lw=1)
ax[1,1].set(xlim=lim, ylim=lim, xlabel="OzNet observed (%)",
            ylabel="downscaled @ station (%)",
            title="(d) Held-out validation at the Yanco stations")
ax[1,1].text(.03, .97,
             f"model4:\nRMSE={M['rmse']:.2f}%  ubRMSE={M['ubrmse']:.2f}%\n"
             f"bias={M['bias']:+.2f}%  r={M['r']:.2f}  NSE={M['nse']:.1f}  n={M['n']}\n"
             f"model1 (same protocol):\nRMSE={M1_REF['rmse']:.1f}%  "
             f"ubRMSE={M1_REF['ubrmse']:.1f}%\nbias={M1_REF['bias']:+.1f}%  "
             f"r={M1_REF['r']:.2f}  NSE={M1_REF['nse']:.0f}\n"
             f"(single-date NSE is unstable; see README)",
             transform=ax[1,1].transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="w", alpha=.8))
ax[1,1].grid(alpha=.3)

fig.suptitle(f"Stage 6 (model4): SMIPS to 30 m over Yanco, {DAY} "
             f"(trained on Kyeamba and Adelong; Yanco held out)", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG, dpi=130)
stamp(f"wrote {FIG.relative_to(REPO)}")
