"""Stage 6 with model5: model4 + spatially smoothed SLGA soil, over Yanco.

Same leave-Yanco-out protocol as ``plot_downscale_model4.py``; the only change is
that the soil rasters are Gaussian-blurred (``emt.slga.smooth_soil``) at training
and inference. Produces a two-panel comparison of the model4 vs model5 30 m field
plus the station validation for both.

Run from repo root::  PYTHONPATH=. python handout/plot_downscale_model5.py
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
from emt.slga import soil_covariates, smooth_soil, SOIL_VARS
from emt.model5.model import (build_estimator, ensure_features, fit, metrics,
                              FEATURES, TARGET, SOIL_SIGMA)

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "downscale_yanco_model5.png"
TABLE = REPO / "data" / "train_catchment_2006_2010.csv"
DAY = date(2008, 7, 31)
CLIM_PERIOD = (date(2006, 1, 1), date(2010, 12, 31))
t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# --- train model4 and model5 leave-Yanco-out ---
tab = pd.read_csv(TABLE)
train = tab[tab.site != "YANCO"]
# model5 (smoothed soil at train + inference)
m5 = fit(train)
stamp(f"trained model5 on {sorted(train.site.unique())} "
      f"({train.station.nunique()} stns)")
# model4 (raw soil) for the side-by-side
from emt.model4.model import (build_estimator as m4_est, ensure_features as m4_feat,
                              FEATURES as M4_FEATURES)
tr4 = m4_feat(train).dropna(subset=M4_FEATURES + [TARGET])
m4 = m4_est(); m4.fit(tr4[M4_FEATURES], tr4[TARGET])
stamp("trained model4 (raw soil) reference")

# --- static AOI rasters ---
q_clim = query_for_focus_area("yanco", *CLIM_PERIOD)
clim = smips_climatology(q_clim, step_days=5)
soil_raw = soil_covariates(q_clim)
soil_sm = smooth_soil(soil_raw, SOIL_SIGMA)
stamp(f"AOI climatology + soil (raw & smoothed, sigma={SOIL_SIGMA}px)")
clim_layers = {"smips_mean_px": clim["smips_mean_px"], "smips_std_px": clim["smips_std_px"]}
extra4 = {**clim_layers, **{v: soil_raw[v] for v in SOIL_VARS}}
extra5 = {**clim_layers, **{v: soil_sm[v] for v in SOIL_VARS}}

# --- downscale both ---
q = query_for_focus_area("yanco", DAY, DAY)
ds4 = downscale(m4, q, DAY, M4_FEATURES, extra_layers=extra4)
ds5 = downscale(m5, q, DAY, FEATURES, extra_layers=extra5)
stamp(f"downscaled model4 & model5 ({ds5['sm_pred'].size/1e6:.1f}M px)")

# --- validate both at the 12 held-out Yanco stations ---
obs = tab[(tab.site == "YANCO") & (tab.time == DAY.isoformat())].copy()
def val(ds):
    p = [float(sample_points(ds["sm_pred"], r.lon, r.lat).values)
         for r in obs.itertuples(index=False)]
    o = obs.assign(pred=p).dropna(subset=["pred"])
    return o, metrics(o[TARGET], o["pred"])
o4, M4 = val(ds4); o5, M5 = val(ds5)
stamp(f"model4: bias={M4['bias']:+.2f} ubRMSE={M4['ubrmse']:.2f} r={M4['r']:.2f}")
stamp(f"model5: bias={M5['bias']:+.2f} ubRMSE={M5['ubrmse']:.2f} r={M5['r']:.2f}")

# --- figure: model4 vs model5 field + detail ---
crs = ds5.rio.crs
ext = [float(ds5.x.min()), float(ds5.x.max()), float(ds5.y.min()), float(ds5.y.max())]
vmin, vmax = np.nanpercentile(ds5["sm_pred"].values, [2, 98])
ny, nx = ds5["sm_pred"].shape
zy, zx = ny // 3, nx // 3
def detail(ds):
    return ds["sm_pred"].values[zy:zy+ny//3, zx:zx+nx//3]
dext = [float(ds5.x[zx]), float(ds5.x[zx+nx//3-1]),
        float(ds5.y[zy+ny//3-1]), float(ds5.y[zy])]

fig, ax = plt.subplots(2, 2, figsize=(15, 12))
for col, (ds, lab, M) in enumerate([(ds4, "model4 (raw soil)", M4),
                                     (ds5, f"model5 (soil σ={SOIL_SIGMA}px)", M5)]):
    im = ax[0,col].imshow(ds["sm_pred"].values, extent=ext, origin="upper",
                          cmap="YlGnBu", vmin=vmin, vmax=vmax)
    ax[0,col].set_title(f"(top) {lab}: 30 m field")
    fig.colorbar(im, ax=ax[0,col], shrink=.8, label="root-zone SM (%)")
    im2 = ax[1,col].imshow(detail(ds), extent=dext, origin="upper",
                           cmap="YlGnBu", vmin=vmin, vmax=vmax)
    ax[1,col].set_title(f"(detail) {lab}\nbias={M['bias']:+.2f}%  "
                        f"ubRMSE={M['ubrmse']:.2f}%  r={M['r']:.2f}  NSE={M['nse']:.1f}")
    fig.colorbar(im2, ax=ax[1,col], shrink=.8, label="root-zone SM (%)")

fig.suptitle(f"Soil smoothing (model5 vs model4): SMIPS to 30 m over Yanco, {DAY} "
             f"(Yanco held out)", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG, dpi=130)
stamp(f"wrote {FIG.relative_to(REPO)}")
