"""model5 -- model4 with spatially smoothed SLGA soil (cleaner downscaled maps).

Identical estimator and feature list to :mod:`emt.model4`; the only change is
that the soil covariates are Gaussian-blurred (:func:`emt.slga.smooth_soil`)
before use, at BOTH training-point sampling and map inference, so the model is
self-consistent.

Motivation: SLGA is effectively a mosaic of map units meeting at hard edges. The
model4 30 m field uses it verbatim and inherits blocky soil boundaries that mute
the fine terrain texture (see the model4 downscaling demonstration; one station,
Y3, was thrown low by a boundary). Smoothing the soil rasters softens those seams
and does clean the map (Yanco demo: ubRMSE 4.9 -> 3.05 %, r 0.30 -> 0.39).

**But this is a tradeoff, not an improvement, and model5 is NOT recommended.**
The sharp per-station soil detail is the #2 predictor and carries real
between-station discrimination; smoothing removes exactly that. A controlled
sigma sweep shows leave-site-out skill degrades monotonically with the blur, with
no sweet spot: NSE 0.354 (sigma=0, == model4) -> 0.286 (1) -> 0.126 (2) ->
0.058 (3). The same soil detail that makes the map blocky is what discriminates
stations, so the map artifacts and the tabular skill cannot be separated by
smoothing. model4 remains the recommended model. model5 exists only to document
this texture-vs-skill tension and, at most, to produce a visually cleaner map
where lower per-station skill is acceptable.

``SOIL_SIGMA`` is in pixels of the soil grid (SLGA native ~90 m); 2 px ~ 180 m.
"""
from __future__ import annotations

import pandas as pd

from emt.features import SMIPS_COL, CLIM_VARS, add_smips_climatology, query_for_station
from emt.covariates import sample_points
from emt.slga import soil_covariates, smooth_soil, SOIL_VARS
# Reuse model4's estimator, feature list, importance and metrics verbatim.
from emt.model4.model import (FEATURES, build_estimator, feature_importance,  # noqa: F401
                              TARGET)
from emt.evaluation import metrics, leave_site_out_cv as _cv  # noqa: F401

SOIL_SIGMA = 2.0


def _add_smoothed_soil(table: pd.DataFrame) -> pd.DataFrame:
    """Attach soil covariates sampled from the *smoothed* per-station rasters."""
    from emt.insitu.coordinates import COORDS_CACHE
    coords = pd.read_csv(COORDS_CACHE).set_index("station")
    times = pd.to_datetime(table["time"])
    start, end = times.min().date(), times.max().date()
    rows = []
    for stn in table["station"].unique():
        lat, lon = float(coords.loc[stn, "lat"]), float(coords.loc[stn, "lon"])
        q = query_for_station(stn, lat, lon, start, end)
        soil = smooth_soil(soil_covariates(q), SOIL_SIGMA)
        pt = sample_points(soil, lon, lat)
        rows.append({"station": stn, **{v: float(pt[v].values) for v in SOIL_VARS}})
    return table.merge(pd.DataFrame(rows), on="station", how="left")


def ensure_features(table: pd.DataFrame) -> pd.DataFrame:
    """Derive model5's features: SMIPS climatology + smoothed soil."""
    if any(v not in table.columns for v in CLIM_VARS):
        table = add_smips_climatology(table)
    if any(v not in table.columns for v in SOIL_VARS):
        table = _add_smoothed_soil(table)
    return table


def fit(table: pd.DataFrame, estimator=None):
    est = estimator if estimator is not None else build_estimator()
    sub = ensure_features(table).dropna(subset=FEATURES + [TARGET])
    est.fit(sub[FEATURES], sub[TARGET])
    return est


def leave_site_out_cv(table: pd.DataFrame, group_col: str = "station") -> dict:
    return _cv(ensure_features(table), FEATURES, build_estimator, group_col=group_col)


if __name__ == "__main__":
    import sys
    table = (pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith(".parquet")
             else pd.read_csv(sys.argv[1]))
    cv = leave_site_out_cv(table)
    print("pooled:", {k: round(v, 3) if isinstance(v, float) else v for k, v in cv["pooled"].items()})
    per = cv["per_site"]
    print(f"per-station NSE>0: {(per['nse'] > 0).sum()}/{len(per)} "
          f"(median {per['nse'].median():.2f})")
