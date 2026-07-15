"""Apply the trained model to a user's area -- the clone-and-run entry point.

    from emt.predict import predict
    ds = predict(bbox=(147.3, -35.5, 147.6, -35.1), day="2008-07-31")  # xr.Dataset
    ds["sm_pred"]  # 30 m root-zone soil moisture (%)

or from the command line:

    python -m emt.predict --bbox 147.3 -35.5 147.6 -35.1 --date 2008-07-31 -o map.tif

Fetches every covariate the model needs for the AOI and day (30 m terrain, SMIPS
lookback, SLGA soil, SILO antecedent weather), loads the shipped trained model
(``data/models/<model>.joblib``) and predicts per 30 m pixel.

REQUIRES network access and, for the soil/climate covariates, a TERN API key and
a SILO email in ``~/.config/PaddockTS.json`` (see the README). The shipped model
is trained on the **Murrumbidgee** catchment only: a run elsewhere in Australia
produces a plausible but **unvalidated** field with a per-site level bias — treat
it as indicative, not calibrated.
"""
from __future__ import annotations

import argparse
import os
from datetime import date as _date
from pathlib import Path

# The 30 m terrain COGs live on the PUBLIC Copernicus-DEM open-data bucket
# (copernicus-dem-30m). Read it anonymously so the tool needs no AWS setup and
# never trips over a stale/expired local AWS profile. setdefault lets a user who
# genuinely needs signed access override with AWS_NO_SIGN_REQUEST=NO.
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.enums import Resampling

from PaddockTS.query import Query

from emt.covariates import terrain_covariates, TERRAIN_VARS
from emt.smips import smips_lookback_day
from emt.slga import soil_covariates, SOIL_VARS
from emt.antecedent import antecedent_grid, antecedent_day_layers
from emt.persist import load_model

WARNING = ("NOTE: the shipped model is trained on the Murrumbidgee catchment; "
           "predictions elsewhere are plausible but UNVALIDATED (per-site level "
           "bias, no out-of-region validation). Treat as indicative.")


def _stub(bbox, day) -> str:
    b = "_".join(f"{v:.3f}" for v in bbox)
    return f"predict_{b}_{day}".replace(".", "p").replace("-", "m")


def _doy(day) -> tuple[float, float]:
    d = pd.Timestamp(day).dayofyear
    return float(np.sin(2 * np.pi * d / 365.25)), float(np.cos(2 * np.pi * d / 365.25))


def predict(bbox, day, model=None, model_name: str = "model6",
            verbose: bool = True) -> xr.Dataset:
    """Downscale to a 30 m root-zone soil-moisture field over ``bbox`` for ``day``.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
        day: ``date`` or ISO string.
        model: a fitted estimator; if ``None``, loads ``model_name`` from
            ``data/models/`` (raises if not present — see README to obtain it).

    Returns an ``xr.Dataset`` on the 30 m grid with ``sm_pred`` (%). Import the
    model's feature list from its package if you need the exact order.
    """
    if isinstance(day, str):
        day = _date.fromisoformat(day)
    from emt.model6 import model as _m6            # feature list + order
    features = _m6.FEATURES

    if model is None:
        model = load_model(model_name)
        if model is None:
            raise FileNotFoundError(
                f"no trained model at data/models/{model_name}.joblib — train it "
                f"with `python -m emt.build_dataset` then fit, or use the shipped one.")
    if verbose:
        print(WARNING, flush=True)

    q = Query(bbox=list(bbox), start=day, end=day, stub=_stub(bbox, day))

    if verbose:
        print("  terrain ...", flush=True)
    terr = terrain_covariates(q)
    grid = terr["elevation"]
    if verbose:
        print("  SMIPS lookback (past 7/30/365 d) ...", flush=True)
    smips_l = smips_lookback_day(q, day)
    if verbose:
        print("  SLGA soil ...", flush=True)
    soil = soil_covariates(q)
    if verbose:
        print("  SILO antecedent ...", flush=True)
    ante = antecedent_day_layers(antecedent_grid(q, day, day), day)

    doy_sin, doy_cos = _doy(day)
    layers: dict = {}
    layers.update({v: terr[v] for v in TERRAIN_VARS})
    layers.update(smips_l)                          # smips_totalbucket, 7/30/365d, anom
    layers.update({v: soil[v] for v in SOIL_VARS})
    layers.update(ante)
    layers["doy_sin"] = doy_sin
    layers["doy_cos"] = doy_cos

    cols = []
    for f in features:
        obj = layers[f]
        if isinstance(obj, xr.DataArray):
            if obj.rio.crs is None:
                obj = obj.rio.write_crs(4326)
            cols.append(obj.rio.reproject_match(grid, resampling=Resampling.nearest)
                        .values.ravel())
        else:
            cols.append(np.full(grid.size, float(obj)))
    X = np.column_stack(cols)
    valid = np.isfinite(X).all(axis=1)
    pred = np.full(X.shape[0], np.nan, dtype="float32")
    if valid.any():
        pred[valid] = model.predict(pd.DataFrame(X[valid], columns=features)).astype("float32")
    pred = pred.reshape(grid.shape)
    if verbose:
        print(f"  predicted {int(valid.sum()):,} pixels, mean "
              f"{np.nanmean(pred):.1f}%", flush=True)
    return xr.Dataset({"sm_pred": (grid.dims, pred)}, coords=grid.coords).rio.write_crs(grid.rio.crs)


def main():
    ap = argparse.ArgumentParser(description="Downscale SMIPS to a 30 m soil-moisture map.")
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("W", "S", "E", "N"), help="lon/lat bounds (EPSG:4326)")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--model", default="model6")
    ap.add_argument("-o", "--out", default=None,
                    help="output GeoTIFF (default: outputs/soil_moisture_<date>.tif)")
    a = ap.parse_args()
    from emt.config import OUTPUTS_DIR
    out = Path(a.out) if a.out else OUTPUTS_DIR / f"soil_moisture_{a.date}.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = predict(tuple(a.bbox), a.date, model_name=a.model)
    ds["sm_pred"].rio.to_raster(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
