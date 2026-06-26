"""Stage 6 -- apply the fitted model to produce a 30 m soil-moisture field.

For a focus AOI and a single day:
  * fetch the coarse SMIPS field (~1 km) and the 30 m terrain covariate stack,
  * put SMIPS onto the 30 m grid (nearest, so each fine pixel inherits its
    overlying ~1 km value -- the field the model *sharpens*),
  * assemble the model's features per pixel and predict root-zone soil moisture.

The result is a 30 m map of predicted root-zone moisture (%) carrying the
terrain structure SMIPS cannot resolve. See ``handout/`` for the metrics and the
documented future work (a coarse-reference mass-conservation step, and an SLGA
soil covariate to cut the residual per-station bias).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.enums import Resampling

from PaddockTS.query import Query

from emt.smips import smips_day
from emt.covariates import terrain_covariates, TERRAIN_VARS
from emt.model import FEATURES, SMIPS_COL


def _doy_features(day: date) -> tuple[float, float]:
    doy = pd.Timestamp(day).dayofyear
    return float(np.sin(2 * np.pi * doy / 365.25)), float(np.cos(2 * np.pi * doy / 365.25))


def downscale(model, query: Query, day: date | str,
              smips_var: str = "totalbucket") -> xr.Dataset:
    """Downscale SMIPS to the 30 m terrain grid for one ``day`` over ``query``.

    Returns a Dataset on the terrain UTM grid with:
      ``sm_pred``       -- 30 m predicted root-zone soil moisture (%),
      ``smips_native``  -- the coarse SMIPS field resampled onto the grid (mm).
    Pixels outside the DEM's valid footprint are NaN.
    """
    if isinstance(day, str):
        day = date.fromisoformat(day)

    terr = terrain_covariates(query)                       # 30 m UTM grid
    grid = terr["elevation"]

    # Coarse SMIPS for the day, put on the 30 m grid (blocky, nearest).
    smips = smips_day(day, tuple(query.bbox), var=smips_var).rio.write_crs(4326)
    smips_on = smips.rio.reproject_match(grid, resampling=Resampling.nearest)

    doy_sin, doy_cos = _doy_features(day)
    layers = {
        SMIPS_COL: smips_on.values,
        **{v: terr[v].values for v in TERRAIN_VARS},
        "doy_sin": np.full(grid.shape, doy_sin),
        "doy_cos": np.full(grid.shape, doy_cos),
    }
    # Flatten in the model's exact feature order; predict only finite pixels.
    cols = [layers[f].ravel() for f in FEATURES]
    X = np.column_stack(cols)
    valid = np.isfinite(X).all(axis=1)

    pred = np.full(X.shape[0], np.nan, dtype="float32")
    if valid.any():
        # Predict with a named frame so feature order matches the fitted model.
        pred[valid] = model.predict(pd.DataFrame(X[valid], columns=FEATURES)).astype("float32")
    pred = pred.reshape(grid.shape)

    ds = xr.Dataset(
        {"sm_pred": (grid.dims, pred),
         "smips_native": (grid.dims, np.where(valid.reshape(grid.shape),
                                               smips_on.values, np.nan))},
        coords=grid.coords,
    )
    return ds.rio.write_crs(grid.rio.crs)
