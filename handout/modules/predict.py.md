# `predict.py`: the clone-and-run inference tool

<!-- NAV -->
[← Downscaling to 30 m](downscale.py.md) · [Index](../README.md)
<!-- /NAV -->

Source: [`../../emt/predict.py`](../../emt/predict.py)

The shipped entry point that turns the trained model into a 30 m map for **any**
Australian area and day, without retraining. Where [`downscale.py`](downscale.py.md)
is the model-agnostic engine (you hand it a fitted model + assembled
`extra_layers`), `predict.py` is the batteries-included wrapper: it loads the
pre-trained model committed to the repo (`data/models/model6.joblib`), fetches
**every** covariate the model needs for the AOI/day, and predicts per pixel.

| Interface | Call |
|---|---|
| Python | `predict(bbox, day, model=None, model_name="model6") -> xr.Dataset` |
| CLI | `python -m emt.predict --bbox W S E N --date YYYY-MM-DD -o map.tif` |

## What it does

For a bounding box (EPSG:4326) and a day it assembles the full feature stack — in
the model's exact `FEATURES` order — and predicts:

1. **30 m terrain** ([`covariates.py`](covariates.py.md)) — the target grid.
2. **SMIPS lookback** ([`smips.smips_lookback_day`](../../emt/smips.py)) — the
   past 7/30/365-day means + anomaly, as of the day.
3. **SLGA soil** ([`slga.soil_covariates`](slga.py.md)) — the static soil stack.
4. **SILO antecedent** ([`antecedent.antecedent_grid`](../../emt/antecedent.py)) —
   the day's gridded trailing-window weather.

Each raster is reprojected onto the 30 m grid, the features are stacked in
`FEATURES` order, and the model predicts every pixel with complete covariates.
Every window is strictly **backward-looking** (the same leak-free features the
model was trained on — see
[Evaluation correction](../README.md#evaluation-correction)). In-situ
observations are never an input.

**Returns** an `xr.Dataset` on the 30 m grid with `sm_pred` (root-zone soil
moisture, %); the CLI writes it as a single-band GeoTIFF.

## Python

```python
from emt.predict import predict
ds = predict(bbox=(147.30, -35.52, 147.62, -35.10), day="2008-07-31")
ds["sm_pred"]                          # xr.DataArray, 30 m soil moisture (%)
ds["sm_pred"].rio.to_raster("soil_moisture_30m.tif")
```

Pass your own fitted estimator via `model=` to bypass the shipped one (e.g. a
leave-region-out model); otherwise it loads `model_name` from `data/models/`.

## CLI

```bash
PYTHONPATH=. python -m emt.predict \
    --bbox 147.30 -35.52 147.62 -35.10 \
    --date 2008-07-31
# writes outputs/soil_moisture_2008-07-31.tif
```

`--bbox` is `W S E N` in EPSG:4326; `--model` defaults to `model6`. Output goes to
`outputs/soil_moisture_<date>.tif` by default (repo-local `outputs/`, gitignored,
`EMT_OUTPUTS_DIR`-overridable); pass `-o PATH` to choose your own. A first run over
a new area fetches ~a year of SMIPS and SILO (a few minutes), cached under the AOI
stub for later days.

## Requirements

Network access, plus two free accounts in `~/.config/PaddockTS.json` (see
[Use the model](../README.md#use-the-model)): a **TERN API key** (SLGA soil) and a
**SILO email** (antecedent weather). SMIPS and the Copernicus-DEM terrain need
nothing — terrain is read anonymously from its public AWS bucket
(`AWS_NO_SIGN_REQUEST`, set by the module), so no AWS credentials are required.

## Caveat: Murrumbidgee-trained, unvalidated elsewhere

The shipped model is trained and validated only in the Murrumbidgee catchment. A
run anywhere else still produces a field, but with an uncorrected per-site level
bias and no out-of-region validation (see the
[skill caveats](../README.md#the-model)). Off-catchment output is **indicative,
not calibrated** — the tool prints this warning on every run.

---
<!-- NAV -->
[← Downscaling to 30 m](downscale.py.md) · [Index](../README.md)
<!-- /NAV -->
