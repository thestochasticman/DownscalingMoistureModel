# Soil-moisture point validation

This folder validates `model6` predictions against the point soil-moisture CSV:

`/Volumes/Dmitry_work/borevitz_projects/Data/soilmoisture_points_coordinates.csv`

It does three things:

1. reads the point observations and dates from the CSV;
2. generates one downscaled GeoTIFF per unique date using `emt.predict`;
3. samples each GeoTIFF at the point coordinates and writes handout-style metrics.

Run from the repository root with the `paddockts` environment active:

```bash
conda activate paddockts

python soilmoisture_points_validation/run_validation.py \
  --input-csv /Volumes/Dmitry_work/borevitz_projects/Data/soilmoisture_points_coordinates.csv
```

Useful preflight, with no network downloads:

```bash
python soilmoisture_points_validation/run_validation.py --dry-run
```

If the TIFFs already exist and you only want to recompute point samples/metrics:

```bash
python soilmoisture_points_validation/run_validation.py --sample-only
```

Outputs are written under `soilmoisture_points_validation/outputs/`:

- `tifs/soil_moisture_<date>.tif` — one model6 prediction raster per date.
- `predictions.csv` — one row per point/date observation with observed and predicted values.
- `metrics_pooled.json` — pooled RMSE, ubRMSE, bias, Pearson `r`, NSE/R² and `n`.
- `metrics_per_point.csv` — the same metrics grouped by `Point_number`.
- `metrics_per_date.csv` — the same metrics grouped by sampling date.
- `excluded_rows.csv` — rows skipped because they lack a date, observation, or usable coordinates.
- `report.md` — compact summary in the same style as the handout.

Notes:

- The CSV columns are named `x_3577` and `y_3577`, but their values are longitude
  and latitude in EPSG:4326. The script treats them as lon/lat.
- Rows without coordinates are excluded from raster sampling.
- The handout model target is OzNet root-zone soil moisture, 0–90 cm. This CSV
  appears to be a shallower point measurement, so these metrics compare model6
  root-zone predictions with the available point values. Treat them as an
  external terrain-transfer diagnostic, not a pure root-zone validation unless
  the measurement depth is reconciled.
- If loading `data/models/model6.joblib` fails with `ModuleNotFoundError: No module
  named '_loss'`, pin scikit-learn to the version used by the shipped model:

```bash
conda activate paddockts
conda install -c conda-forge "scikit-learn=1.8.0"
```
