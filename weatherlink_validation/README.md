# WeatherLink soil-moisture validation

This folder adds Davis WeatherLink v2 ingestion to the existing model6 dense
validation workflow.

The workflow is:

1. download WeatherLink historic soil-moisture records;
2. normalize them into the generic point/date CSV used by
   `soilmoisture_points_validation/run_validation.py`;
3. generate/sampling model6 daily GeoTIFFs; and
4. run the same two-stage terrain-bias and local-data spiking analysis used for
   the dense point campaign.

## Credentials

WeatherLink v2 currently requires both:

- an API key passed as the `api-key` query parameter; and
- an API secret passed as the `X-Api-Secret` request header.

Do not commit these. The recommended local setup is:

```bash
mkdir -p .secrets
chmod 700 .secrets

cat > .secrets/weatherlink.env <<'EOF'
WEATHERLINK_API_KEY=your_api_key_here
WEATHERLINK_API_SECRET=your_api_secret_here
EOF
chmod 600 .secrets/weatherlink.env
```

The `.secrets/` folder is ignored by git.

## First: list stations and sensors

```bash
python weatherlink_validation/download_weatherlink_soil_moisture.py --list-only
```

This writes:

- `weatherlink_validation/outputs/weatherlink_sensors.csv`
- `weatherlink_validation/outputs/weatherlink_stations.json`
- `weatherlink_validation/outputs/weatherlink_sensors_raw.json`
- `weatherlink_validation/outputs/weatherlink_nodes.json`

Use this to identify the WeatherLink `station_id` and the soil-probe `lsid`
values.

## Sensor coordinates and centibar conversion

Davis #6440 soil-moisture sensors report soil-water tension as
`moist_soil_last` in centibars. Model6 predicts volumetric soil moisture in
percent, so centibars must be converted before RMSE/NSE validation is meaningful.

If WeatherLink does not provide point-level coordinates for each probe, create a
CSV like:

```csv
lsid,point,lon,lat,depth_cm,theta_r,theta_s,alpha_1_per_cm,n
1234567,Davis_10cm,148.9321,-35.0962,10,0.04,0.43,0.015,1.6
1234568,Davis_30cm,148.9323,-35.0964,30,0.04,0.43,0.015,1.6
```

The hydraulic parameters are van Genuchten parameters. If these are unknown,
leave them for a soil/hydrology human to fill in; otherwise use
`--value-mode relative_wetness` only as a relative wet/dry diagnostic, not as a
handout-style absolute validation.

## Download only

```bash
python weatherlink_validation/download_weatherlink_soil_moisture.py \
  --station-id YOUR_STATION_ID \
  --start-date 2025-04-30 \
  --end-date 2025-07-17 \
  --timezone Australia/Sydney \
  --sensor-locations-csv /path/to/weatherlink_sensor_locations.csv \
  --value-mode centibar_vg
```

The generic output CSV is:

```text
weatherlink_validation/outputs/weatherlink_soil_moisture_generic_model6_validation.csv
```

## Full model6 validation + spiking run

For Drill & Drop profile sensors, prefer the profile-mean point-only workflow
below. It averages depths to one daily profile value and evaluates model6 at the
profile coordinate without generating full daily GeoTIFFs:

```bash
NUMBA_CACHE_DIR=/private/tmp/numba_cache \
/opt/miniconda3/envs/paddockts/bin/python \
  weatherlink_validation/validate_profile_mean_point_only.py \
  --station-id 149046 \
  --lsid 591644 \
  --start-date 2025-07-21 \
  --end-date 2026-07-20
```

Depth-level validation is retained only as a diagnostic for vertical-profile
behaviour. It should not be treated as the primary model6 validation target.

```bash
python weatherlink_validation/run_weatherlink_validation.py \
  --station-id YOUR_STATION_ID \
  --start-date 2025-04-30 \
  --end-date 2025-07-17 \
  --timezone Australia/Sydney \
  --sensor-locations-csv /path/to/weatherlink_sensor_locations.csv \
  --value-mode centibar_vg \
  --overwrite-tifs
```

Outputs are written under:

```text
weatherlink_validation/outputs/
```

Key subfolders:

- `download/` — raw flattened WeatherLink records and generic validation CSV.
- `model6_point_validation/` — model6 GeoTIFFs, sampled predictions and pooled/per-point/per-date metrics.
- `Validation_2stage/` — model-input bias diagnostics, point-quality rasters and spiking learning curves.

## Important caveats

- Historic WeatherLink access generally requires Pro or Pro+ permission for the
  station.
- The WeatherLink API limits historic calls to 24-hour windows, so this script
  downloads day-sized chunks.
- Soil-water tension in centibars is not the same quantity as volumetric soil
  moisture percent. Do not use `--value-mode centibar_as_percent` except for
  debugging.
- If there is only one WeatherLink soil point, spatial spiking will be skipped;
  temporal self-spiking still needs at least three dates.
