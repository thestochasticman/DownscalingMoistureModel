# `smips.py` — coarse SMIPS input (~1 km) + the native-grid fix

Source: [`../../emt/smips.py`](../../emt/smips.py)

Loads the **coarse field we are downscaling**: SMIPS `TotalBucket` profile soil
water (mm), ~1 km daily, from TERN. This is the model's single coarse predictor
and, ultimately, the field that gets sharpened to 30 m.

## What it does
| function | role |
|---|---|
| `download_smips(query, var="totalbucket")` | cached `(time, y, x)` DataArray of raw SMIPS for a `Query` |
| `smips_cube(start, end, bbox, …)` | fetch a multi-day cube (threaded, one request per day) |
| `smips_day(d, bbox, …)` | one day as a 2-D field |
| `snap_bbox(bbox, pad=1)` | **expand a request bbox to native grid lines** (the fix below) |

## Why an EMT-local loader at all
PaddockTS's own SMIPS downloader points at TERN's decommissioned WMS (404). The
replacement GeoServer **WMS** only serves a styled 8-bit palette image (no raw
values); the daily COGs need a TERN login. The GeoServer **WCS** endpoint is
public and returns the raw float32 field — so EMT reads SMIPS from there.

## The bug this module fixes ⚠️
TERN's GeoServer WCS **resamples every `GetCoverage` to fit an integer pixel
count into the _requested_ bbox**. So a point's sampled value depended on the
request window: a narrow per-station box and a wide cluster box land on
differently-shifted grids and can pick **different ~1 km cells**. Observed at
station K6: **43.7 / 49.6 / 61.6 mm for three windows** — a ~40% swing for the
same place and day.

- `scaleFactor=1` does **not** help — verified no-op (byte-identical output).
- **Fix:** `snap_bbox()` expands the request to the native cell boundaries (from
  the WCS `DescribeCoverage`: west `112.90499`, south `-43.73500`, step
  `0.0099976°`). With an aligned envelope GeoServer's integer-pixel fit *equals*
  the native grid, so sampling is **window-independent** — a station gets its
  true native pixel regardless of how much area is requested.

This is what makes the cluster-fetch optimisation in
[`features.py`](features.py.md) provably identical to per-station fetching, and
it corrected the training data — see Figure 1 in the [README](../README.md).

> **Consequence:** any SMIPS cached before this fix held window-dependent
> (wrong) values; the training table and model were rebuilt with caches cleared.
