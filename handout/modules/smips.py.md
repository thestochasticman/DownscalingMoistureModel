# `smips.py`: SMIPS coarse predictor (≈1 km)

<!-- NAV -->
[← Study areas](queries.py.md) · [Index](../README.md) · [Terrain covariates →](covariates.py.md)
<!-- /NAV -->

Source: [`../../emt/smips.py`](../../emt/smips.py)

Retrieves the SMIPS `TotalBucket` profile soil-water field (mm, ≈1 km, daily)
from TERN. This field is the coarse predictor being downscaled.

| Function | Role |
|---|---|
| `download_smips(query, var="totalbucket")` | Cached `(time, y, x)` field for a `Query` |
| `smips_cube(start, end, bbox, …)` | Multi-day field (one threaded request per day) |
| `smips_day(d, bbox, …)` | Single-day 2-D field |
| `snap_bbox(bbox, pad=1)` | Align a request bbox to the native SMIPS grid |

## Rationale for a project-local loader

The PaddockTS SMIPS downloader targets TERN's decommissioned WMS endpoint
(returns 404). The replacement GeoServer WMS serves a styled 8-bit palette image
rather than raw values, and the daily Cloud-Optimized GeoTIFFs require
authentication. The GeoServer WCS endpoint serves the raw float32 field without
authentication and is used here.

## Grid alignment

The WCS resamples each `GetCoverage` response to fit an integer pixel count
within the requested bounding box, so the returned grid origin and cell size
depend on the request extent. A fixed point can therefore fall in different
≈1 km cells depending on the window requested; for station K6 the sampled value
ranged over 43.7–61.6 mm across three windows for the same location and date.

`snap_bbox` aligns each request to the native grid taken from the WCS
`DescribeCoverage` (origin 112.90499°E, −43.73500°N; cell size 0.0099976°). With
an aligned envelope the server's integer-pixel fit coincides with the native
grid, making the sampled value independent of the request window. The
`scaleFactor=1` parameter was tested and produces byte-identical output (no
effect).

This alignment makes the cluster-fetch retrieval in
[`features.py`](features.py.md) value-identical to per-station retrieval
(verified: maximum absolute difference 0.000 mm). SMIPS cached before this change
used window-dependent values; downstream products were rebuilt after clearing the
cache.

---

---
<!-- NAV -->
[← Study areas](queries.py.md) · [Index](../README.md) · [Terrain covariates →](covariates.py.md)
<!-- /NAV -->
