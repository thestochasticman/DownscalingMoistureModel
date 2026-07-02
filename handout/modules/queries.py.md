# `queries.py`: study-area definitions

<!-- NAV -->
[← Ground truth (OzNet)](oznet.py.md) · [Index](../README.md) · [SMIPS coarse predictor →](smips.py.md)
<!-- /NAV -->

Source: [`../../emt/queries.py`](../../emt/queries.py)

Constructs PaddockTS [`Query`](../../emt/queries.py) objects. A `Query` carries a
bounding box and date range and drives every PaddockTS download and its on-disk
cache. The project reuses this type throughout rather than defining a separate
area-of-interest abstraction, so that SMIPS and terrain retrieval share a single
caching scheme.

| Function | Role |
|---|---|
| `query_for_station(station, lat, lon, start, end, buffer_km=1.5)` | Small window around one OzNet station for point extraction |
| `query_for_focus_area(name, start, end)` | One clustered focus catchment (`yanco` / `kyeamba` / `adelong`) for full-field downscaling |
| `queries_for_stations(coords, …)` | One per-station query for each row of a coordinates table |

## Design decisions

- **Stub encodes the date range** (`oznet_{station}_{YYYYMMDD}_{YYYYMMDD}`). The
  PaddockTS registry maps each stub to a single `(bbox, time)` pair; encoding the
  period keeps the same station across different study periods in distinct cache
  entries and avoids registry collisions.
- **`FOCUS_AREAS` covers the three clustered catchments only.** The dispersed
  regional M1–M7 sites are handled per-station; a single grid over the full
  catchment (≈82,000 km²) is intentionally avoided.

Queries are consumed by [`smips.py`](smips.py.md) and
[`covariates.py`](covariates.py.md).

---

---
<!-- NAV -->
[← Ground truth (OzNet)](oznet.py.md) · [Index](../README.md) · [SMIPS coarse predictor →](smips.py.md)
<!-- /NAV -->
