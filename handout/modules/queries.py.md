# `queries.py` — study-area definitions (AOIs)

Source: [`../../emt/queries.py`](../../emt/queries.py)

Thin helpers that build PaddockTS [`Query`](../../emt/queries.py) objects — the
single object that carries a bbox + date range and drives **every** PaddockTS
downloader and its on-disk cache. EMT does not define its own AOI type; it reuses
`Query` everywhere so SMIPS and terrain share one caching scheme.

| function | role |
|---|---|
| `query_for_station(station, lat, lon, start, end, buffer_km=1.5)` | small square window around one OzNet station, for point extraction |
| `query_for_focus_area(name, start, end)` | one of the clustered focus catchments (`yanco` / `kyeamba` / `adelong`) for full-field downscaling |
| `queries_for_stations(coords, …)` | one per-station `Query` for every row of a coords table |

## Key decisions
- **The stub encodes the date range** (`oznet_{station}_{YYYYMMDD}_{YYYYMMDD}`).
  PaddockTS's registry maps each stub to a single `(bbox, time)`, so embedding the
  period keeps the same station over different study periods in distinct,
  human-readable cache entries (and avoids registry collisions).
- `FOCUS_AREAS` holds the three **clustered** catchments only; the scattered
  regional M1–M7 sites are handled per-station (see the design note in the
  project memory — don't build one giant grid over the whole 82,000 km²).

→ Queries are consumed by [`smips.py`](smips.py.md) and
[`covariates.py`](covariates.py.md).
