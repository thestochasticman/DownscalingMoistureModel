# `oznet.py` — OzNet in-situ ground truth (Stage 1)

Source: [`../../emt/insitu/oznet.py`](../../emt/insitu/oznet.py)

Downloads and parses the OzNet Murrumbidgee in-situ soil-moisture archive into
the **target** the model learns: a daily, root-zone (0–90 cm) volumetric soil
moisture series per station.

## What it does
The OzNet site serves per-station, per-season legacy `.xls` files listed in a
JSON manifest. The module walks that manifest → downloads → parses → reduces to
a daily root-zone series.

| function | role |
|---|---|
| `fetch_manifest(sites=…)` | DataFrame of every available file: `site, station, year, period, url` |
| `download_oznet(…)` | download the matching `.xls` files into a local cache (adds a `path` column) |
| `parse_xls(path)` | read one `.xls` → tidy **sub-daily** DataFrame, one column per measured variable |
| `load_daily_rootzone(…)` | the public entry point → combined **daily root-zone** series for all stations |

## Key decisions
- **Root zone = 0–90 cm** to match SMIPS `TotalBucket`: the mean of the
  `SM 0-30cm`, `SM 30-60cm`, `SM 60-90cm` layers (`ROOTZONE_LAYERS`). The surface
  layer (0–5/0–8 cm) varies by file generation and is excluded.
- A day's root-zone value requires **all available layers present** that day
  (`skipna=False`) — partial days are dropped rather than biased.
- Quirks handled: main sheet is `30min Data`/`20min Data`; `DATE-TIME` is an
  Excel serial (Australian EST, no DST); missing flag `-99` → NaN; overlapping
  seasonal files are **averaged** at their boundaries.

## Output
Long format, one row per station-day:
`[site, station, time, sm_rootzone_pct, n_layers]` (volumetric %).

→ consumed by [`features.py`](features.py.md) as the `sm_rootzone_pct` target.
