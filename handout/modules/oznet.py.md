# `oznet.py`: OzNet in-situ ground truth (Stage 1)

<!-- NAV -->
[Index](../README.md) · [Study areas →](queries.py.md)
<!-- /NAV -->

Source: [`../../emt/insitu/oznet.py`](../../emt/insitu/oznet.py)

Retrieves and processes the OzNet Murrumbidgee in-situ soil-moisture archive to
produce the regression target: a daily root-zone (0–90 cm) volumetric
soil-moisture series per station.

## Interface

The OzNet archive provides per-station, per-season `.xls` files indexed by a JSON
manifest. The module reads the manifest, downloads the files, parses them, and
reduces them to a daily root-zone series.

| Function | Role |
|---|---|
| `fetch_manifest(sites=…)` | Table of available files: `site, station, year, period, url` |
| `download_oznet(…)` | Download matching `.xls` files to a local cache (adds a `path` column) |
| `parse_xls(path)` | Parse one `.xls` to a sub-daily table, one column per measured variable |
| `load_daily_rootzone(…)` | Entry point; combined daily root-zone series across stations |

## Processing decisions

- **Root zone defined as 0–90 cm** to correspond to SMIPS `TotalBucket`: the mean
  of the `SM 0-30cm`, `SM 30-60cm`, and `SM 60-90cm` layers (`ROOTZONE_LAYERS`).
  The surface layer (0–5/0–8 cm) varies between file generations and is excluded.
- **Complete layers required per day** (`skipna=False`): days lacking any of the
  three layers are dropped rather than averaged from a partial profile.
- **Format handling:** primary sheet `30min Data`/`20min Data`; `DATE-TIME` stored
  as an Excel serial (Australian EST, no daylight saving); missing-value flag −99
  mapped to NaN; overlapping seasonal files averaged at their boundaries.
- **Header robustness:** some files contain duplicate or blank column headers;
  columns are collected positionally and names are uniquified (`_unique_headers`)
  to prevent column collisions during parsing.

## Output

Long format, one record per station-day:
`[site, station, time, sm_rootzone_pct, n_layers]` (volumetric %). Consumed by
[`features.py`](features.py.md) as the `sm_rootzone_pct` target.

---

---
<!-- NAV -->
[Index](../README.md) · [Study areas →](queries.py.md)
<!-- /NAV -->
