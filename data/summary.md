# EMT Data Summary

Soil-moisture downscaling project: **SMIPS ~1 km → ~30 m** (topographic resolution),
trained/validated against the OzNet Murrumbidgee in-situ network.

Branch: `EMT`. Environment: conda `paddockts` (Python 3.11; `xlrd` added for legacy `.xls`).

---

## Data sources

| Role | Source | Resolution | Status |
|---|---|---|---|
| Coarse input (to downscale) | SMIPS `TotalBucketRaw` (profile soil water, mm), TERN WMS | ~1 km daily | adapter not yet ported |
| Fine covariates / target grid | Copernicus DEM 30 m + slope/aspect/TWI/HLI/flow-accumulation | ~30 m | adapter not yet ported |
| Ground truth (train/validate) | OzNet Murrumbidgee in-situ, root-zone 0–90 cm | point, calibrated | **done** |

Target pairing: SMIPS `TotalBucketRaw` ↔ in-situ **root-zone 0–90 cm** (= mean of the
0–30, 30–60, 60–90 cm layers). Model class: **ML regression (Random Forest / gradient boosting)**.

---

## Stage 1 — OzNet in-situ ingest (DONE)

Code: `emt/insitu/oznet.py`

- **Manifest:** `https://www.oznet.org.au/mdbdata/jsonData.json` → 3,626 files,
  75 stations across 4 Murrumbidgee sites (JAXA flux site excluded).
- **File URL pattern:** `.../data/processed/webData/{site}/{station}/{station}_{yy}_{season}.xls`
  (season ∈ su/au/wi/sp). Cached locally mirroring this layout.
- **Excel layout:** main sheet `30min`/`20min Data`; row 1 = headers, row 2 = units,
  row 3+ = data. `DATE-TIME` is an Excel serial (Australian EST, no DST). Missing flag = `-99.0`.
  Soil moisture in volumetric %.
- **Pipeline:** `fetch_manifest()` → `download_oznet()` (cached) → `parse_xls()` →
  `load_daily_rootzone()` (daily root-zone 0–90 cm per station, long format).

**Validation:** ADELONG A1/A3/A5 root-zone means = 32.1 / 22.9 / 24.1 % vol, inside the
paper's reported Adelong annual-median range of 23.5–33.6 % v/v. ✅

---

## Station coordinates (DONE)

Code: `emt/insitu/coordinates.py` → cached at `data/oznet/station_coords.csv`

Coordinates are not in any OzNet data file. Each core station has an HTML page
`https://www.oznet.org.au/{station}.html` with a line
`Latitude: -35.3088, Longitude: 149.2000 Elevation: 639m`, scraped by regex.

**Coverage: 38/38 core sites** (matches the paper's "38 core sites"):

| Site | Stations w/ coords | Elevation range |
|---|---|---|
| MURRUMBIDGEE | M1–M7 (7) | — (lat/lon only; fill from DEM) |
| YANCO | Y1–Y13 (13) | 113–149 m |
| KYEAMBA | K1–K14 (13) | 184–437 m |
| ADELONG | A1–A5 (5) | 379–772 m |

Dense Yanco focus-grid stations (`YA*`/`YB*`, ~37) have no page (404) → only via ISMN if
ever needed. Spatial extent of the 38 sites: lon 143.549° → 149.200°, lat −36.293° → −33.938°
(west-low → east-high, consistent with the paper).

**Design implication:** the 38 sites span the entire ~82,000 km² catchment. Do **not** build a
single 30 m grid over the whole catchment — extract SMIPS + topo in small windows at each
station for training, and only produce full downscaled 30 m fields over chosen focus AOIs
(Yanco / Kyeamba / Adelong).

---

## Remaining stages

2. **SMIPS adapter** — gridded daily cube over AOI (port from PaddockTS `download_smips`).
3. **Terrain adapter** — DEM 30 m + slope/aspect/TWI/HLI/flow-accumulation (port from PaddockTS TerrainTiles).
4. **Feature table** — at each (station, date): SMIPS-at-pixel + topo-at-pixel + temporal features → target (needs coords ✓).
5. **Model** — Random Forest / GBM with leave-site-out spatial cross-validation.
6. **Downscale** — apply to every 30 m pixel; optional mass-conservation (re-aggregate to SMIPS mean).
7. **Evaluate** — RMSE / ubRMSE / R² / bias vs held-out sites.

---

## Detailed findings (terminal output)

### Manifest inventory (`fetch_manifest`)

```
manifest: 3,626 files | 75 stations | years 2001–2025 | seasons su/au/wi/sp
stations per site:  YANCO 50   KYEAMBA 13   MURRUMBIDGEE 7   ADELONG 5
```

(50 YANCO stations include the dense YA*/YB* focus-grid; only Y1–Y13 have web pages.)

### Parser check on two sample files (`parse_xls`)

```
m2_08_wi_sm.xls  (MURRUMBIDGEE M2, winter 2008, first generation)
  shape (4416, 13)
  cols: Temp 4cm, Temp 15cm, Temp 45cm, Temp 75cm,
        SM 0-8cm, SM 0-30cm, SM 30-60cm, SM 60-90cm,
        Suction 4/15/45/75cm, 30min Rainfall
  time range 2008-06-01 → 2008-08-31 23:30
  → daily root-zone: 13 valid days (rest -99 / mid-season outage), mean 17.5 %vol

y1_25_su.xls  (YANCO Y1, summer 2025, second generation)
  shape (6480, 7)
  cols: Temp 2.5cm, Temp 15cm, SM 0-5cm, SM 0-30cm, SM 30-60cm, SM 60-90cm, 20min Rainfall
  time range 2025-12-01 → 2026-02-28 23:40
  SM 0-5cm all NaN (Hydraprobe incomplete) — root-zone still formed from 3 deep layers
  → daily root-zone: 90 valid days, mean 34.4 %vol
```

### End-to-end validation (ADELONG, 2006–2007)

```
subset files: 31  → downloaded 31/31  → 1,757 station-days
station   count   mean(%vol)
  A1        724    32.1
  A3        309    22.9
  A5        724    24.1
Paper's reported Adelong annual-median root-zone range: 23.5–33.6 %v/v  ✅
```

### Station coordinates (`station_coords.csv`) — 38/38 core sites

```
Extent: lon 143.549 → 149.200   lat -36.293 → -33.938   elevation 113 → 772 m
```

**MURRUMBIDGEE (7)** — regional sites, lat/lon only (elevation NaN → fill from DEM):

```
station        lat        lon
   M1   -36.293033  148.970567
   M2   -35.308800  149.200000   (Canberra airport, 639 m per page)
   M3   -34.629867  148.036500
   M4   -33.938267  147.196183
   M5   -34.658370  143.548630   (far-western, semiarid)
   M6   -34.547117  144.867000
   M7   -34.249000  146.070000
```

**YANCO (13)** — western plains:

```
station      lat       lon    elev(m)
   Y1   -34.62888  145.84895    120
   Y2   -34.65478  146.11028    130
   Y3   -34.62080  146.42390    144
   Y4   -34.71943  146.02003    130
   Y5   -34.72835  146.29317    136
   Y6   -34.84262  145.86692    121
   Y7   -34.85183  146.11530    128
   Y8   -34.84697  146.41398    149
   Y9   -34.96777  146.01632    122
   Y10  -35.00535  146.30988    119
   Y11  -35.10975  145.93553    113
   Y12  -35.06960  146.16893    120
   Y13  -35.09025  146.30648    121
```

**KYEAMBA (13)** — mid catchment, gentle slopes:

```
station      lat       lon    elev(m)
   K1   -35.49322  147.55912    437
   K2   -35.43525  147.53052    351
   K3   -35.43408  147.56893    318
   K4   -35.42688  147.60000    296
   K5   -35.41928  147.60408    306
   K6   -35.38978  147.45720    317
   K7   -35.39392  147.56618    259
   K8   -35.31627  147.34387    326
   K10  -35.32395  147.53480    232
   K11  -35.27202  147.42902    327
   K12  -35.22750  147.48500    220
   K13  -35.23887  147.53330    261
   K14  -35.12493  147.49740    184
```

**ADELONG (5)** — eastern hills, steep, highest elevations:

```
station        lat         lon    elev(m)
   A1   -35.497492  148.106488    772
   A2   -35.428312  148.131626    595
   A3   -35.399688  148.101076    472
   A4   -35.373106  148.066082    457
   A5   -35.360194  148.085427    379
```

**Missing (37 dense Yanco focus-grid, no page / 404):**

```
YA1, YA3, YA4A–E, YA5, YA7A/B/D/E, YA8A–D, YA9, YA9B–D,
YB1, YB3, YB5A/B/D/E, YB7A/C/D/E, YB9   (+ lowercase duplicates)
```

---

## Cache layout (`data/`, gitignored)

```
data/
├── summary.md                  # this file
├── oznet/
│   ├── station_coords.csv      # 38 core station lat/lon/elevation
│   └── {site}/{station}/*.xls  # downloaded seasonal files (mirrors remote layout)
├── smips/                      # (stage 2)
└── terrain/                    # (stage 3)
```
