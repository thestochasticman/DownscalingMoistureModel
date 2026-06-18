# OzNet In-Situ Data Summary

Everything about the OzNet (Murrumbidgee Soil Moisture Monitoring Network) in-situ
data used as ground truth for the SMIPS → 30 m downscaling model.

Reference: Smith et al. (2012), *The Murrumbidgee soil moisture monitoring network
data set*, Water Resources Research 48, W07701. (PDF in `paper/`.)

Code: `emt/insitu/oznet.py` (download + parse) and `emt/insitu/coordinates.py` (station lat/lon).
Environment: conda `paddockts` (Python 3.11) with `xlrd` added for legacy `.xls`.

---

## 1. How the data is served

OzNet (https://www.oznet.org.au) serves per-station, per-season **legacy `.xls`** files,
listed in a JSON manifest. No registration is needed for data older than ~12 months
(a 12-month embargo applies to the most recent data).

| Resource | URL |
|---|---|
| File manifest | `https://www.oznet.org.au/mdbdata/jsonData.json` |
| Site → station map | `https://www.oznet.org.au/mdbdata/jsonMap.json` |
| One data file | `.../data/processed/webData/{site}/{station}/{station}_{yy}_{season}.xls` |
| Station coords page | `https://www.oznet.org.au/{station}.html` |

- `jsonData.json` → `{"data": [{link, period, site, station, year}, ...]}`. The `link`
  field is HTML (`<a href='…'>Download</a>`); the file URL is extracted with regex.
- `season` ∈ `su` (summer), `au` (autumn), `wi` (winter), `sp` (spring); `yy` is the 2-digit year.
- Some filenames carry a `_sm` suffix.

---

## 2. Inventory (`fetch_manifest`)

```
3,626 files | 75 stations | years 2001–2025 | seasons su/au/wi/sp
stations per site:  YANCO 50   KYEAMBA 13   MURRUMBIDGEE 7   ADELONG 5
```

- 4 Murrumbidgee catchment sites are kept; the **JAXA** flux site is excluded.
- The 50 YANCO stations include the dense post-2009 SMAP cal/val focus-grid
  (`YA*` / `YB*`); only the 13 primary Y1–Y13 are used here (see §5).

---

## 3. File structure (`.xls`)

Each workbook contains sheets: a main sub-daily sheet (`30min Data` or `20min Data`),
`Daily Rainfall`, sometimes `6min Rainfall`, `MetaData`, and chart sheets.

Main data sheet:

| Row | Content |
|---|---|
| 0 | Site title (e.g. `Canberra Airport - Flat (M2)`) |
| 1 | Column headers |
| 2 | Units |
| 3+ | Data |

- **`DATE-TIME`** is an Excel serial number (epoch 1899-12-30; times in
  Australian Eastern Standard Time, no daylight saving).
- **Missing value flag = `-99.0`** (mapped to NaN; values `<= -99` masked).
- Soil moisture is **volumetric %** (`%vol`).

Columns vary by site generation:

| Generation | Example | Temp depths | Soil-moisture columns |
|---|---|---|---|
| First (2001) | M2 | 4, 15, 45, 75 cm | `SM 0-8cm`, `SM 0-30cm`, `SM 30-60cm`, `SM 60-90cm` (+ Suction) |
| Second (2003+) | Y1 | 2.5, 15 cm | `SM 0-5cm`, `SM 0-30cm`, `SM 30-60cm`, `SM 60-90cm` |

The surface layer (`SM 0-5cm` / `SM 0-8cm`) differs by generation and is **not** part of
the root-zone integral.

---

## 4. Processing pipeline

```
fetch_manifest()      → DataFrame [site, station, year, period, url]
download_oznet()      → downloads .xls into data/oznet/ (cached, mirrors remote layout)
parse_xls()           → tidy sub-daily DataFrame (Excel-serial → datetime, -99 → NaN)
load_daily_rootzone() → daily root-zone 0–90 cm per station, long format
```

**Root-zone 0–90 cm** (the target variable, matched to SMIPS `TotalBucketRaw`):

```
sm_rootzone = mean(SM 0-30cm, SM 30-60cm, SM 60-90cm)   # equal 30 cm layers
```

Computed daily; a day is only kept if all available deep layers are present.
Seasonal files overlap at boundaries → duplicate station-days are averaged.
Output columns: `[site, station, time, sm_rootzone_pct, n_layers]`.

### Parser checks (two sample files)

```
m2_08_wi_sm.xls  (M2, winter 2008, 1st gen)   shape (4416, 13)   2008-06-01 → 08-31
  → 13 valid daily root-zone values (mid-season -99 outage), mean 17.5 %vol
y1_25_su.xls     (Y1, summer 2025, 2nd gen)   shape (6480, 7)    2025-12-01 → 2026-02-28
  → SM 0-5cm all NaN (Hydraprobe incomplete); root-zone still formed from 3 deep layers
  → 90 valid daily root-zone values, mean 34.4 %vol
```

### End-to-end validation (ADELONG, 2006–2007)

```
31 files → downloaded 31/31 → 1,757 station-days
  A1  count 724  mean 32.1 %vol
  A3  count 309  mean 22.9 %vol
  A5  count 724  mean 24.1 %vol
Paper's Adelong annual-median root-zone range: 23.5–33.6 %v/v  ✅ reproduced
```

---

## 5. Station coordinates (`station_coords.csv`)

Coordinates are **not** in any data file or JSON manifest. Each core station has an HTML
page with a line like:

```
Latitude: -35.3088, Longitude: 149.2000 Elevation: 639m
```

`emt/insitu/coordinates.py` scrapes this by regex and caches to
`data/oznet/station_coords.csv` (skips cached / known-404 stations on re-run).

**Coverage: 38/38 core sites** (= the paper's "38 core sites").
Spatial extent: lon **143.549° → 149.200°**, lat **−36.293° → −33.938°**, elevation **113–772 m**.

**MURRUMBIDGEE (7)** — regional, lat/lon only (no elevation on page → fill from DEM):

| station | lat | lon |
|---|---|---|
| M1 | −36.293033 | 148.970567 |
| M2 | −35.308800 | 149.200000 |
| M3 | −34.629867 | 148.036500 |
| M4 | −33.938267 | 147.196183 |
| M5 | −34.658370 | 143.548630 |
| M6 | −34.547117 | 144.867000 |
| M7 | −34.249000 | 146.070000 |

**YANCO (13)** — western plains, 113–149 m:

| station | lat | lon | elev (m) |
|---|---|---|---|
| Y1 | −34.62888 | 145.84895 | 120 |
| Y2 | −34.65478 | 146.11028 | 130 |
| Y3 | −34.62080 | 146.42390 | 144 |
| Y4 | −34.71943 | 146.02003 | 130 |
| Y5 | −34.72835 | 146.29317 | 136 |
| Y6 | −34.84262 | 145.86692 | 121 |
| Y7 | −34.85183 | 146.11530 | 128 |
| Y8 | −34.84697 | 146.41398 | 149 |
| Y9 | −34.96777 | 146.01632 | 122 |
| Y10 | −35.00535 | 146.30988 | 119 |
| Y11 | −35.10975 | 145.93553 | 113 |
| Y12 | −35.06960 | 146.16893 | 120 |
| Y13 | −35.09025 | 146.30648 | 121 |

**KYEAMBA (13)** — mid catchment, gentle slopes, 184–437 m:

| station | lat | lon | elev (m) |
|---|---|---|---|
| K1 | −35.49322 | 147.55912 | 437 |
| K2 | −35.43525 | 147.53052 | 351 |
| K3 | −35.43408 | 147.56893 | 318 |
| K4 | −35.42688 | 147.60000 | 296 |
| K5 | −35.41928 | 147.60408 | 306 |
| K6 | −35.38978 | 147.45720 | 317 |
| K7 | −35.39392 | 147.56618 | 259 |
| K8 | −35.31627 | 147.34387 | 326 |
| K10 | −35.32395 | 147.53480 | 232 |
| K11 | −35.27202 | 147.42902 | 327 |
| K12 | −35.22750 | 147.48500 | 220 |
| K13 | −35.23887 | 147.53330 | 261 |
| K14 | −35.12493 | 147.49740 | 184 |

**ADELONG (5)** — eastern hills, steep, 379–772 m:

| station | lat | lon | elev (m) |
|---|---|---|---|
| A1 | −35.497492 | 148.106488 | 772 |
| A2 | −35.428312 | 148.131626 | 595 |
| A3 | −35.399688 | 148.101076 | 472 |
| A4 | −35.373106 | 148.066082 | 457 |
| A5 | −35.360194 | 148.085427 | 379 |

**Missing — 37 dense Yanco focus-grid stations (404, no page):**

```
YA1, YA3, YA4A–E, YA5, YA7A/B/D/E, YA8A–D, YA9, YA9B–D,
YB1, YB3, YB5A/B/D/E, YB7A/C/D/E, YB9   (plus lowercase duplicates)
```

These are the post-2009 SMAP cal/val additions. If denser Yanco training is wanted later,
source their coordinates from the ISMN (International Soil Moisture Network).

The west→east elevation rise (Yanco ~120 m → Kyeamba ~300 m → Adelong up to 772 m)
matches the paper's climate/topography gradient, confirming the scrape is correct.

---

## 6. Design note for downscaling

The 38 sites span the **entire ~82,000 km² catchment** (regional M sites from lon 143.5°
to 149.2°). Do **not** build a single 30 m DEM/SMIPS grid over the whole catchment.
Instead: extract SMIPS + topography in small windows at each station for *training*, and
only produce full downscaled 30 m fields over chosen focus AOIs (Yanco / Kyeamba / Adelong).

---

## Files

```
data/oznet/
├── station_coords.csv          # 38 core stations: station, lat, lon, elevation_m
└── {site}/{station}/*.xls       # downloaded seasonal files (mirrors remote layout)
```
