"""Download and parse OzNet Murrumbidgee in-situ soil moisture.

The OzNet archive (https://www.oznet.org.au) serves per-station, per-season
legacy ``.xls`` files listed in a JSON manifest. This module:

    fetch_manifest()      -> DataFrame of every available file (site/station/year/period/url)
    download_oznet()      -> download the matching .xls files into a local cache
    parse_xls()           -> read one .xls into a tidy sub-daily DataFrame
    load_daily_rootzone() -> combined daily, root-zone (0-90 cm) series for all stations

Data notes (see also the project memory ``oznet-data-access``):
  * Main sheet is ``30min Data`` or ``20min Data``; row 1 = headers, row 2 = units, row 3+ = data.
  * ``DATE-TIME`` is an Excel serial number (Australian Eastern Standard Time, no DST).
  * Missing value flag is -99.0; soil moisture is volumetric %.
  * Root-zone 0-90 cm (to match SMIPS TotalBucketRaw) = mean of the 0-30, 30-60, 60-90 cm layers.
"""
from __future__ import annotations

import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import xlrd

from emt.config import OZNET_DIR, MURRUMBIDGEE_SITES

MANIFEST_URL = "https://www.oznet.org.au/mdbdata/jsonData.json"
MAP_URL = "https://www.oznet.org.au/mdbdata/jsonMap.json"

# Soil-moisture layers present in every file generation, used to build the
# root-zone 0-90 cm average. The surface layer (SM 0-5cm / SM 0-8cm) varies by
# generation and is *not* part of the root-zone integral.
ROOTZONE_LAYERS = ("SM 0-30cm", "SM 30-60cm", "SM 60-90cm")

MISSING_FLAG = -99.0

_HREF_RE = re.compile(r"href='([^']+)'")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def fetch_manifest(sites: tuple[str, ...] | None = MURRUMBIDGEE_SITES,
                   timeout: int = 60) -> pd.DataFrame:
    """Fetch the OzNet file manifest as a DataFrame.

    Args:
        sites: Site names to keep (case-insensitive). ``None`` keeps all
            sites including the JAXA flux site. Defaults to the four
            Murrumbidgee catchment sites.
        timeout: Request timeout in seconds.

    Returns:
        DataFrame with columns ``[site, station, year, period, url]``,
        sorted by site/station/year/period.
    """
    r = requests.get(MANIFEST_URL, timeout=timeout)
    r.raise_for_status()
    rows = r.json()["data"]

    records = []
    for row in rows:
        m = _HREF_RE.search(row.get("link", ""))
        if not m:
            continue
        records.append({
            "site": row["site"],
            "station": row["station"],
            "year": int(row["year"]),
            "period": row["period"],
            "url": m.group(1),
        })
    df = pd.DataFrame.from_records(records)

    if sites is not None:
        wanted = {s.upper() for s in sites}
        df = df[df["site"].str.upper().isin(wanted)]

    return df.sort_values(["site", "station", "year", "period"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _local_path(url: str, out_dir: Path) -> Path:
    """Mirror the remote ``.../{site}/{station}/{file}.xls`` layout locally."""
    parts = url.split("/data/processed/webData/", 1)[-1]
    return out_dir / parts


def download_file(url: str, out_dir: Path = OZNET_DIR, timeout: int = 120) -> Path:
    """Download a single OzNet .xls, skipping if already cached. Returns the path."""
    dest = _local_path(url, out_dir)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


# The OzNet server throttles PER CONNECTION, not per client. Measured: a single
# stream holds ~9.5 KB/s regardless of which file, while eight concurrent
# streams each held ~10.3 KB/s for an aggregate of 75.9 KB/s -- an 8.2x
# speed-up with no per-connection penalty. Sequentially the full 2001-2025
# record is ~110 hours of downloading; at this width it is hours, not days.
#
# Kept deliberately modest: this is a public research archive on a plainly slow
# link, and there is nothing to gain from opening more streams than needed.
DOWNLOAD_WORKERS = 12


def download_oznet(sites: tuple[str, ...] | None = MURRUMBIDGEE_SITES,
                   out_dir: Path = OZNET_DIR,
                   manifest: pd.DataFrame | None = None,
                   verbose: bool = True,
                   workers: int = DOWNLOAD_WORKERS) -> pd.DataFrame:
    """Download all OzNet files for ``sites`` into ``out_dir`` (cached).

    Fetches ``workers`` files concurrently -- see ``DOWNLOAD_WORKERS`` for why
    that is the whole ballgame here. Cached files short-circuit inside
    :func:`download_file` without opening a connection, so a resumed run costs
    nothing for what it already holds.

    Returns:
        The manifest DataFrame with an added ``path`` column pointing at the
        local file (``None`` for any download that failed).
    """
    if manifest is None:
        manifest = fetch_manifest(sites=sites)

    urls = list(manifest["url"])
    n = len(urls)
    have = sum(1 for u in urls
               if (d := _local_path(u, out_dir)).exists() and d.stat().st_size > 0)
    if verbose and n:
        print(f"  {have}/{n} already cached; fetching {n - have} with "
              f"{workers} workers", flush=True)

    def fetch(url: str):
        try:
            return url, download_file(url, out_dir=out_dir)
        except (requests.RequestException, OSError) as e:      # noqa: BLE001
            return url, e

    paths: dict[str, Path | None] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for url, res in ex.map(fetch, urls):
            done += 1
            if isinstance(res, Path):
                paths[url] = res
            else:
                paths[url] = None
                if verbose:
                    print(f"  FAILED {url}: {type(res).__name__}: {res}", flush=True)
            if verbose and (done % 50 == 0 or done == n):
                print(f"  [{done}/{n}] cached", flush=True)

    out = manifest.copy()
    out["path"] = [paths.get(u) for u in out["url"]]
    return out


# --------------------------------------------------------------------------- #
# Parse
# --------------------------------------------------------------------------- #
def _main_sheet_name(book: xlrd.book.Book) -> str:
    """Return the name of the sub-daily data sheet (``'30min Data'``/``'20min Data'``)."""
    for name in book.sheet_names():
        low = name.lower()
        if "min data" in low or low.endswith("data"):
            return name
    # Fall back to the first sheet.
    return book.sheet_names()[0]


def _unique_headers(names: list[str]) -> list[str]:
    """Make column names unique (repeats get a ``.1``/``.2``… suffix).

    Some OzNet files repeat a header or leave it blank; keyed by name those
    columns would collapse into one. The first occurrence keeps its exact name
    (so e.g. ``'SM 0-30cm'`` stays matchable), later repeats are suffixed.
    """
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def parse_xls(path: str | Path) -> pd.DataFrame:
    """Parse one OzNet .xls into a tidy sub-daily DataFrame.

    Returns:
        DataFrame indexed by timezone-naive datetime (Australian EST), with one
        column per measured variable as named in the file (e.g. ``'SM 0-30cm'``,
        ``'Temp 4cm'``, ``'30min Rainfall'``). Missing values (-99) become NaN.
        Returns an empty DataFrame if the sheet has no parseable data rows.
    """
    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        book = xlrd.open_workbook(path)
    sheet = book.sheet_by_name(_main_sheet_name(book))

    if sheet.nrows < 4:
        return pd.DataFrame()

    # Row 1 = headers (row 0 is the site title). Collect columns positionally
    # (not keyed by header name) and uniquify the names, so files with a
    # duplicate or blank header don't collapse two columns into one.
    headers = _unique_headers([str(sheet.cell_value(1, c)).strip()
                               for c in range(sheet.ncols)])

    times = []
    data: list[list] = [[] for _ in range(1, sheet.ncols)]
    for r in range(3, sheet.nrows):
        serial = sheet.cell_value(r, 0)
        if serial in ("", None):
            continue
        try:
            dt = xlrd.xldate.xldate_as_datetime(float(serial), book.datemode)
        except (ValueError, TypeError):
            continue
        times.append(dt)
        for j, c in enumerate(range(1, sheet.ncols)):
            val = sheet.cell_value(r, c)
            data[j].append(val if val != "" else float("nan"))

    if not times:
        return pd.DataFrame()

    df = pd.DataFrame(dict(zip(headers[1:], data)),
                      index=pd.DatetimeIndex(times, name="time"))
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.mask(df <= MISSING_FLAG)  # -99 (and anything below) -> NaN
    return df


def _station_daily_rootzone(path: str | Path) -> pd.DataFrame | None:
    """Daily root-zone (0-90 cm) mean soil moisture from one file, or None."""
    df = parse_xls(path)
    if df.empty:
        return None
    have = [c for c in ROOTZONE_LAYERS if c in df.columns]
    if not have:
        return None
    daily = df[have].resample("D").mean()
    # Require all available layers present that day to form the integral.
    rootzone = daily[have].mean(axis=1, skipna=False)
    out = rootzone.dropna().rename("sm_rootzone_pct").to_frame()
    out["n_layers"] = len(have)
    return out


def load_daily_rootzone(manifest: pd.DataFrame | None = None,
                        sites: tuple[str, ...] | None = MURRUMBIDGEE_SITES,
                        out_dir: Path = OZNET_DIR,
                        verbose: bool = True) -> pd.DataFrame:
    """Build the combined daily root-zone (0-90 cm) series for all stations.

    Downloads any missing files first, then parses and concatenates.

    Returns:
        Long-format DataFrame with columns
        ``[site, station, time, sm_rootzone_pct, n_layers]`` (one row per
        station-day), volumetric %.
    """
    if manifest is None or "path" not in getattr(manifest, "columns", []):
        manifest = download_oznet(sites=sites, out_dir=out_dir, manifest=manifest,
                                  verbose=verbose)

    frames = []
    for row in manifest.itertuples(index=False):
        if row.path is None:
            continue
        rz = _station_daily_rootzone(row.path)
        if rz is None:
            continue
        rz = rz.reset_index()
        rz.insert(0, "site", row.site)
        rz.insert(1, "station", row.station)
        frames.append(rz)

    if not frames:
        return pd.DataFrame(columns=["site", "station", "time", "sm_rootzone_pct", "n_layers"])

    out = pd.concat(frames, ignore_index=True)
    # Seasonal files overlap at boundaries; average duplicate station-days.
    out = (out.groupby(["site", "station", "time"], as_index=False)
              .agg(sm_rootzone_pct=("sm_rootzone_pct", "mean"),
                   n_layers=("n_layers", "max")))
    return out.sort_values(["site", "station", "time"]).reset_index(drop=True)


if __name__ == "__main__":
    man = fetch_manifest()
    print(f"manifest: {len(man)} files across {man['site'].nunique()} sites, "
          f"{man['station'].nunique()} stations, years {man['year'].min()}-{man['year'].max()}")
    print(man.head())
