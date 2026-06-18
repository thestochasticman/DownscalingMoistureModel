"""Scrape OzNet station coordinates from per-station web pages.

Station lat/lon/elevation are not in any OzNet data file or JSON manifest, but
each core station has an HTML page at ``https://www.oznet.org.au/{station}.html``
containing a line of the form::

    Latitude: -35.3088, Longitude: 149.2000 Elevation: 639m

The dense Yanco focus-grid stations (``YA*`` / ``YB*``, added ~2009 for SMAP
validation) have no individual page and are reported as missing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

from emt.config import OZNET_DIR

STATION_PAGE = "https://www.oznet.org.au/{station}.html"
COORDS_CACHE = OZNET_DIR / "station_coords.csv"

_COORD_RE = re.compile(
    r"Latitude:\s*(-?\d+\.\d+).{0,15}?Longitude:\s*(-?\d+\.\d+)"
    r"(?:.{0,25}?Elevation:\s*(\d+))?",
    re.I | re.S,
)


def _scrape_one(station: str, timeout: int = 30) -> dict | None:
    """Return ``{lat, lon, elevation_m}`` for one station, or None if no page/match."""
    url = STATION_PAGE.format(station=station.lower())
    r = requests.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    text = re.sub(r"<[^>]+>", " ", r.text)
    m = _COORD_RE.search(text)
    if not m:
        return None
    lat, lon, elev = m.groups()
    return {
        "lat": float(lat),
        "lon": float(lon),
        "elevation_m": float(elev) if elev else float("nan"),
    }


def fetch_station_coords(stations: list[str],
                         cache: Path = COORDS_CACHE,
                         refresh: bool = False,
                         verbose: bool = True) -> pd.DataFrame:
    """Scrape (and cache) coordinates for ``stations``.

    Only stations missing from the cache are fetched. Stations with no page
    are recorded with NaN coords so they aren't retried every run.

    Args:
        stations: Station codes (e.g. ``['M2', 'Y1', 'K1']``).
        cache: CSV cache path.
        refresh: If True, ignore the cache and re-scrape everything.
        verbose: Print progress.

    Returns:
        DataFrame with columns ``[station, lat, lon, elevation_m, has_coords]``,
        restricted to the requested ``stations``.
    """
    cached = pd.DataFrame(columns=["station", "lat", "lon", "elevation_m"])
    if cache.exists() and not refresh:
        cached = pd.read_csv(cache)

    known = set(cached["station"]) if len(cached) else set()
    todo = [s for s in dict.fromkeys(stations) if s not in known]

    new_rows = []
    for i, st in enumerate(todo, 1):
        try:
            info = _scrape_one(st)
        except requests.RequestException as e:
            if verbose:
                print(f"  [{i}/{len(todo)}] {st}: request error {e}")
            continue
        row = {"station": st, "lat": float("nan"), "lon": float("nan"),
               "elevation_m": float("nan")}
        if info:
            row.update(info)
        new_rows.append(row)
        if verbose and (i % 20 == 0 or i == len(todo)):
            print(f"  [{i}/{len(todo)}] scraped", flush=True)

    if new_rows:
        cached = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
        cached = cached.drop_duplicates("station", keep="last")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(cache, index=False)

    out = cached[cached["station"].isin(stations)].copy()
    out["has_coords"] = out["lat"].notna() & out["lon"].notna()
    return out.sort_values("station").reset_index(drop=True)


if __name__ == "__main__":
    from emt.insitu.oznet import fetch_manifest
    stations = sorted(fetch_manifest()["station"].unique())
    coords = fetch_station_coords(stations)
    n_ok = int(coords["has_coords"].sum())
    print(f"\n{n_ok}/{len(coords)} stations have coordinates")
    print(coords[~coords["has_coords"]]["station"].tolist(), "= missing")
