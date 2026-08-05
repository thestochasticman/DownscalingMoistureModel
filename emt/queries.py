"""Build PaddockTS ``Query`` objects for the EMT study areas.

EMT does not define its own AOI type -- it uses PaddockTS's :class:`Query`
(which carries bbox/dates and drives every PaddockTS downloader and its cache).
These helpers just construct Queries for the things EMT cares about: small
windows around individual OzNet stations (for training-point extraction) and
the three Murrumbidgee focus catchments (for full-field downscaling).
"""
from __future__ import annotations

from datetime import date

from PaddockTS.query import Query

# Approximate bounding boxes of the three clustered focus catchments
# (the scattered regional M1-M7 sites are handled per-station instead).
FOCUS_AREAS = {
    "yanco":   [145.8, -35.15, 146.45, -34.60],
    "kyeamba": [147.30, -35.52, 147.62, -35.10],
    "adelong": [148.05, -35.52, 148.15, -35.35],
}


def _period(start: date, end: date) -> str:
    """Compact date-range tag, e.g. ``20200101_20201231`` (keeps stubs unique per period)."""
    return f"{start:%Y%m%d}_{end:%Y%m%d}"


DEFAULT_BUFFER_KM = 1.5


def query_for_station(station: str, lat: float, lon: float,
                      start: date, end: date,
                      buffer_km: float = DEFAULT_BUFFER_KM) -> Query:
    """A small square Query centred on one OzNet station (for point extraction).

    The stub embeds the date range because PaddockTS's registry requires a stub
    to map to a single (bbox, time) -- so the same station over different study
    periods gets distinct, human-readable cache entries. For the same reason a
    non-default ``buffer_km`` gets its own ``_b<km>`` stub: a widened retry (see
    :func:`emt.features.station_terrain`) must not reuse the original stub,
    which is already pinned to the narrower bbox.
    """
    tag = (f"oznet_{station}_{_period(start, end)}"
           if buffer_km == DEFAULT_BUFFER_KM else
           f"oznet_{station}_b{buffer_km:g}_{_period(start, end)}")
    return Query.from_lat_lon(lat=lat, lon=lon, buffer_km=buffer_km,
                              start=start, end=end, stub=tag)


def query_for_focus_area(name: str, start: date, end: date) -> Query:
    """A Query covering one clustered focus catchment (yanco / kyeamba / adelong)."""
    key = name.lower()
    if key not in FOCUS_AREAS:
        raise ValueError(f"unknown focus area {name!r}; choose from {list(FOCUS_AREAS)}")
    return Query(bbox=FOCUS_AREAS[key], start=start, end=end,
                 stub=f"focus_{key}_{_period(start, end)}")


def queries_for_stations(coords, start: date, end: date,
                         buffer_km: float = 1.5) -> dict[str, Query]:
    """One per-station Query for every row in a station-coords table.

    Args:
        coords: DataFrame with ``station``, ``lat``, ``lon`` columns
            (e.g. ``data/oznet/station_coords.csv``, filtered to has-coords).
    """
    out = {}
    for r in coords.itertuples(index=False):
        out[r.station] = query_for_station(r.station, float(r.lat), float(r.lon),
                                            start, end, buffer_km=buffer_km)
    return out
