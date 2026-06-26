"""Stage 4 -- build the model training table.

For every OzNet station-day we have a calibrated root-zone target, attach the
predictors the downscaling model learns from:

    target    sm_rootzone_pct            OzNet 0-90 cm volumetric % (emt.insitu)
    coarse    smips_totalbucket          SMIPS TotalBucket (mm) at the station pixel
    terrain   elevation, slope,          30 m DEM derivatives at the station
              northness, eastness,
              twi, hli, accumulation
    temporal  doy_sin, doy_cos           day-of-year seasonality
    id/space  station, site, lat, lon    (station id is the spatial CV group)

SMIPS comes from the EMT WCS loader and terrain from PaddockTS -- both driven by
the per-station PaddockTS :class:`Query` (so everything is cached on disk).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from PaddockTS.query import Query

from emt.queries import query_for_station
from emt.smips import download_smips
from emt.covariates import terrain_covariates, sample_points, TERRAIN_VARS

SMIPS_COL = "smips_totalbucket"

# Stations within this lon/lat span share one SMIPS cube ("cluster fetch": one
# WCS request per day for the whole cluster instead of one per station). Wider
# groups (e.g. the scattered regional M-sites) fall back to per-station cubes.
MAX_CLUSTER_DEG = 0.7


def add_temporal_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Add cyclic day-of-year features (``doy_sin``, ``doy_cos``)."""
    doy = pd.to_datetime(df[time_col]).dt.dayofyear
    df = df.copy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def station_features(station: str, site: str, lat: float, lon: float,
                     oznet_station: pd.DataFrame, start: date, end: date,
                     buffer_km: float = 1.5, smips_var: str = "totalbucket",
                     smips_series: pd.Series | None = None) -> pd.DataFrame:
    """Feature rows for one station over ``[start, end]``.

    Args:
        oznet_station: Daily root-zone rows for THIS station (columns
            ``time``, ``sm_rootzone_pct``), e.g. a slice of
            :func:`emt.insitu.oznet.load_daily_rootzone`.
        smips_series: Pre-sampled SMIPS time series at this station (index =
            dates). If ``None``, a per-station SMIPS cube is downloaded. Pass a
            cluster-sampled series to avoid redundant downloads.

    Returns:
        DataFrame of station-days with target + SMIPS + terrain columns. Days
        where SMIPS is missing are dropped.
    """
    df = oznet_station.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    df = df[(df["time"] >= pd.Timestamp(start)) & (df["time"] <= pd.Timestamp(end))]
    if df.empty:
        return df

    q = query_for_station(station, lat, lon, start, end, buffer_km=buffer_km)

    # Static terrain covariates at the station point (one sample, broadcast).
    terr = sample_points(terrain_covariates(q), lon, lat)
    for v in TERRAIN_VARS:
        df[v] = float(terr[v].values)

    # SMIPS time series at the station pixel (cluster-supplied or per-station).
    if smips_series is None:
        cube = download_smips(q, var=smips_var)
        smips_series = sample_points(cube, lon, lat).to_pandas()
    smips_series = smips_series.copy()
    smips_series.index = pd.to_datetime(smips_series.index).normalize()
    df[SMIPS_COL] = df["time"].map(smips_series)

    # oznet_station already carries 'site' and 'station'; just add coordinates.
    df["site"] = site if site else df.get("site")
    df["lat"] = lat
    df["lon"] = lon
    return df.dropna(subset=[SMIPS_COL])


def _partition_clusters(coords: pd.DataFrame) -> list[pd.DataFrame]:
    """Split stations into SMIPS fetch-groups.

    Stations of one ``site`` whose envelope is tighter than ``MAX_CLUSTER_DEG``
    share a cube; otherwise each station is its own group.
    """
    groups: list[pd.DataFrame] = []
    for _site, g in coords.dropna(subset=["lat", "lon"]).groupby("site"):
        span_ok = ((g["lon"].max() - g["lon"].min()) <= MAX_CLUSTER_DEG and
                   (g["lat"].max() - g["lat"].min()) <= MAX_CLUSTER_DEG)
        if span_ok and len(g) > 1:
            groups.append(g)
        else:
            groups.extend(g.iloc[[i]] for i in range(len(g)))
    return groups


def _cluster_smips(group: pd.DataFrame, start: date, end: date,
                   smips_var: str, buffer_km: float = 2.0) -> dict[str, pd.Series]:
    """Download one SMIPS cube over a station cluster; sample each station.

    Returns ``{station: smips_series}``.
    """
    lon_b = buffer_km / (111.0 * 0.82)   # ~cos(35 deg)
    lat_b = buffer_km / 111.0
    bbox = [float(group["lon"].min() - lon_b), float(group["lat"].min() - lat_b),
            float(group["lon"].max() + lon_b), float(group["lat"].max() + lat_b)]
    tag = "-".join(sorted(group["station"])[:3]) + f"_{len(group)}"
    q = Query(bbox=bbox, start=start, end=end,
              stub=f"smipscl_{tag}_{start:%Y%m%d}_{end:%Y%m%d}")
    cube = download_smips(q, var=smips_var)
    return {r.station: sample_points(cube, float(r.lon), float(r.lat)).to_pandas()
            for r in group.itertuples(index=False)}


def build_training_table(coords: pd.DataFrame, oznet_daily: pd.DataFrame,
                         start: date, end: date, buffer_km: float = 1.5,
                         smips_var: str = "totalbucket", verbose: bool = True) -> pd.DataFrame:
    """Assemble the full training table across all stations with coordinates.

    Args:
        coords: Station coords (``station``, ``site``, ``lat``, ``lon``);
            rows without lat/lon are skipped.
        oznet_daily: Combined daily root-zone table from
            :func:`emt.insitu.oznet.load_daily_rootzone`.
        start, end: Study period.

    Returns:
        Long-format training table; one row per station-day with target,
        SMIPS, terrain and temporal features.
    """
    frames = []
    clusters = _partition_clusters(coords)
    n_st = sum(len(g) for g in clusters)
    done = 0
    for gi, group in enumerate(clusters, 1):
        # One SMIPS cube per cluster (or per station for wide groups).
        try:
            smips_by_station = _cluster_smips(group, start, end, smips_var)
        except Exception as e:
            if verbose:
                print(f"  cluster {gi}/{len(clusters)} ({list(group.station)}): "
                      f"SMIPS FAILED ({type(e).__name__}: {e})")
            smips_by_station = {}

        for r in group.itertuples(index=False):
            done += 1
            site = getattr(r, "site", "")
            sub = oznet_daily[oznet_daily["station"] == r.station]
            if sub.empty:
                continue
            try:
                f = station_features(r.station, site, float(r.lat), float(r.lon),
                                     sub, start, end, buffer_km=buffer_km,
                                     smips_var=smips_var,
                                     smips_series=smips_by_station.get(r.station))
            except Exception as e:
                if verbose:
                    print(f"  [{done}/{n_st}] {r.station}: SKIP ({type(e).__name__}: {e})")
                continue
            if not f.empty:
                frames.append(f)
            if verbose:
                print(f"  [{done}/{n_st}] {r.station}: {len(f)} rows", flush=True)

    if not frames:
        return pd.DataFrame()

    table = pd.concat(frames, ignore_index=True)
    table = add_temporal_features(table, "time")
    cols = (["site", "station", "time", "lat", "lon", "sm_rootzone_pct", SMIPS_COL]
            + list(TERRAIN_VARS) + ["doy_sin", "doy_cos"])
    return table[[c for c in cols if c in table.columns]]


if __name__ == "__main__":
    import pandas as pd
    from emt.insitu.oznet import fetch_manifest, load_daily_rootzone
    from emt.insitu.coordinates import COORDS_CACHE

    coords = pd.read_csv(COORDS_CACHE)
    coords = coords.merge(fetch_manifest()[["site", "station"]].drop_duplicates(), on="station")
    sub = coords[coords["station"].isin(["K5", "K6", "K7"])]

    man = fetch_manifest()
    man = man[man["station"].isin(sub["station"]) & (man["year"] == 2020)]
    daily = load_daily_rootzone(manifest=man, verbose=False)

    table = build_training_table(sub, daily, date(2020, 6, 1), date(2020, 7, 31))
    print(table.head())
    print("shape:", table.shape)
