"""Antecedent-meteorology features from SILO (last week / month / year).

Soil moisture is driven by how much water has recently arrived versus left.
These trailing-window features summarise that history at each station-day:

    rain_7, rain_30, rain_365     trailing rainfall sums (mm)
    ppet_30, ppet_365             trailing water balance, P - PET (mm)
    vpd_30                        trailing mean vapour-pressure deficit
    rain_365_anom                 rain_365 minus the pixel's mean 365 d rain
                                  (a simple drought / wet-year index)

All are *dynamic* (a per-pixel time series, not a static site value) and come
from SILO, a national ~5 km daily grid, so they are computable at every 30 m
pixel at inference and cannot memorise station identity -- the same leakage test
the other features pass.

SILO is fetched via PaddockTS (:func:`download_silo`), extended one year before
the study start so the 365-day window is complete at every training date.
"""
from __future__ import annotations

from datetime import date
import glob

import pandas as pd

from emt.queries import query_for_station
from PaddockTS.Environmental.SILO.download_silo import download_silo, get_filename

ANTECEDENT_VARS = ["rain_7", "rain_30", "rain_365",
                   "ppet_30", "ppet_365", "vpd_30", "rain_365_anom"]

# SILO columns used (rainfall, Morton potential ET, vapour-pressure deficit).
_RAIN, _PET, _VPD = "daily_rain", "et_morton_potential", "vp_deficit"


def _trailing(silo: pd.DataFrame) -> pd.DataFrame:
    """Trailing-window features from a station's daily SILO frame."""
    s = silo.copy()
    s = s.rename(columns={s.columns[0]: "time"}) if "time" not in s.columns else s
    s["time"] = pd.to_datetime(s["time"])
    s = s.set_index("time").sort_index()
    rain, ppet, vpd = s[_RAIN], s[_RAIN] - s[_PET], s[_VPD]
    out = pd.DataFrame({
        "rain_7":   rain.rolling(7,   min_periods=4).sum(),
        "rain_30":  rain.rolling(30,  min_periods=15).sum(),
        "rain_365": rain.rolling(365, min_periods=180).sum(),
        "ppet_30":  ppet.rolling(30,  min_periods=15).sum(),
        "ppet_365": ppet.rolling(365, min_periods=180).sum(),
        "vpd_30":   vpd.rolling(30,   min_periods=15).mean(),
    })
    out["rain_365_anom"] = out["rain_365"] - out["rain_365"].mean()
    return out.reset_index()


def _station_silo(stn: str, lat: float, lon: float,
                  start: date, end: date) -> pd.DataFrame | None:
    """Fetch (cached) SILO for one station over [start-1yr, end]."""
    q = query_for_station(stn, lat, lon, date(start.year - 1, 1, 1), end)
    # reuse an existing cache file if present (either extended or study-period).
    for pat in (get_filename(q),
                f"{q.tmp_dir}/Environmental/oznet_{stn}_*_silo.csv"):
        hit = glob.glob(pat)
        if hit:
            return pd.read_csv(sorted(hit)[-1])
    try:
        return download_silo(q)
    except Exception as e:                       # noqa: BLE001
        print(f"  SILO {stn}: FAIL {type(e).__name__}: {e}", flush=True)
        return None


def add_antecedent(table: pd.DataFrame, coords: pd.DataFrame,
                   start: date, end: date, verbose: bool = True) -> pd.DataFrame:
    """Attach ``ANTECEDENT_VARS`` to ``table`` (one row per station-day)."""
    coords = coords.set_index("station") if "station" in coords.columns else coords
    tab = table.copy()
    tab["time"] = pd.to_datetime(tab["time"])
    frames = []
    for stn in tab["station"].unique():
        silo = _station_silo(stn, float(coords.loc[stn, "lat"]),
                             float(coords.loc[stn, "lon"]), start, end)
        if silo is None:
            continue
        f = _trailing(silo)
        f["station"] = stn
        frames.append(f)
        if verbose:
            print(f"  antecedent {stn}: {len(f)} days", flush=True)
    if not frames:
        return tab
    ante = pd.concat(frames, ignore_index=True)
    return tab.merge(ante, on=["station", "time"], how="left")
