"""Build the model training table from OzNet in-situ + national covariates.

The canonical training set is the OzNet core stations with resolved
coordinates: the Yanco (``Y*``), Kyeamba (``K*``) and Adelong (``A*``) clusters
**plus the scattered regional Murrumbidgee sites** (``M1``-``M7``). The M-sites
span the whole catchment (west to the semi-arid plains, south to the alpine
fringe) and were added to broaden the level information the model sees; see the
handout "Extending coverage" section for the leave-site-out effect.

Everything attached here — SMIPS, terrain, SMIPS pixel-climatology and SLGA soil
— is a *national* Australian covariate, so this same builder produces a training
row for any Australian station once its coordinates are known. Extending beyond
OzNet (e.g. a national ISMN/CosmOZ set) only needs more ``(station, lat, lon,
daily root-zone)`` inputs; the covariate side is unchanged.

Run::  PYTHONPATH=. python -m emt.build_dataset            # full 36-station table
       PYTHONPATH=. python -m emt.build_dataset K5 K6 M7   # a subset
"""
from __future__ import annotations

from datetime import date
import re
import sys

import pandas as pd

from emt.insitu.oznet import fetch_manifest, load_daily_rootzone
from emt.insitu.coordinates import COORDS_CACHE
from emt.features import (build_training_table, add_soil_covariates, SMIPS_COL,
                          SMIPS_WORKERS)
from emt.covariates import TERRAIN_VARS
from emt.slga import SOIL_VARS

# OzNet station-prefix -> site (the leave-region-out grouping).
SITE_OF_PREFIX = {"A": "ADELONG", "K": "KYEAMBA", "Y": "YANCO", "M": "MURRUMBIDGEE"}

# Only the CORE PROFILE stations belong in the 0-90 cm root-zone target: a
# letter followed by digits (Y1-Y13, K1-K14, A1-A5, M1-M7).
#
# The archive also carries the Yanco SMAP focus grid -- 24 stations named YA*
# and YB*, installed 2009 on a 3 km/9 km lattice. Those are SURFACE ONLY
# (0-5 cm) and must never become root-zone training rows: a shallow sensor
# dries faster than a profile mean, so the model would be fitted against a
# reference that is systematically drier than what it predicts. Y3, the one
# core station whose target is a single layer rather than three, shows the
# signature -- blocked bias +4.27 %, the model reading too wet.
#
# Until now they were excluded only INCIDENTALLY: no coordinate page exists
# for them, so build_target dropped them for want of lat/lon. That is not a
# decision, it is an accident of the scraper, and it would reverse the moment
# a coordinate source appeared (they are published in the SMAPEx literature).
# Note "YA1"[:1] == "Y", so the prefix map alone would file them under YANCO.
CORE_STATION_RE = re.compile(r"^[YKAM]\d+$")


def is_core_profile_station(station: str) -> bool:
    """True for the 0-90 cm profile stations, False for the surface grid."""
    return bool(CORE_STATION_RE.match(str(station).upper()))

DEFAULT_OUT = "data/train_catchment_plus_m_2006_2010.csv"
DEFAULT_START, DEFAULT_END = date(2006, 1, 1), date(2010, 12, 31)

# SMIPS climatology (CLIM_VARS) is deliberately NOT baked in: it is an
# as-of-date expanding statistic, so it is always recomputed fresh from the
# raw SMIPS column by each model's ``ensure_features`` (baking it once risked
# serving a stale/leaky version). Soil is static and safe to bake.
FINAL_COLS = (["site", "station", "time", "lat", "lon", "sm_rootzone_pct", SMIPS_COL]
              + list(TERRAIN_VARS) + ["doy_sin", "doy_cos"] + list(SOIL_VARS))


def site_of(station: str) -> str | None:
    """Site label for a core profile station; ``None`` for anything else.

    Returning None for the surface grid is what keeps it out of the target --
    every builder drops rows with a null site.
    """
    if not is_core_profile_station(station):
        return None
    return SITE_OF_PREFIX.get(str(station)[:1].upper())


def build(stations: list[str] | None = None, start: date = DEFAULT_START,
          end: date = DEFAULT_END, out: str | None = DEFAULT_OUT,
          verbose: bool = True, workers: int = SMIPS_WORKERS) -> pd.DataFrame:
    """Build (and optionally cache) the full training table.

    Args:
        stations: station ids to include; ``None`` = every core OzNet station
            with resolved coordinates (Y*/K*/A*/M1-M7).
        workers: Concurrent per-day SMIPS WCS requests -- the build's dominant
            cost (see :data:`emt.features.SMIPS_WORKERS`).
    Returns the long-format table (one row per station-day).
    """
    coords = pd.read_csv(COORDS_CACHE).dropna(subset=["lat", "lon"]).copy()
    coords["site"] = coords["station"].map(site_of)
    coords = coords.dropna(subset=["site"])
    if stations is not None:
        coords = coords[coords["station"].isin(stations)]
    if verbose:
        print(f"stations: {coords['station'].nunique()} "
              f"({coords.groupby('site')['station'].nunique().to_dict()})", flush=True)

    man = fetch_manifest()
    man = man[man["station"].isin(coords["station"])
              & man["year"].between(start.year, end.year)]
    daily = load_daily_rootzone(manifest=man, verbose=verbose)
    daily["site"] = daily["station"].map(site_of)
    if verbose:
        print(f"daily root-zone rows: {len(daily)} "
              f"({daily['station'].nunique()} stations)", flush=True)

    tab = build_training_table(coords, daily, start, end, verbose=verbose,
                               workers=workers)
    tab = add_soil_covariates(tab, coords, start, end)     # SLGA soil (static)
    tab = tab[[c for c in FINAL_COLS if c in tab.columns]]

    if out:
        tab.to_csv(out, index=False)
        if verbose:
            print(f"\nsaved {out}: {tab['station'].nunique()} stations, "
                  f"{len(tab)} rows", flush=True)
            print(tab.groupby("site")["station"].nunique().to_string(), flush=True)
    return tab


if __name__ == "__main__":
    args = sys.argv[1:]
    build(stations=args or None)
