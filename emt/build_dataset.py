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
import sys

import pandas as pd

from emt.insitu.oznet import fetch_manifest, load_daily_rootzone
from emt.insitu.coordinates import COORDS_CACHE
from emt.features import (build_training_table, add_smips_climatology,
                          add_soil_covariates, SMIPS_COL, CLIM_VARS)
from emt.covariates import TERRAIN_VARS
from emt.slga import SOIL_VARS

# OzNet station-prefix -> site (the leave-region-out grouping).
SITE_OF_PREFIX = {"A": "ADELONG", "K": "KYEAMBA", "Y": "YANCO", "M": "MURRUMBIDGEE"}

DEFAULT_OUT = "data/train_catchment_plus_m_2006_2010.csv"
DEFAULT_START, DEFAULT_END = date(2006, 1, 1), date(2010, 12, 31)

FINAL_COLS = (["site", "station", "time", "lat", "lon", "sm_rootzone_pct", SMIPS_COL]
              + list(TERRAIN_VARS) + ["doy_sin", "doy_cos"]
              + list(CLIM_VARS) + list(SOIL_VARS))


def site_of(station: str) -> str | None:
    return SITE_OF_PREFIX.get(str(station)[:1].upper())


def build(stations: list[str] | None = None, start: date = DEFAULT_START,
          end: date = DEFAULT_END, out: str | None = DEFAULT_OUT,
          verbose: bool = True) -> pd.DataFrame:
    """Build (and optionally cache) the full training table.

    Args:
        stations: station ids to include; ``None`` = every core OzNet station
            with resolved coordinates (Y*/K*/A*/M1-M7).
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

    tab = build_training_table(coords, daily, start, end, verbose=verbose)
    tab = add_smips_climatology(tab)                       # SMIPS level/anomaly
    tab = add_soil_covariates(tab, coords, start, end)     # SLGA soil
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
