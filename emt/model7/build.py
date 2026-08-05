"""Build model7's inputs: OzNet target, SILO daily forcing, terrain statics.

model7 does not use the ML training table (no SMIPS, no soil): it needs the
target series, a *continuous* daily forcing record per station (from one year
before the study start, so calibration follows a full spin-up year), and the
per-station terrain statics for the offset variant. All three are national,
public inputs — SILO needs only a registered email, the Copernicus DEM none.

Outputs (``data/``, gitignored like every other table)::

    process_target_2006_2010.csv     station-day root-zone VWC (the target)
    process_forcing_2005_2010.csv    per-station daily rain / PET / VPD
    process_climate_statics.csv      per-station climate normals (rain, PET,
                                     aridity P/PET) derived from the forcing
    process_terrain_statics.csv      per-station TERRAIN_VARS at the point
    process_soil_statics.csv         per-station SLGA SOIL_VARS (needs a TERN
                                     key; skipped with a notice if absent)

Run::  PYTHONPATH=. python -m emt.model7.build
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from emt.build_dataset import DEFAULT_START, DEFAULT_END, site_of
from emt.covariates import TERRAIN_VARS, sample_points, terrain_covariates
from emt.insitu.coordinates import fetch_station_coords
from emt.insitu.oznet import fetch_manifest, load_daily_rootzone
from emt.queries import _period, query_for_station
from PaddockTS.Environmental.SILO.download_silo import download_silo
from PaddockTS.query import Query

TARGET_CSV = "data/process_target_2006_2010.csv"
FORCING_CSV = "data/process_forcing_2005_2010.csv"
STATICS_CSV = "data/process_terrain_statics.csv"
SOIL_CSV = "data/process_soil_statics.csv"
CLIMATE_CSV = "data/process_climate_statics.csv"


def build_target(start: date = DEFAULT_START, end: date = DEFAULT_END,
                 out: str | None = TARGET_CSV) -> pd.DataFrame:
    """OzNet daily root-zone target for every core station with coordinates."""
    man = fetch_manifest()
    coords = fetch_station_coords(sorted(man["station"].unique()))
    coords = coords.dropna(subset=["lat", "lon"]).copy()
    coords["site"] = coords["station"].map(site_of)
    coords = coords.dropna(subset=["site"])
    man = man[man["station"].isin(coords["station"])
              & man["year"].between(start.year, end.year)]
    daily = load_daily_rootzone(manifest=man)
    daily["site"] = daily["station"].map(site_of)
    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily[(daily["time"] >= pd.Timestamp(start))
                  & (daily["time"] <= pd.Timestamp(end))]
    if out:
        daily.to_csv(out, index=False)
    return daily


def build_forcing(stations: list[str], start: date = DEFAULT_START,
                  end: date = DEFAULT_END,
                  out: str | None = FORCING_CSV) -> pd.DataFrame:
    """Continuous daily SILO rain/PET/VPD per station over [start-1yr, end]."""
    coords = fetch_station_coords(stations).set_index("station")
    frames = []
    for stn in stations:
        q = query_for_station(stn, float(coords.loc[stn, "lat"]),
                              float(coords.loc[stn, "lon"]),
                              date(start.year - 1, 1, 1), end)
        try:
            silo = download_silo(q)
        except Exception as e:                              # noqa: BLE001
            print(f"  SILO {stn}: FAIL {type(e).__name__}: {e}", flush=True)
            continue
        silo = silo.rename(columns={silo.columns[0]: "time"})
        keep = silo[["time", "daily_rain", "et_morton_potential", "vp_deficit"]].copy()
        keep["station"] = stn
        frames.append(keep)
    forcing = pd.concat(frames, ignore_index=True)
    if out:
        forcing.to_csv(out, index=False)
    return forcing


def build_terrain_statics(stations: list[str], start: date = DEFAULT_START,
                          end: date = DEFAULT_END,
                          out: str | None = STATICS_CSV) -> pd.DataFrame:
    """Sample TERRAIN_VARS at each station point (30 m Copernicus DEM).

    A station whose 1.5 km window degenerates at a DEM tile boundary (Y9 does)
    is retried with a larger buffer under a distinct stub — PaddockTS's registry
    pins each stub to one bbox, so the retry must not reuse the original stub.
    """
    coords = fetch_station_coords(stations).set_index("station")
    rows = []
    for stn in stations:
        lat, lon = float(coords.loc[stn, "lat"]), float(coords.loc[stn, "lon"])
        for buf in (1.5, 2.0, 3.0):
            try:
                stub = (f"oznet_{stn}_{_period(start, end)}" if buf == 1.5 else
                        f"oznet_{stn}_b{buf:g}_{_period(start, end)}")
                q = Query.from_lat_lon(lat=lat, lon=lon, buffer_km=buf,
                                       start=start, end=end, stub=stub)
                terr = terrain_covariates(q)
                row = {"station": stn}
                for v in TERRAIN_VARS:
                    row[v] = float(sample_points(terr[v], lon, lat).values)
                rows.append(row)
                break
            except Exception as e:                          # noqa: BLE001
                print(f"  terrain {stn} (buffer {buf}): "
                      f"{type(e).__name__}: {e}", flush=True)
    statics = pd.DataFrame(rows)
    if out:
        statics.to_csv(out, index=False)
    return statics


def build_climate_statics(forcing: pd.DataFrame,
                          out: str | None = CLIMATE_CSV) -> pd.DataFrame:
    """Per-station climate normals from the forcing store (no extra fetch).

    Mean annual rain and Morton PET (mm/yr) and their ratio (aridity, P/PET)
    over the forcing period. The aridity normal is model8's climate static:
    the level channel that transfers to unseen blocks (see the handout's
    blocked-validation page).
    """
    g = forcing.groupby("station").agg(rain=("daily_rain", "mean"),
                                       pet=("et_morton_potential", "mean"))
    stats = pd.DataFrame({"rain_mean": g["rain"] * 365.25,
                          "pet_mean": g["pet"] * 365.25})
    stats["aridity"] = stats["rain_mean"] / stats["pet_mean"]
    stats = stats.reset_index()
    if out:
        stats.to_csv(out, index=False)
    return stats


def build_soil_statics(stations: list[str], start: date = DEFAULT_START,
                       end: date = DEFAULT_END,
                       out: str | None = SOIL_CSV) -> pd.DataFrame:
    """Sample SLGA SOIL_VARS at each station point (needs a TERN API key)."""
    from emt.slga import SOIL_VARS, soil_covariates
    coords = fetch_station_coords(stations).set_index("station")
    rows = []
    for stn in stations:
        lat, lon = float(coords.loc[stn, "lat"]), float(coords.loc[stn, "lon"])
        q = query_for_station(stn, lat, lon, start, end)
        try:
            soil = soil_covariates(q)
            row = {"station": stn}
            for v in SOIL_VARS:
                row[v] = float(sample_points(soil[v], lon, lat).values)
            rows.append(row)
        except Exception as e:                              # noqa: BLE001
            print(f"  soil {stn}: FAIL {type(e).__name__}: {e}", flush=True)
    statics = pd.DataFrame(rows)
    if out and len(statics):
        statics.to_csv(out, index=False)
    return statics


def build(start: date = DEFAULT_START, end: date = DEFAULT_END) -> None:
    print("=== target (OzNet) ===", flush=True)
    daily = build_target(start, end)
    stations = sorted(daily["station"].unique())
    print(f"  {len(daily)} rows, {len(stations)} stations", flush=True)
    print("=== forcing (SILO) ===", flush=True)
    forcing = build_forcing(stations, start, end)
    print(f"  {len(forcing)} rows, {forcing['station'].nunique()} stations", flush=True)
    print("=== climate statics (from the forcing) ===", flush=True)
    clim = build_climate_statics(forcing)
    print(f"  {len(clim)} stations", flush=True)
    print("=== terrain statics (Copernicus DEM) ===", flush=True)
    statics = build_terrain_statics(stations, start, end)
    print(f"  {len(statics)} stations", flush=True)
    print("=== soil statics (SLGA) ===", flush=True)
    from PaddockTS.config import config
    if not config.tern_api_key:
        print("  no TERN API key in ~/.config/PaddockTS.json -- skipped "
              "(model7 then runs without the soil variants)", flush=True)
    else:
        soil = build_soil_statics(stations, start, end)
        print(f"  {len(soil)} stations", flush=True)


if __name__ == "__main__":
    build()
