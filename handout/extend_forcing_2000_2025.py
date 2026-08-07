"""Extend the SILO forcing store to 2000-2025 (new file)."""
from datetime import date
import pandas as pd
from emt.model7.build import build_forcing, build_climate_statics

stations = sorted(pd.read_csv("data/process_soil_statics.csv")["station"])
f = build_forcing(stations, date(2001, 1, 1), date(2025, 12, 31),
                  out="data/process_forcing_2000_2025.csv")
f["time"] = pd.to_datetime(f["time"])
print(f"\nforcing rows {len(f)}  stations {f.station.nunique()}  "
      f"{f.time.min().date()} -> {f.time.max().date()}")
c = build_climate_statics(f, out="data/process_climate_statics_2000_2025.csv")
print(f"climate statics: {len(c)} stations, aridity "
      f"{c.aridity.min():.2f}-{c.aridity.max():.2f}")
