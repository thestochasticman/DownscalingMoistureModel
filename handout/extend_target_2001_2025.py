"""Build the OzNet target table over the full 2001-2025 record (new file)."""
from datetime import date
import pandas as pd
from emt.model7.build import build_target
from emt.insitu.base import check_target

OUT = "data/process_target_2001_2025.csv"
t = build_target(date(2001, 1, 1), date(2025, 12, 31), out=OUT)
t["time"] = pd.to_datetime(t["time"])
print(f"\nrows {len(t)}  stations {t.station.nunique()}  "
      f"{t.time.min().date()} -> {t.time.max().date()}")
check_target(t, network="oznet-2001-2025", strict=True)
per = t.groupby(t.time.dt.year).agg(rows=("station","size"), stations=("station","nunique"))
print("\nper year:"); print(per.to_string())
span = t.groupby("station")["time"].agg(["min","max","size"])
span["years"] = (span["max"] - span["min"]).dt.days / 365.25
print(f"\nstations with >=5 years of record: {(span.years >= 5).sum()}/{len(span)}")
print(f"total station-days: {len(t)}  (2006-2010 table has 50,623)")
