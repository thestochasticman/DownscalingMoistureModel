"""Validate the QC detectors against rain: a fault rule must not find weather.

The falsifiable claim behind ``emt.insitu.qc`` is that its ``spike`` rule
detects logger malfunction rather than wetting. If it detects malfunction, its
catch should carry rain no more often than an arbitrary day does. If it is
enriched in rain, it is finding real events and must not be dropped.

This script prints the enrichment ratio for each flag. A ratio near 1.0 clears
the rule; the plain robust-z outlier test this replaced scored 1.71.

    PYTHONPATH=. python handout/validate_qc.py
"""
from __future__ import annotations

import pandas as pd

from emt.insitu import qc

TARGET = "data/process_target_2001_2025.csv"
FORCING = "data/process_forcing_2000_2025.csv"
RAIN_MM, RAIN_DAYS = 5.0, 3


def main() -> None:
    target = pd.read_csv(TARGET, parse_dates=["time"])
    forcing = pd.read_csv(FORCING, parse_dates=["time"]).sort_values(["station", "time"])
    forcing["rain3"] = forcing.groupby("station").daily_rain.transform(
        lambda s: s.rolling(RAIN_DAYS, min_periods=1).sum())

    flagged = qc.flag_target(target, forcing)
    df = flagged.merge(forcing[["station", "time", "rain3"]], on=["station", "time"],
                       how="left").dropna(subset=["rain3"])
    baseline = (df.rain3 >= RAIN_MM).mean()

    print(f"baseline: {baseline*100:.1f}% of station-days carry "
          f">={RAIN_MM:g} mm over {RAIN_DAYS} days\n")
    print(f"{'flag':10s} {'rows':>7s} {'%rain':>7s} {'ratio':>7s}  verdict")
    for f in qc.FLAGS:
        m = df[f]
        if not m.any():
            print(f"{f:10s} {0:7d} {'-':>7s} {'-':>7s}  no detections")
            continue
        share = (df.loc[m, "rain3"] >= RAIN_MM).mean()
        ratio = share / baseline
        verdict = ("CLEAN - no rain enrichment" if ratio < 1.25 else
                   "SUSPECT - fires with rain, likely real weather")
        print(f"{f:10s} {int(m.sum()):7d} {share*100:6.1f}% {ratio:7.2f}  {verdict}")

    print(qc.summarise(flagged).to_string(index=False))


if __name__ == "__main__":
    main()
