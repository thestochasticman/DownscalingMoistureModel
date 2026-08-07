"""Train and validate model8 on the extended 2001-2025 OzNet record.

Every result this project has published rests on 2006-2010: 151 station-years,
one anomalous wet year, and a single hydrological era. The extended record
carries **442 station-years** spanning the Millennium Drought, the 2010-12 La
Nina, the 2017-19 drought and the 2020-22 La Nina -- the regime variety needed
to say whether the process model transfers in *time* as well as space.

Three input swaps against ``run_blocked_cv.py``:

* target  ``process_target_2001_2025_qc.csv``  -- QC'd by :mod:`emt.insitu.qc`;
  2,979 range violations removed (435 of them exactly 65535, a logger
  sentinel), concentrated in 2021-2024. Spike and jump flags are reported but
  NOT dropped -- see that module for why.
* forcing ``process_forcing_2000_2025.csv``
* climate ``process_climate_statics_2000_2025.csv``

That third swap is worth naming: the aridity static is computed over the same
window the model is trained on, so the training/inference window mismatch
tracked on the ``aridity-reference-window`` branch does not arise here. It is
fixed by construction rather than by patch.

**Reading the year folds.** Station availability is not constant: 34 stations
clear 200 days in 2006, 10 in 2024. A year fold late in the record is a
smaller and differently-composed sample, so a skill change there may be the
network shrinking rather than the model failing. Per-year station counts are
printed beside per-year skill for exactly this reason, and no year-on-year
comparison should be read without them.

Run::  PYTHONPATH=. python handout/run_extended_cv.py [block|year|blockyear|station]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from emt.evaluation import metrics
from emt.model7 import model as m7
from emt.model7.model import BucketEstimator
from emt.model8 import model as m8

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
TARGET = "sm_rootzone_pct"

TARGET_CSV = DATA / "process_target_2001_2025_qc.csv"
FORCING_CSV = DATA / "process_forcing_2000_2025.csv"
CLIMATE_CSV = DATA / "process_climate_statics_2000_2025.csv"

MIN_FOLD_OBS = 200          # skip folds too small to score meaningfully


def load_table() -> pd.DataFrame:
    """QC'd target restricted to stations that have forcing and statics."""
    forcing = m7.load_forcing(FORCING_CSV, reload=True)
    statics = m8.load_statics(climate_csv=CLIMATE_CSV, climate=True)
    tab = pd.read_csv(TARGET_CSV, parse_dates=["time"])
    keep = set(statics.index) & set(forcing.stations)
    dropped = sorted(set(tab.station) - keep)
    if dropped:
        n = len(tab[tab.station.isin(dropped)])
        print(f"  excluded {dropped} -- no forcing/statics ({n:,} rows)")
    tab = tab[tab.station.isin(keep)].reset_index(drop=True)
    tab["block"] = tab["station"].map(m8.block_of)
    tab["year"] = tab["time"].dt.year
    tab["blockyear"] = tab["block"] + "|" + tab["year"].astype(str)
    return tab, statics


def year_station_counts(tab: pd.DataFrame) -> pd.Series:
    """Stations clearing 200 observations, per year -- the like-for-like caveat."""
    c = tab.groupby(["year", "station"]).size()
    return (c >= MIN_FOLD_OBS).groupby(level=0).sum()


def run_cv(tab: pd.DataFrame, statics: pd.DataFrame, holdout: str) -> pd.DataFrame:
    """Hold out each group of ``holdout`` in turn and predict it."""
    out = tab[["station", "block", "year", "time", TARGET]].copy()
    out["pred"] = np.nan
    groups = sorted(tab[holdout].unique())
    for grp in groups:
        te = (tab[holdout] == grp).values
        if te.sum() < MIN_FOLD_OBS:
            print(f"    skip {grp}: only {te.sum()} obs", flush=True)
            continue
        if holdout == "blockyear":
            blk, yr = grp.split("|")
            tr = tab.loc[(tab["block"] != blk) & (tab["year"] != int(yr))]
        else:
            tr = tab.loc[~te]
        est = BucketEstimator(static=statics, capacity=m8.awc_capacity(),
                              weight_fn=m8.stratified_weights)
        est.fit(tr[m8.FEATURES], tr[TARGET])
        out.loc[te, "pred"] = est.predict(tab.loc[te, m8.FEATURES])
        print(f"    held out {grp} (trained on {tr['station'].nunique()} "
              f"stations, {len(tr):,} rows)", flush=True)
    return out.dropna(subset=["pred"])


def summarise(name: str, out: pd.DataFrame, counts: pd.Series | None = None) -> None:
    p = metrics(out[TARGET], out["pred"])
    by = lambda k: out.groupby(k).apply(                       # noqa: E731
        lambda g: pd.Series(metrics(g[TARGET], g["pred"])), include_groups=False)
    stn, blk = by("station"), by("block")
    print(f"\n== {name}  ({len(out):,} predictions, "
          f"{out.station.nunique()} stations, {out.year.nunique()} years)")
    print(f"  pooled       NSE {p['nse']:+.3f}  r {p['r']:.2f}  "
          f"ubRMSE {p['ubrmse']:.2f}  bias {p['bias']:+.2f}")
    print(f"  block-level  median NSE {blk['nse'].median():+.3f}  "
          f"NSE>0 {int((blk['nse'] > 0).sum())}/{len(blk)}")
    print(f"  station      median NSE {stn['nse'].median():+.3f}  "
          f"NSE>0 {int((stn['nse'] > 0).sum())}/{len(stn)}")
    yr = by("year")
    print(f"\n  per year (n_stations is the availability caveat -- "
          f"folds are NOT like-for-like):")
    print(f"    {'year':>5s} {'NSE':>7s} {'r':>5s} {'bias':>6s} "
          f"{'n_obs':>7s} {'n_stn':>6s}")
    for y, r in yr.iterrows():
        ns = f"{counts.get(y, 0):d}" if counts is not None else "-"
        print(f"    {y:>5} {r['nse']:+7.3f} {r['r']:5.2f} {r['bias']:+6.2f} "
              f"{int(r['n']):7d} {ns:>6s}")


def main() -> None:
    tab, statics = load_table()
    counts = year_station_counts(tab)
    print(f"extended record: {len(tab):,} rows, {tab.station.nunique()} stations, "
          f"{tab.year.min()}-{tab.year.max()}")
    print(f"legacy window for reference: "
          f"{len(tab[tab.year.between(2006, 2010)]):,} rows\n")

    args = sys.argv[1:] or ["block"]
    years = [a for a in args if a.startswith("--years=")]
    if years:
        lo, hi = (int(v) for v in years[0].split("=")[1].split("-"))
        tab = tab[tab.year.between(lo, hi)].reset_index(drop=True)
        counts = year_station_counts(tab)
        print(f"restricted to {lo}-{hi}: {len(tab):,} rows, "
              f"{tab.station.nunique()} stations\n")
        args = [a for a in args if not a.startswith("--years=")] or ["block"]

    for holdout in args:
        print(f"--- {holdout} folds ---", flush=True)
        out = run_cv(tab, statics, holdout)
        out.to_csv(DATA / f"model8_extended_{holdout}cv_predictions.csv", index=False)
        summarise(f"model8 extended / {holdout}", out, counts)


if __name__ == "__main__":
    main()
