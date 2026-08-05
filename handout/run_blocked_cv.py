"""Blocked (leave-one-block-out) cross-validation for model6 and model8.

The leave-one-STATION-out harness holds out one station while its cluster
neighbours -- stations sharing the same ~5 km SILO forcing cells and SLGA map
units -- stay in training, so it measures interpolation next to an instrumented
site. This script measures spatial *transfer* instead: it holds out whole
spatially independent **blocks** (YANCO, KYEAMBA, ADELONG, and each regional
M-site on its own = 9 folds) and scores the held-out block.

It also runs the two treatments evaluated on top (training-side only --
validation is always unweighted):

  * **stratified weights**: aridity (P/PET) tertiles x block cells, each
    stratum given equal total weight, split equally over its blocks, then over
    samples; tempered by **0.5 to guard tiny cells (A4 has 89 obs).
  * **aridity static** (model8 only): per-station mean P/PET joins the soil +
    terrain statics in the ridge offset stage (data/process_climate_statics.csv,
    derived from the SILO forcing).

Writes out-of-fold predictions to data/model{6,8}_blockcv*_predictions.csv and
prints pooled / block-level / station-level summaries. Results and discussion:
handout/modules/blocked_validation.md.

Run from repo root::  PYTHONPATH=. python handout/run_blocked_cv.py [m8|m8w|m8a|m8aw|m6|m6w ...]

model6 needs its feature table; build it once with
``emt.model6.model.ensure_features`` over the training table (cached here to
data/model6_features_2006_2010.csv -- the first build fetches the 2005 SMIPS
climatology seed and per-station SILO).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from emt.evaluation import metrics
from emt.model6 import model as m6
from emt.model7.model import BucketEstimator
from emt.model8 import model as m8

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
TARGET = "sm_rootzone_pct"
TEMPER = 0.5           # w <- w**TEMPER: between flat (0) and full equalisation (1)


def block_of(stn: str) -> str:
    """Spatially independent location: cluster prefix, or the M-site itself."""
    return {"Y": "YANCO", "K": "KYEAMBA", "A": "ADELONG"}.get(stn[0], stn)


def strata_from_forcing() -> pd.Series:
    """Aridity (P/PET) tertile per station, from the SILO forcing store."""
    f = pd.read_csv(DATA / "process_forcing_2005_2010.csv")
    g = f.groupby("station").agg(rain=("daily_rain", "mean"),
                                 pet=("et_morton_potential", "mean"))
    return pd.qcut(g["rain"] / g["pet"], 3, labels=["dry", "mid", "wet"])


def make_weights(sub: pd.DataFrame, temper: float = TEMPER) -> np.ndarray:
    """Hierarchical sample weights: stratum -> block -> sample, tempered.

    ideal w(sample) = 1 / (n_strata * n_blocks_in_stratum * n_samples_in_cell),
    cell = (stratum, block): each stratum carries equal total weight, split
    equally over the blocks inside it, then over that cell's samples -- so ten
    Yanco stations no longer outvote three scattered dry M-sites. ``temper``
    then pulls the weights toward flat (w**temper, renormalised) so a tiny cell
    cannot dominate the loss.
    """
    cell = list(zip(sub["stratum"], sub["block"]))
    cell_n = pd.Series(cell).value_counts()
    blocks_in = (sub[["stratum", "block"]].drop_duplicates()
                 .groupby("stratum", observed=True).size())
    n_strata = sub["stratum"].nunique()
    w = np.array([1.0 / (n_strata * blocks_in[s] * cell_n[(s, b)]) for s, b in cell])
    w = (w / w.mean()) ** temper
    return w / w.mean()


def prep(tab: pd.DataFrame, features) -> pd.DataFrame:
    sub = tab.dropna(subset=list(features) + [TARGET]).reset_index(drop=True).copy()
    sub["block"] = sub["station"].map(block_of)
    sub["stratum"] = sub["station"].map(strata_from_forcing())
    return sub


def blocked_cv(sub: pd.DataFrame, features, factory, weighted: bool,
               label: str) -> pd.DataFrame:
    out = sub[["station", "block", "stratum", "time", TARGET]].copy()
    out["pred"] = np.nan
    for blk in sorted(sub["block"].unique()):
        te = (sub["block"] == blk).values
        tr = sub.loc[~te]
        est = factory()
        if weighted:
            est.fit(tr[features], tr[TARGET], sample_weight=make_weights(tr))
        else:
            est.fit(tr[features], tr[TARGET])
        out.loc[te, "pred"] = est.predict(sub.loc[te, features])
        print(f"    {label}: held out {blk} "
              f"(trained on {tr['station'].nunique()} stations)", flush=True)
    return out


def summarise(name: str, out: pd.DataFrame) -> None:
    p = metrics(out[TARGET], out["pred"])
    stn = out.groupby("station").apply(
        lambda g: pd.Series(metrics(g[TARGET], g["pred"])), include_groups=False)
    blk = out.groupby("block").apply(
        lambda g: pd.Series(metrics(g[TARGET], g["pred"])), include_groups=False)
    print(f"\n== {name}")
    print(f"  pooled       NSE {p['nse']:+.3f}  r {p['r']:.2f}  "
          f"ubRMSE {p['ubrmse']:.2f}  bias {p['bias']:+.2f}")
    print(f"  block-level  mean NSE {blk['nse'].mean():+.3f}  "
          f"median {blk['nse'].median():+.3f}  NSE>0 {int((blk['nse'] > 0).sum())}/{len(blk)}")
    print(f"  station      median NSE {stn['nse'].median():+.2f}  "
          f"NSE>0 {int((stn['nse'] > 0).sum())}/{len(stn)}  "
          f"median |bias| {stn['bias'].abs().median():.2f}")
    for b, r in blk.sort_values("nse").iterrows():
        print(f"    {b:<9} NSE {r['nse']:+.3f}  r {r['r']:.2f}  "
              f"bias {r['bias']:+.2f}  n {int(r['n'])}")


def m8_statics_with_aridity() -> pd.DataFrame:
    clim = pd.read_csv(DATA / "process_climate_statics.csv").set_index("station")
    return m8.load_statics().join(clim[["aridity"]], how="inner")


def awc_capacity() -> pd.Series:
    """SLGA AWC as per-station bucket capacity (model8's 'tested and not
    defaulted' physical route -- it stacks with aridity+weights under blocked
    validation even though its solo station-out gain was negligible)."""
    return pd.read_csv(DATA / "process_soil_statics.csv").set_index("station")["soil_awc"]


def model6_features() -> pd.DataFrame:
    cache = DATA / "model6_features_2006_2010.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["time"])
    feat = m6.ensure_features(pd.read_csv(DATA / "train_catchment_plus_m_2006_2010.csv"))
    feat.to_csv(cache, index=False)
    return feat


RUNS = {
    # key: (table loader, features, estimator factory, weighted, output name)
    "m8":  (lambda: pd.read_csv(DATA / "process_target_2006_2010.csv", parse_dates=["time"]),
            m8.FEATURES, m8.build_estimator, False, "model8_blockcv"),
    "m8w": (lambda: pd.read_csv(DATA / "process_target_2006_2010.csv", parse_dates=["time"]),
            m8.FEATURES, m8.build_estimator, True, "model8_blockcv_weighted"),
    "m8a": (lambda: pd.read_csv(DATA / "process_target_2006_2010.csv", parse_dates=["time"]),
            m8.FEATURES, lambda: BucketEstimator(static=m8_statics_with_aridity()),
            False, "model8_blockcv_aridity"),
    "m8aw": (lambda: pd.read_csv(DATA / "process_target_2006_2010.csv", parse_dates=["time"]),
             m8.FEATURES, lambda: BucketEstimator(static=m8_statics_with_aridity()),
             True, "model8_blockcv_aridity_weighted"),
    "m8cap": (lambda: pd.read_csv(DATA / "process_target_2006_2010.csv", parse_dates=["time"]),
              m8.FEATURES,
              lambda: BucketEstimator(static=m8.load_statics(), capacity=awc_capacity()),
              False, "model8_blockcv_capacity"),
    "m8capaw": (lambda: pd.read_csv(DATA / "process_target_2006_2010.csv", parse_dates=["time"]),
                m8.FEATURES,
                lambda: BucketEstimator(static=m8_statics_with_aridity(),
                                        capacity=awc_capacity()),
                True, "model8_blockcv_capacity_aridity_weighted"),
    "m6":  (model6_features, m6.FEATURES, m6.build_estimator, False, "model6_blockcv"),
    "m6w": (model6_features, m6.FEATURES, m6.build_estimator, True, "model6_blockcv_weighted"),
}


if __name__ == "__main__":
    for key in (sys.argv[1:] or list(RUNS)):
        loader, features, factory, weighted, name = RUNS[key]
        sub = prep(loader(), features)
        out = blocked_cv(sub, features, factory, weighted, key)
        out.to_csv(DATA / f"{name}_predictions.csv", index=False)
        summarise(name.replace("_", " "), out)
