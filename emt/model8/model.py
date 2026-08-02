"""model8 -- model7 plus SLGA soil: the process model with a soil level anchor.

Identical bucket water balance to :mod:`emt.model7` (same five calibrated
parameters, same forcing store, same two-stage fit); the only change is the
static set handed to the ridge offset stage: **SLGA root-zone soil**
(clay / sand / AWC / bulk density) alongside model7's terrain
(TWI / slope / elevation). One line of configuration -- and it nearly doubles
the pooled leave-site-out skill, to parity with the ML models:

    model7 (terrain only)   pooled NSE +0.18   median station NSE -0.03
    model8 (+ SLGA soil)    pooled NSE +0.40   median station NSE +0.22

This is the process-side confirmation of model6's feature-importance finding:
the between-site *level* structure the ML models learn lives in the soil maps.
Soil enters here exactly as terrain does -- a ridge-regularised per-station
readout offset fitted on training-station mean residuals -- so a held-out
station's level comes from its own SLGA values, never its observations.

Tested and not defaulted: SLGA AWC as *per-station bucket capacity*
(``capacity=`` on the estimator, the physically-motivated route). Its LOSO
gain is negligible (+0.16 vs +0.15 pooled) because AWC barely varies across
these stations (10.9 +/- 0.7 %); the offsets carry the signal. The option
remains available for regions with real AWC contrast.

Requires ``data/process_soil_statics.csv`` (SLGA needs a TERN API key; built
by :mod:`emt.model7.build`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from emt.evaluation import TARGET, leave_site_out_cv as _cv, metrics  # noqa: F401
from emt.model7 import model as m7
from emt.model7.model import BucketEstimator, FEATURES  # noqa: F401
from emt.slga import SOIL_VARS

TERRAIN_STATIC_VARS = ("twi", "slope", "elevation")
STATIC_VARS = [*SOIL_VARS, *TERRAIN_STATIC_VARS]

SOIL_CSV = Path("data/process_soil_statics.csv")
TERRAIN_CSV = Path("data/process_terrain_statics.csv")


def load_statics(soil_csv: Path = SOIL_CSV,
                 terrain_csv: Path = TERRAIN_CSV) -> pd.DataFrame:
    """Per-station soil + terrain statics (index = station)."""
    soil = pd.read_csv(soil_csv).set_index("station")
    terr = pd.read_csv(terrain_csv).set_index("station")
    return soil.join(terr, how="inner")[STATIC_VARS]


def build_estimator(**kwargs) -> BucketEstimator:
    """model7's bucket with the soil+terrain static set. ``capacity=`` (e.g.
    the ``soil_awc`` column) switches on per-station bucket capacity; all
    other kwargs pass through to :class:`~emt.model7.model.BucketEstimator`."""
    kwargs.setdefault("static", load_statics())
    return BucketEstimator(**kwargs)


def ensure_features(table: pd.DataFrame) -> pd.DataFrame:
    """model8 needs only (station, time) keys -- present in every EMT table."""
    return table


def fit(table: pd.DataFrame, estimator: BucketEstimator | None = None) -> BucketEstimator:
    est = estimator if estimator is not None else build_estimator()
    sub = table.dropna(subset=FEATURES + [TARGET])
    est.fit(sub[FEATURES], sub[TARGET])
    return est


def leave_site_out_cv(table: pd.DataFrame, group_col: str = "station",
                      **est_kwargs) -> dict:
    return _cv(table, FEATURES, lambda: build_estimator(**est_kwargs),
               group_col=group_col)


parameters = m7.parameters


if __name__ == "__main__":
    import sys
    table = (pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith(".parquet")
             else pd.read_csv(sys.argv[1]))
    cv = leave_site_out_cv(table)
    p, ps = cv["pooled"], cv["per_site"]
    print("pooled:", {k: round(v, 3) if isinstance(v, float) else v for k, v in p.items()})
    print(f"per-station NSE>0: {(ps['nse'] > 0).sum()}/{len(ps)} "
          f"(median {ps['nse'].median():.2f}); median |bias| {ps['bias'].abs().median():.2f}")
