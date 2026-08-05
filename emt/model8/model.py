"""model8 -- the process model, full stack: soil + terrain + climate statics,
AWC bucket capacity, stratified training weights.

Identical bucket water balance to :mod:`emt.model7` (same five calibrated
parameters, same forcing store, same two-stage fit); the configuration adds
three things on top, each validated under **blocked** (leave-one-block-out)
cross-validation as well as the classic station-out harness (see the
handout's blocked-validation page for the full experiment):

* **Statics for the ridge offset stage**: SLGA root-zone soil (clay / sand /
  AWC / bulk density) + terrain (TWI / slope / elevation) + the **aridity
  normal** (mean P/PET from the SILO forcing) -- soil carries the between-site
  level within a climate, aridity carries the level *across* climates.
* **SLGA AWC as per-station bucket capacity** (``capacity=``): higher-AWC
  soils genuinely hold and read out more water. Negligible alone (AWC spread
  here is only 10.9 +/- 0.7 %), it stacks with the other two.
* **Stratified sample weights** (:func:`stratified_weights`): aridity-tertile
  x block cells weighted so ten clustered Yanco stations no longer outvote
  three scattered dry M-sites; tempered ``w**0.5``. Owned by the estimator
  (``weight_fn``), so the standard ``fit(X, y)`` harness applies them.

Skill (37 stations, 2006-2010), against the pre-stack configuration:

    station-out   pooled NSE +0.41 (was +0.40)   median station NSE +0.13 (was +0.22)
    blocked       pooled NSE +0.32 (was +0.22)   median station NSE +0.07 (was -0.18)

The station-out column is the *interpolation* estimate (cluster neighbours in
training), the blocked column the *transfer* estimate -- the honest figure
for a national product. The one cost of the stack is the station-out median
(the aridity term overshoots the best-behaved station of a held-out cluster's
climate, A5); everything else improves in both harnesses. An earlier
configuration without capacity/aridity/weights is retained in the handout's
tables as the published reference.

Requires ``data/process_soil_statics.csv`` (TERN API key) and
``data/process_climate_statics.csv``; both built by :mod:`emt.model7.build`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from emt.evaluation import TARGET, leave_site_out_cv as _cv, metrics  # noqa: F401
from emt.model7 import model as m7
from emt.model7.model import BucketEstimator, FEATURES  # noqa: F401
from emt.slga import SOIL_VARS

TERRAIN_STATIC_VARS = ("twi", "slope", "elevation")
CLIMATE_STATIC_VARS = ("aridity",)
STATIC_VARS = [*SOIL_VARS, *TERRAIN_STATIC_VARS, *CLIMATE_STATIC_VARS]

SOIL_CSV = Path("data/process_soil_statics.csv")
TERRAIN_CSV = Path("data/process_terrain_statics.csv")
CLIMATE_CSV = Path("data/process_climate_statics.csv")

# Stratified-weight configuration (see stratified_weights).
N_STRATA = 3
WEIGHT_TEMPER = 0.5


def load_statics(soil_csv: Path = SOIL_CSV,
                 terrain_csv: Path = TERRAIN_CSV,
                 climate_csv: Path = CLIMATE_CSV,
                 climate: bool = True) -> pd.DataFrame:
    """Per-station statics (index = station). ``climate=False`` gives the
    pre-stack soil+terrain set (the handout's published reference config)."""
    soil = pd.read_csv(soil_csv).set_index("station")
    terr = pd.read_csv(terrain_csv).set_index("station")
    out = soil.join(terr, how="inner")
    if climate:
        clim = pd.read_csv(climate_csv).set_index("station")
        out = out.join(clim[[*CLIMATE_STATIC_VARS]], how="inner")
        return out[STATIC_VARS]
    return out[[*SOIL_VARS, *TERRAIN_STATIC_VARS]]


def awc_capacity(soil_csv: Path = SOIL_CSV) -> pd.Series:
    """SLGA available water capacity per station -- the bucket-capacity input."""
    return pd.read_csv(soil_csv).set_index("station")["soil_awc"]


def block_of(station: str) -> str:
    """Spatially independent location: cluster prefix, or the M-site itself."""
    return {"Y": "YANCO", "K": "KYEAMBA", "A": "ADELONG"}.get(station[0], station)


def stratified_weights(X: pd.DataFrame,
                       climate_csv: Path = CLIMATE_CSV,
                       temper: float = WEIGHT_TEMPER) -> np.ndarray:
    """Hierarchical training weights: aridity stratum -> block -> sample.

    Stations are cut into ``N_STRATA`` aridity (P/PET) tertiles; each stratum
    receives equal total weight, split equally over the spatial *blocks*
    inside it, then over each cell's samples -- so a dense cluster no longer
    outvotes scattered single stations of the same climate. ``temper`` pulls
    the weights toward flat (``w**temper``, renormalised) so a tiny cell
    cannot dominate the loss.

    The tertile *edges* are fixed once from the climate-statics file (all
    stations), not recomputed per training subset: aridity is a covariate
    known everywhere (no target information), and per-fold edges destabilise
    the design -- removing a whole block shifts the cut points enough to cost
    ~0.4 blocked station-median NSE. Cell membership and counts still come
    from the training rows only.
    """
    aridity = pd.read_csv(climate_csv).set_index("station")["aridity"]
    stations = pd.Series(np.asarray(X["station"]))
    strat_by_station = pd.qcut(aridity, N_STRATA, labels=False)
    stratum = stations.map(strat_by_station)
    block = stations.map(block_of)
    cell = pd.Series(list(zip(stratum, block)))
    cell_n = cell.value_counts()
    blocks_in = (pd.DataFrame({"s": stratum, "b": block}).drop_duplicates()
                 .groupby("s").size())
    n_strata = stratum.nunique()
    w = np.array([1.0 / (n_strata * blocks_in[s] * cell_n[(s, b)])
                  for s, b in cell])
    w = (w / w.mean()) ** temper
    return w / w.mean()


def build_estimator(**kwargs) -> BucketEstimator:
    """The full-stack bucket: soil+terrain+aridity statics, AWC capacity,
    stratified weights. Any of the three is overridable (``static=``,
    ``capacity=None``, ``weight_fn=None``); other kwargs pass through to
    :class:`~emt.model7.model.BucketEstimator`."""
    kwargs.setdefault("static", load_statics())
    kwargs.setdefault("capacity", awc_capacity())
    kwargs.setdefault("weight_fn", stratified_weights)
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
