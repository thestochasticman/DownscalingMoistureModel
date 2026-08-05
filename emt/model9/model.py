"""model9 -- the process model with a **pedotransfer readout**.

model7 and model8 calibrate the storage->moisture readout as two *global*
constants::

    vwc% = theta_r + dtheta * S/smax        theta_r = 17.72, dtheta = 16.87

so every site in Australia is modelled as spanning the same 17.7-34.6 % range.
That is a hard ceiling, and sites live above it: **K12's mean observation
(39.0 %) exceeds the highest value model8 can produce there**, which is why no
model8 configuration -- aridity, capacity or weights -- ever moved it (its
per-station NSE sat at about -16 in both harnesses).

But those two numbers are not process rates. They are soil hydraulic
properties, and texture predicts them. model9 replaces them with per-site
Saxton & Rawls (2006) limits from the SLGA clay/sand already in hand
(:mod:`emt.pedotransfer`)::

    vwc% = WP_i + gamma * (FC_i - WP_i) * S/smax  + z.c

``WP_i`` (wilting point) is the site's empty state and ``FC_i - WP_i`` its
plant-available range; a single global ``gamma`` scales that range, because the
observed core swing (p95-p5, mean 14.6 %) runs wider than the textbook
available range (mean 10.6 %) -- soils here wet beyond field capacity. So two
calibrated levels become **one calibrated scale plus a lookup**, and the output
range varies by site instead of being a global cap.

Everything else is model8: the same bucket, the same soil/terrain/aridity ridge
offsets, the same AWC capacity and stratified weights.

Why texture and not SLGA's own AWC product: measured across these 37 stations,
Saxton-Rawls WP/FC correlate +0.51..+0.66 with the observed site levels while
SLGA ``soil_awc`` correlates about 0.00 -- it spans only 9.7-13.0 % here and
carries no between-site signal (which is also why model8's capacity route was
weak on its own).

Requires ``data/process_pedotransfer_statics.csv`` alongside model8's inputs;
all are built by :mod:`emt.model7.build` with no extra download.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from emt.evaluation import TARGET, leave_site_out_cv as _cv, metrics  # noqa: F401
from emt.model7 import model as m7
from emt.model7.model import BucketEstimator, FEATURES  # noqa: F401
from emt.model8.model import (STATIC_VARS, TERRAIN_STATIC_VARS,  # noqa: F401
                              CLIMATE_STATIC_VARS, awc_capacity, block_of,
                              load_statics, stratified_weights)

PEDO_CSV = Path("data/process_pedotransfer_statics.csv")
HYDRAULIC_CSV = Path("data/process_slga_hydraulic_statics.csv")

# Which pedotransfer limits define the bucket's empty and full states. The
# "available" pairing (WP -> FC) is the physical default; "saturation" spans
# WP -> SAT for soils that routinely wet past field capacity.
READOUT_SPAN = "available"

# Where the limits come from. "saxton" estimates them from texture via
# Saxton & Rawls (:mod:`emt.pedotransfer`), which carries a nominal
# organic-matter assumption. "slga" uses SLGA's own published drained upper
# limit and 15-bar limit instead -- no regression, no OM assumption.
READOUT_SOURCE = "saxton"


def load_readout_limits(pedo_csv: Path = PEDO_CSV,
                        span: str = READOUT_SPAN,
                        source: str = READOUT_SOURCE,
                        hydraulic_csv: Path = HYDRAULIC_CSV) -> pd.DataFrame:
    """Per-station ``theta_r`` / ``dtheta`` (index = station), in volumetric %.

    ``theta_r`` is always the wilting point and ``dtheta`` the range above it.
    With ``source="saxton"`` these are Saxton-Rawls estimates from texture and
    ``span`` selects the plant-available range (WP->FC) or the full range to
    saturation. With ``source="slga"`` they are SLGA's published DUL and L15,
    for which only the available span exists.
    """
    if source == "slga":
        h = pd.read_csv(hydraulic_csv).set_index("station")
        return pd.DataFrame({"theta_r": h["soil_l15"],
                             "dtheta": h["soil_dul"] - h["soil_l15"]})
    p = pd.read_csv(pedo_csv).set_index("station")
    top = p["field_capacity"] if span == "available" else p["saturation"]
    return pd.DataFrame({"theta_r": p["wilting_point"],
                         "dtheta": top - p["wilting_point"]})


def build_estimator(span: str = READOUT_SPAN, source: str = READOUT_SOURCE,
                    **kwargs) -> BucketEstimator:
    """model8's full stack with the readout constants replaced by per-site
    hydraulic limits. ``source`` selects Saxton-Rawls-from-texture (default) or
    SLGA's own DUL/L15; ``span`` selects WP->FC (default) or WP->saturation for
    the Saxton-Rawls source. All other kwargs pass through to
    :class:`~emt.model7.model.BucketEstimator`."""
    kwargs.setdefault("static", load_statics())
    kwargs.setdefault("capacity", awc_capacity())
    kwargs.setdefault("weight_fn", stratified_weights)
    kwargs.setdefault("readout_limits", load_readout_limits(span=span, source=source))
    est = BucketEstimator(**kwargs)
    # Recorded so inference can rebuild the same limits per pixel.
    est.readout_span_ = span
    est.readout_source_ = source
    return est


def ensure_features(table: pd.DataFrame) -> pd.DataFrame:
    """model9 needs only (station, time) keys -- present in every EMT table."""
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
