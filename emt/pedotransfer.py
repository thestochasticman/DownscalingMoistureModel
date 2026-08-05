"""Soil hydraulic limits from texture -- Saxton & Rawls (2006) pedotransfer.

model7/model8 calibrate the readout as two *global* constants (``theta_r``,
``dtheta``), so every site in Australia is modelled as spanning the same
volumetric range -- 17.7 % to 34.6 % for the shipped model8 fit. That is a hard
ceiling, and sites live above it: K12's *mean* observation (39.0 %) exceeds the
highest value model8 can produce there, which is why no configuration ever
improved it.

But those two numbers are not process rates -- they are soil hydraulic
properties, and texture predicts them. This module estimates, per location:

    wilting_point   theta at 1500 kPa   -- the bucket's empty state
    field_capacity  theta at 33 kPa     -- drained upper limit
    saturation      theta at 0 kPa      -- total porosity

from SLGA sand/clay and an organic-matter term, using the Saxton & Rawls (2006)
regression set (*Soil Sci. Soc. Am. J.* 70:1569-1578), the standard
texture-based pedotransfer for exactly this purpose.

**Measured against the OzNet stations** (37 sites), these track the observed
levels where SLGA's own AWC product does not:

    predictor        corr with site p5   with p95   with observed swing
    wilting_point         +0.51            +0.60         +0.39
    field_capacity        +0.52            +0.62         +0.40
    field_cap - wilting   +0.55            +0.66         +0.43
    SLGA soil_awc         +0.08            -0.01         -0.10

SLGA's ``Available_Water_Capacity`` is effectively uninformative here (it spans
only 9.7-13.0 % across the catchment) -- which is why model8's capacity route
was weak on its own. The Saxton-Rawls range carries real between-site signal.

**Organic matter** is not currently fetched (SLGA serves SOC, but the EMT soil
loader pulls clay/sand/AWC/bulk-density only), so ``om_pct`` defaults to a
nominal 1.5 % w/w -- typical for these agricultural profiles. The estimates are
mildly sensitive to it; sourcing real SOC is a natural refinement.
"""
from __future__ import annotations

import numpy as np

DEFAULT_OM_PCT = 1.5


def _as_fraction(pct):
    return np.asarray(pct, dtype=float) / 100.0


def wilting_point(clay_pct, sand_pct, om_pct: float = DEFAULT_OM_PCT):
    """Volumetric water content at 1500 kPa (%), Saxton & Rawls eq. 1."""
    S, C = _as_fraction(sand_pct), _as_fraction(clay_pct)
    t = (-0.024 * S + 0.487 * C + 0.006 * om_pct + 0.005 * (S * om_pct)
         - 0.013 * (C * om_pct) + 0.068 * (S * C) + 0.031)
    return (t + (0.14 * t - 0.02)) * 100.0


def field_capacity(clay_pct, sand_pct, om_pct: float = DEFAULT_OM_PCT):
    """Volumetric water content at 33 kPa (%), Saxton & Rawls eq. 2."""
    S, C = _as_fraction(sand_pct), _as_fraction(clay_pct)
    t = (-0.251 * S + 0.195 * C + 0.011 * om_pct + 0.006 * (S * om_pct)
         - 0.027 * (C * om_pct) + 0.452 * (S * C) + 0.299)
    return (t + (1.283 * t ** 2 - 0.374 * t - 0.015)) * 100.0


def saturation(clay_pct, sand_pct, om_pct: float = DEFAULT_OM_PCT):
    """Total porosity / saturated water content (%), Saxton & Rawls eqs. 3-5."""
    S, C = _as_fraction(sand_pct), _as_fraction(clay_pct)
    fc = field_capacity(clay_pct, sand_pct, om_pct) / 100.0
    t = (0.278 * S + 0.034 * C + 0.022 * om_pct - 0.018 * (S * om_pct)
         - 0.027 * (C * om_pct) - 0.584 * (S * C) + 0.078)
    t = t + (0.636 * t - 0.107)
    return (fc + t - 0.097 * S + 0.043) * 100.0


def limits(clay_pct, sand_pct, om_pct: float = DEFAULT_OM_PCT) -> dict:
    """``{wilting_point, field_capacity, saturation, available}`` in volumetric %."""
    wp = wilting_point(clay_pct, sand_pct, om_pct)
    fc = field_capacity(clay_pct, sand_pct, om_pct)
    return {"wilting_point": wp, "field_capacity": fc,
            "saturation": saturation(clay_pct, sand_pct, om_pct),
            "available": fc - wp}


if __name__ == "__main__":
    # Saxton & Rawls table 1 sanity check: a silt loam and a sand.
    for name, (clay, sand) in {"silt loam": (15, 20), "sand": (5, 90),
                               "clay": (50, 20)}.items():
        L = limits(clay, sand)
        print(f"{name:10s} clay {clay:>2}% sand {sand:>2}%  ->  "
              f"WP {L['wilting_point']:.1f}%  FC {L['field_capacity']:.1f}%  "
              f"SAT {L['saturation']:.1f}%  avail {L['available']:.1f}%")
