"""Anomaly detection for in-situ soil-moisture series.

The 2006-2010 window this project was built on is clean. The wider 2001-2025
record is not: 2,770 of 161,422 station-days carry impossible values, 435 of
them exactly **65535** (2^16 - 1, a logger sentinel), with others at 43696,
38239 and 21870 -- raw counts leaking through rather than moisture. The
contamination is almost entirely in 2021-2024 and rises year on year, which
reads as ageing loggers.

``emt.insitu.oznet`` masks ``df <= -99`` -- a lower bound only, so an upper
sentinel passes straight through. A symmetric threshold would catch the
obvious 65535 and miss everything beneath it, so this module flags by
*behaviour* as well as by value.

**The flags**, each independent, so a row can carry several:

``range``     outside 0-65 % volumetric. Below zero is impossible; above ~65 %
              exceeds the porosity of all but peats. Catches every sentinel by
              construction.
``spike``     an *isolated excursion*: the day departs from its neighbours by
              more than ``SPIKE_Z`` robust deviations AND the series returns to
              where it came from within a day. See below -- this test replaced
              a plain robust-z outlier rule, which did not work.
``flatline``  the identical value repeated for ``FLATLINE_DAYS`` or more. A
              real profile mean varies at the fourth decimal; an exactly
              constant run is a stuck sensor or a held last-good value.
``jump``      a day-on-day change exceeding ``JUMP_PCT`` with less than
              ``JUMP_RAIN_MM`` of rain. Soil moisture cannot rise sharply
              without water arriving.

**Why ``spike`` is a persistence test and not an outlier test.** The first
version of this module flagged any robust-z outlier. Validated against SILO
rain, that rule proved to be detecting *weather*: 65.6 % of what it flagged sat
above the local median, those wet excursions carried a median 5.3 mm of rain
over the preceding three days against 0.0 mm for the dry ones, and flagged rows
were rain-bearing 37 % of the time against a 21.5 % baseline. A detector that
fires preferentially when it rains is finding real wetting, and dropping its
catch would have manufactured a dry bias -- the precise failure this project is
trying to remove.

The discriminator is **persistence**. Wetting a soil profile takes water in and
the water stays: a genuine event holds for days and decays slowly. A logger
glitch is a single sample that reverts immediately. So a value is flagged only
when it departs from its neighbours *and* the series closes back to its
pre-excursion level within ``SPIKE_RETURN_DAYS``. The test that this rule is
sound is that its catch shows **no** rain enrichment over baseline; the numbers
are in ``handout/modules/qc.md`` and are re-checked by
``handout/validate_qc.py``.

**On ``jump``.** Yanco is an irrigation district, so a sharp rise without rain
can be a real irrigation event rather than a fault. This flag is therefore
advisory: reported, and deliberately NOT part of :func:`drop_invalid`.

The design principle throughout: automatic removal is limited to the
indefensible, everything else is surfaced for a human.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VWC_MIN, VWC_MAX = 0.0, 65.0
SPIKE_WINDOW = 31          # days, centred
SPIKE_Z = 5.0              # robust z; 5 is conservative, ~1 in 3.5M under normality
SPIKE_RETURN_DAYS = 1      # an excursion must close within this many days
SPIKE_RETURN_FRAC = 0.35   # ...to within this fraction of its own size
FLATLINE_DAYS = 14
JUMP_PCT = 8.0             # percentage points day-on-day
JUMP_RAIN_MM = 2.0

FLAGS = ["range", "spike", "flatline", "jump"]


def _robust_z(s: pd.Series, window: int = SPIKE_WINDOW) -> pd.Series:
    """Deviation from a centred rolling median, scaled by the rolling MAD."""
    med = s.rolling(window, center=True, min_periods=max(5, window // 4)).median()
    mad = (s - med).abs().rolling(window, center=True,
                                  min_periods=max(5, window // 4)).median()
    scale = 1.4826 * mad                      # MAD -> sigma for a normal
    scale = scale.where(scale > 1e-6)         # a genuinely flat run has MAD 0
    return (s - med).abs() / scale


def _isolated_excursion(s: pd.Series) -> pd.Series:
    """Departs from neighbours *and* reverts -- a glitch, not a wetting event.

    Real wetting persists: water enters the profile and drains over days. A
    logger glitch is one bad sample with sound values either side. The
    excursion size is measured against the previous day, and it counts as
    closed if the level ``SPIKE_RETURN_DAYS`` later has come back to within
    ``SPIKE_RETURN_FRAC`` of that size.
    """
    z = _robust_z(s)
    prev, nxt = s.shift(1), s.shift(-SPIKE_RETURN_DAYS)
    size = s - prev
    returned = (nxt - prev).abs() <= SPIKE_RETURN_FRAC * size.abs()
    big = size.abs() > 1e-9
    return ((z > SPIKE_Z) & returned & big).fillna(False)


def _flatline(s: pd.Series, days: int = FLATLINE_DAYS) -> pd.Series:
    """True where a value belongs to a run of >= ``days`` identical values."""
    grp = (s != s.shift()).cumsum()
    return s.groupby(grp).transform("size").ge(days) & s.notna()


def flag_station(df: pd.DataFrame, rain: pd.Series | None = None,
                 value: str = "sm_rootzone_pct") -> pd.DataFrame:
    """Flag anomalies in one station's daily series (must be time-sorted)."""
    s = df[value].astype(float)
    out = pd.DataFrame(index=df.index)
    out["range"] = s.notna() & ((s < VWC_MIN) | (s > VWC_MAX))
    # a sentinel would otherwise dominate its own neighbourhood statistics
    clean = s.where(~out["range"])
    out["spike"] = _isolated_excursion(clean)
    out["flatline"] = _flatline(clean)
    if rain is not None:
        d = clean.diff().abs()
        out["jump"] = (d > JUMP_PCT) & (rain.reindex(df.index).fillna(0) < JUMP_RAIN_MM)
    else:
        out["jump"] = False
    return out.fillna(False).astype(bool)


def flag_target(target: pd.DataFrame, forcing: pd.DataFrame | None = None,
                value: str = "sm_rootzone_pct") -> pd.DataFrame:
    """Flag anomalies across a whole target table.

    Args:
        target: the contract table (``station``, ``time``, ``value``).
        forcing: optional daily forcing with ``station``, ``time``,
            ``daily_rain``; enables the ``jump`` flag.

    Returns:
        ``target`` with one boolean column per flag plus ``any_flag``.
    """
    t = target.copy()
    t["time"] = pd.to_datetime(t["time"])
    rain_by_station = {}
    if forcing is not None:
        f = forcing.copy()
        f["time"] = pd.to_datetime(f["time"])
        for stn, g in f.groupby("station"):
            rain_by_station[stn] = g.set_index("time")["daily_rain"]

    parts = []
    for stn, g in t.groupby("station", sort=False):
        g = g.sort_values("time")
        flags = flag_station(g.set_index("time"), rain_by_station.get(stn), value)
        flags.index = g.index
        parts.append(flags)
    flags = pd.concat(parts).reindex(t.index)
    out = pd.concat([t, flags], axis=1)
    out["any_flag"] = flags[FLAGS].any(axis=1)
    return out


DROP_FLAGS = ["range", "flatline"]


def drop_invalid(flagged: pd.DataFrame) -> pd.DataFrame:
    """Remove only ``range`` and ``flatline`` -- what cannot be real.

    **``spike`` is deliberately excluded**, and that is a finding rather than
    an oversight. Even after the persistence test cut the plain outlier rule's
    catch by 92 %, the survivors remained 1.54x enriched in rain against
    baseline (``handout/validate_qc.py``). A single-day wet excursion that
    drains by the next day is physically ordinary in a shallow or sandy
    profile, so the rule cannot separate malfunction from fast wetting in this
    record. Dropping its catch would bias the training target dry.

    ``jump`` is likewise advisory: at Yanco a rise without rain may be
    irrigation.

    Both are reported so a human can inspect them; neither is deleted.
    """
    keep = ~flagged[DROP_FLAGS].any(axis=1)
    return flagged.loc[keep].drop(columns=FLAGS + ["any_flag"])


def summarise(flagged: pd.DataFrame) -> pd.DataFrame:
    """Per-flag counts, shares, and the worst-affected stations."""
    n = len(flagged)
    rows = []
    for f in FLAGS + ["any_flag"]:
        m = flagged[f]
        worst = flagged.loc[m, "station"].value_counts().head(3)
        rows.append({"flag": f, "rows": int(m.sum()),
                     "pct": round(m.sum() / n * 100, 3),
                     "stations": int(flagged.loc[m, "station"].nunique()),
                     "worst": ", ".join(f"{k}({v})" for k, v in worst.items())})
    return pd.DataFrame(rows)
