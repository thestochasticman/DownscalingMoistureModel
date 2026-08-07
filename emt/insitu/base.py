"""The in-situ contract: what every network loader must produce.

Everything downstream of this package -- :mod:`emt.build_dataset`,
:mod:`emt.model7.build`, and every cross-validation harness in ``handout/`` --
consumes one table shape and one coordinate table. Until now that shape was
implicit in :mod:`emt.insitu.oznet`, which is why the project has exactly one
in-situ network. This module states the contract so a second network is a
parser rather than a rewrite.

**The target table** (long format, one row per station-day)::

    site              str    grouping label; the spatial CV block derives from it
    station           str    unique station code across ALL networks
    time              datetime64  midnight-normalised day
    sm_rootzone_pct   float  daily mean volumetric water content, per cent
    n_layers          int    how many sensor depths formed the integral

**The coordinate table** (see :mod:`emt.insitu.coordinates`)::

    station, lat, lon   (EPSG:4326)

**On ``sm_rootzone_pct``.** The target is the 0-90 cm profile mean, chosen to
match the SMIPS ``TotalBucket`` layer the project set out to downscale. A
network reporting other depths must be reconciled to that, and *how* it is
reconciled is a scientific choice with consequences -- not a formatting step.
:func:`depth_weighted_mean` performs the mechanical part; the choice of which
depths represent the root zone belongs to each network's module, stated in its
docstring.

**On station codes.** They must be unique across networks, because ``station``
is the cross-validation grouping key and a collision would silently merge two
sites. :func:`check_target` enforces uniqueness within a table;
:func:`assert_disjoint` checks it across two.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["site", "station", "time", "sm_rootzone_pct", "n_layers"]
COORD_COLUMNS = ["station", "lat", "lon"]

# Volumetric water content limits. Below zero or above 100 % is impossible and
# is treated as an error. A *median* outside MEDIAN_BAND is how a unit error
# actually shows up -- a table in fractions has median ~0.2, which sits happily
# inside 0-100 and would otherwise pass unnoticed. Individual values above
# IMPLAUSIBLE are only warned about: sustained inundation genuinely produces
# them (OzNet M6 held 62-70 % for twenty days during the record November 2010
# La Nina flooding), so they are suspicious, not invalid.
VWC_MIN, VWC_MAX = 0.0, 100.0
MEDIAN_BAND = (2.0, 60.0)
IMPLAUSIBLE = 55.0


@runtime_checkable
class InSituNetwork(Protocol):
    """What a network module must expose to be usable by the pipeline.

    Implemented by :mod:`emt.insitu.oznet`; see that module as the reference.
    A module satisfies this structurally -- no inheritance required.
    """

    NETWORK: str

    def load_daily_rootzone(self, **kwargs) -> pd.DataFrame:
        """Return the target table for this network (see module docstring)."""

    def station_coords(self, stations: list[str], **kwargs) -> pd.DataFrame:
        """Return ``[station, lat, lon]`` for the requested stations."""


def depth_weighted_mean(frame: pd.DataFrame,
                        depths: dict[str, float],
                        require_all: bool = True) -> pd.Series:
    """Combine per-depth sensor columns into one profile mean.

    Args:
        frame: columns are sensor names, index is time.
        depths: ``{column: layer thickness}``. Thickness is the weight, so a
            0-30/30-60/60-90 cm set of equal thirds reduces to a plain mean --
            which is what OzNet does, and why its loader predates this helper.
        require_all: if True a timestamp missing any listed depth yields NaN,
            so the integral is never formed from a partial profile. This is the
            conservative choice and the one OzNet has always made.

    Returns:
        Series of the weighted mean, same index as ``frame``.
    """
    have = [c for c in depths if c in frame.columns]
    if not have:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    w = np.array([depths[c] for c in have], dtype=float)
    vals = frame[have].to_numpy(dtype=float)
    if require_all:
        bad = ~np.isfinite(vals).all(axis=1)
    else:
        bad = ~np.isfinite(vals).any(axis=1)
    with np.errstate(invalid="ignore"):
        num = np.nansum(vals * w, axis=1)
        den = np.nansum(np.where(np.isfinite(vals), w, 0.0), axis=1)
        out = np.where(den > 0, num / den, np.nan)
    out[bad] = np.nan
    return pd.Series(out, index=frame.index, dtype=float)


def check_target(df: pd.DataFrame, network: str = "?", strict: bool = True) -> pd.DataFrame:
    """Validate a network's target table against the contract.

    Raises ``ValueError`` on any violation when ``strict``; otherwise prints
    the problems and returns the frame unchanged. Warnings are always printed
    and never raise.

    Errors: missing columns, wrong dtypes, unnormalised days, duplicate
    station-days, null keys, values outside 0-100 %, or a median outside
    ``MEDIAN_BAND`` (which is how a fraction-vs-per-cent mix-up presents).

    Warnings: values above ``IMPLAUSIBLE``, and a station set whose profile
    depths disagree -- both are real in the OzNet data and neither invalidates
    the table.
    """
    problems: list[str] = []
    warnings: list[str] = []
    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    if not problems:
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            problems.append("'time' is not datetime64")
        elif (df["time"] != df["time"].dt.normalize()).any():
            problems.append("'time' has non-midnight values; days must be normalised")
        if not pd.api.types.is_numeric_dtype(df["sm_rootzone_pct"]):
            problems.append("'sm_rootzone_pct' is not numeric")
        else:
            v = df["sm_rootzone_pct"]
            bad = v.notna() & ((v < VWC_MIN) | (v > VWC_MAX))
            if bad.any():
                problems.append(
                    f"{int(bad.sum())} rows outside {VWC_MIN}-{VWC_MAX} % "
                    f"(min {v.min():.2f}, max {v.max():.2f})")
            med = v.median()
            if pd.notna(med) and not (MEDIAN_BAND[0] <= med <= MEDIAN_BAND[1]):
                problems.append(
                    f"median {med:.3f} outside {MEDIAN_BAND} -- volumetric PER CENT "
                    f"expected, not a fraction")
            n_hi = int((v > IMPLAUSIBLE).sum())
            if n_hi:
                warnings.append(
                    f"{n_hi} rows above {IMPLAUSIBLE} % (max {v.max():.1f}); "
                    f"plausible only under sustained inundation -- check the sensor")
        depths = df.groupby("station")["n_layers"].max()
        if depths.nunique() > 1:
            odd = depths[depths < depths.max()]
            warnings.append(
                f"profile depth is not homogeneous: {len(odd)} of {len(depths)} "
                f"stations use fewer than {int(depths.max())} sensor layers "
                f"({', '.join(f'{k}:{int(x)}' for k, x in odd.head(6).items())}"
                f"{' ...' if len(odd) > 6 else ''}). Their target is a shallower "
                f"profile than the rest, which is a level bias the model cannot "
                f"see")
        dup = df.duplicated(["station", "time"]).sum()
        if dup:
            problems.append(f"{int(dup)} duplicate station-days")
        if df["station"].isna().any() or df["site"].isna().any():
            problems.append("null station or site")

    if warnings:
        print(f"[{network}] contract warnings:\n  ! " + "\n  ! ".join(warnings),
              flush=True)
    if problems:
        msg = f"[{network}] target table violates the in-situ contract:\n  - " + \
              "\n  - ".join(problems)
        if strict:
            raise ValueError(msg)
        print(msg, flush=True)
    return df


def assert_disjoint(a: pd.DataFrame, b: pd.DataFrame,
                    name_a: str = "A", name_b: str = "B") -> None:
    """Fail if two networks share a station code.

    ``station`` is the cross-validation grouping key, so a collision would
    silently merge two physically separate sites into one fold.
    """
    clash = sorted(set(a["station"]) & set(b["station"]))
    if clash:
        raise ValueError(
            f"{name_a} and {name_b} share station codes {clash[:10]}"
            f"{' ...' if len(clash) > 10 else ''}; codes must be unique across "
            f"networks because 'station' is the CV grouping key")


def combine(*tables: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Concatenate per-network target tables into one, checking the contract.

    Station codes must already be disjoint; the result is sorted and
    re-validated so a combined table is as trustworthy as its parts.
    """
    tables = [t for t in tables if t is not None and len(t)]
    if not tables:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            assert_disjoint(tables[i], tables[j], f"table{i}", f"table{j}")
    out = pd.concat([t[TARGET_COLUMNS] for t in tables], ignore_index=True)
    out = out.sort_values(["site", "station", "time"]).reset_index(drop=True)
    return check_target(out, network="combined", strict=strict)
