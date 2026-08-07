# Data quality: what the 25-year record contains

<!-- NAV -->
[← In-situ networks](insitu_networks.md) · [Index](../README.md) · [Downscaling to 30 m →](downscale.py.md)
<!-- /NAV -->

Source: [`../../emt/insitu/qc.py`](../../emt/insitu/qc.py) ·
[`../validate_qc.py`](../validate_qc.py) · [`../run_extended_cv.py`](../run_extended_cv.py)

Extending OzNet from 2006–2010 to 2001–2025 was meant to buy regime variety.
It did — and it also revealed that **the wider record is not usable as
delivered**. This page records what is wrong with it, how much survives, and
the two rules that were tried and rejected.

## 1. Logger sentinels

The contract check ([`insitu/base.py`](../../emt/insitu/base.py)) refused the
extended table on sight:

```
ValueError: 2770 rows outside 0.0-100.0 % (min 0.00, max 65535.00)
```

**65535 is 2¹⁶−1** — a logger sentinel, not a measurement. Others sit at 43696,
38239 and 21870: raw counts leaking through. In total 2,979 rows (1.85 %)
violate a 0–65 % volumetric range.

Two properties matter. **None of it is in 2006–2010** — the window every
published result used is clean, which is why this was never seen. And it is
concentrated in the recent years, rising steadily: 660 rows in 2021, 825 in
2022, 1,161 in 2023. The existing loader masked `df <= -99`, a lower bound
only, so an upper sentinel passed straight through.

These are dropped. They are the only thing dropped.

## 2. Two detectors that were tried and rejected

**A robust-z outlier rule.** The obvious next step — flag anything more than
five robust deviations from a 31-day rolling median — flagged 3,412 rows,
2.18 % of even the "clean" 2006–2010 window. Before acting on that, it was
validated against SILO rain, and it failed:

| | flagged | with ≥5 mm/3 d | vs baseline |
|---|---|---|---|
| baseline (any day) | — | 21.9 % | 1.00 |
| robust-z outliers | 3,412 | 37.4 % | **1.71** |

65.6 % of its catch sat *above* the local median, and those wet excursions
carried a median 5.3 mm of preceding rain against 0.0 mm for the dry ones. **A
detector that fires when it rains is detecting weather.** Dropping its catch
would have deleted real wetting events and biased the training target dry —
the exact failure the project is trying to remove.

**A persistence test.** Replacing it with a physical discriminator — a real
wetting event holds for days, a glitch reverts within one — cut the catch by
92 %, to 264 rows. But the survivors were still **1.54×** rain-enriched. A
one-day wet excursion that drains by the next day is ordinary in a shallow or
sandy profile.

So spike detection **does not drive deletion here**. The rule is retained as a
reported flag, and [`validate_qc.py`](../validate_qc.py) re-runs the
enrichment test so the claim stays falsifiable.

## 3. The finding that matters: calibration discontinuities

With sentinels removed, model8 was trained on all 25 years. Skill is positive
through 2013 and then collapses:

| era | pooled NSE | bias |
|---|---|---|
| 2002–2013 | +0.13 … +0.34 | within ±2 pp |
| 2017–2019 | −1.29 … −2.13 | **−6.6 … −7.2 pp** |
| 2023–2025 | −0.27 … −0.35 | r goes **negative** |

A model does not anti-correlate with reality. Testing the observations against
rainfall — an independent reference the sensors cannot influence — shows why.
Across the 19 stations reporting in both eras, **18 read wetter in the later
record**, and the shift is not explained by rain:

| station | early mean | late mean | Δ | rain |
|---|---|---|---|---|
| **Y6** | 25.9 % | 46.0 % | **+20.1 pp** | 322 → 359 mm (+11 %) |
| **Y13** | 21.8 % | 35.0 % | +13.1 pp | 343 → 469 mm |
| **Y4** | 27.8 % | 34.4 % | +6.6 pp | 366 → 376 mm (+3 %) |

The clearest single number: **2019** was the Tinderbox Drought, 217 mm of
rain, among the driest years the Murrumbidgee has recorded — and the network
reports the **wettest** mean soil moisture in the entire 25-year archive,
15.1 % per 100 mm of rain against ~5 in the 2000s. Soil moisture cannot rise
while rainfall stays flat.

A rain-adjusted step scan locates it as **per-station discontinuities, not one
network-wide event**: 15 stations carry a significant step, scattered across
the record and running in both directions (Y6 +17.8 pp at 2016-02, Y13 +10.9
at 2013-04, K12 −9.5 as early as 2005-11). Pre-2013 steps skew negative,
post-2013 strongly positive. This is sensor replacement and recalibration at
individual sites — routine in a long in-situ record, and invisible to any
per-row filter, because every affected value is individually plausible.

## 4. What is usable

The homogeneous era is **2002–2013**, and it is a clear gain:

| | rows | pooled | block-median | blocks>0 | station-median |
|---|---|---|---|---|---|
| legacy 2006–2010 | 47,786 | +0.322 | +0.249 | 7/9 | +0.07 |
| **extended 2002–2013** | **101,524** | +0.316 | +0.243 | 7/9 | **+0.160** |
| full 2001–2025 | 157,364 | +0.263 | +0.254 | 5/9 | +0.023 |

Twice the data and twelve years instead of five, with transfer skill held and
**station-level skill more than doubled**. The added years bring the
Millennium Drought and the 2010–12 La Niña — the regime variety the
five-year window lacked. Per-year bias stays inside ±2 pp throughout.

A side benefit: the aridity static is computed over the same window the model
trains on, so the training/inference window mismatch tracked on the
`aridity-reference-window` branch does not arise — it is fixed by
construction.

**M3** is excluded (1,079 rows): it has coordinates but no forcing or statics,
and has been silently absent from every result to date.

## 5. Reading year folds

Station availability is not constant — **34 stations clear 200 days in 2006,
10 in 2024**. A late-record year fold is a smaller and differently-composed
sample, so a skill change there may be the network shrinking rather than the
model failing. Per-year station counts are printed beside per-year skill, and
no year-on-year comparison should be read without them.

```bash
PYTHONPATH=. python handout/validate_qc.py
PYTHONPATH=. python handout/run_extended_cv.py block --years=2002-2013
```

## Open

Recovering 2014–2025 needs the discontinuities corrected, not filtered. The
natural route is a per-station-segment offset: model8 already fits a
ridge-regularised per-station offset, so splitting a station at a detected
break into two calibration segments fits its existing architecture. That
offset is predicted *from statics* so it can generalise to uninstrumented
sites, and statics do not change at a break — so the segment term would have
to be a training-time nuisance parameter, not part of the shipped readout.
Untested.

---
<!-- NAV -->
[← In-situ networks](insitu_networks.md) · [Index](../README.md) · [Downscaling to 30 m →](downscale.py.md)
<!-- /NAV -->
