# Temporal validation — which kind of generalisation is actually hard

<!-- NAV -->
[← model9 · Pedotransfer readout](model9.md) · [Index](../README.md) · [Downscaling to 30 m →](downscale.py.md)
<!-- /NAV -->

Source: [`../run_blocked_cv.py`](../run_blocked_cv.py) (`@year` folds) ·
figure: [`../plot_temporal_validation.py`](../plot_temporal_validation.py)

Every skill number elsewhere in this handout holds out **places**. None held out
**time** — and 2006–2010 is not five interchangeable years. It spans the tail of
the Millennium Drought and its break:

| year | rainfall (mm, per-station mean) | mean observed moisture |
|---|---|---|
| 2006 | 252 | 20.9 % |
| 2007 | 538 | 22.3 % |
| 2008 | 470 | 21.6 % |
| 2009 | 411 | 21.0 % |
| **2010** | **951** | **27.6 %** |

A **3.8× range** in annual rainfall, with 2010 wetter than the other four
combined would suggest. That matters because [model8](model8.md) and
[model9](model9.md) are sold on running forward to *any* date — 2025 included —
and every model here had seen all five years in training.

This page holds each year out whole (`@year` folds in the same harness) for
[model6](model6.md), model8 and model9.

## Results

| pooled NSE | leave-one-YEAR-out | leave-one-STATION-out | leave-one-BLOCK-out |
|---|---|---|---|
| model6 | **+0.771** | — ¹ | +0.355 |
| model8 (shipped) | +0.475 | +0.408 | +0.322 |
| model9 | +0.447 | +0.397 | +0.347 |

¹ not run — model6's station-out reference is the published 36-station figure
(+0.377), not a same-rows rerun on this table.

Skill of each held-out year:

| year | rain | model6 NSE / bias | model8 NSE / bias | model9 NSE |
|---|---|---|---|---|
| 2006 | 252 | +0.790 / −0.31 | +0.440 / −0.75 | +0.440 |
| 2007 | 538 | +0.776 / −0.75 | +0.492 / +0.41 | +0.461 |
| 2008 | 470 | +0.747 / +1.30 | +0.466 / +0.64 | +0.458 |
| 2009 | 411 | +0.827 / +0.85 | +0.470 / +0.73 | +0.428 |
| **2010** | **951** | **+0.635 / −2.39** | **+0.269 / −2.23** | **+0.208** |

![temporal validation](../figures/temporal_validation.png)

## Findings

**1 · Time is the easy axis; space is the hard one.** For the *same* model8,
pooled NSE runs +0.475 (year) → +0.408 (station) → +0.322 (block). Holding out
a year leaves every station in training, so the model already knows each site's
*level* and only has to cope with unfamiliar weather — which is precisely the
part the bucket does well. Holding out a district removes the level information
too. **The generalisation this project actually needs is the one that scores
worst.**

**2 · Only the wet extreme breaks — and it is the same failure as in space.**
Four of the five years score within 0.05 NSE of each other for every model.
2010 collapses: model8 +0.47 → +0.27, model9 +0.46 → +0.21, model6 +0.78 →
+0.64. And it collapses *in the same direction* as the spatial failures — a
negative bias, predicting too dry, on the year that ran wetter than anything
in calibration. The driest year (2006, 252 mm) is not hard at all. Wet extremes
break these models; dry extremes do not.

**3 · The prediction that the process track would transfer better across time
was wrong.** The reasoning was that model6 carries lookback *features* while
model8/9 carry a *state*, so the process models should ride a regime shift
better. The opposite happened: **model6 wins the year folds outright**, +0.771
against +0.475, with a median per-station NSE of +0.45 and a median |bias| of
0.47 % — its best result anywhere in this handout.

**4 · But that model6 number is not a national-product number.** Under year
folds nothing spatial is withheld, so model6's 25 features can identify each
station and learn its level directly. +0.771 is the score for *"we have a
sensor at this site and want another year of record"* — a real use case, and
the one model6 is best at. The same model scores +0.355 pooled and **+0.09
block-median** when the site is genuinely new. The three harnesses reorder the
two tracks:

| harness | winner | by |
|---|---|---|
| leave-one-year-out | model6 | +0.77 vs +0.48 |
| leave-one-block-out, pooled | model6 | +0.36 vs +0.32 |
| leave-one-block-out, block-median | **model8** | +0.25 vs +0.09 |

Which is the whole argument for stating the harness beside every number.

## What this does and does not establish

**It does** establish that a regime shift of this size costs roughly 0.2 NSE at
the wet end and nothing measurable elsewhere, and that model8's "any date"
claim is sound for ordinary years while optimistic for exceptionally wet ones.

**It does not** isolate time cleanly. Both tracks leak across the fold boundary
by construction: model6's 365-day lookback features on a held-out January draw
on the previous (training) December, and the process models' bucket state is
simulated *continuously* through the training years into the held-out one. So
this measures **"a year whose weather was not in the calibration"**, not "a year
in isolation". Removing that would need a buffer of a year on each side of every
fold, which on a five-year record leaves almost nothing to test.

**Not attempted:** the combined block × year design (new place *and* new
regime), which is the honest worst case. With 9 blocks × 5 years the cells get
thin, but it is the natural next experiment.

## Reproduce

```bash
PYTHONPATH=. python handout/run_blocked_cv.py m8capaw@year m9@year m6@year
PYTHONPATH=. python handout/plot_temporal_validation.py
```

The `@year` suffix works on any configuration in
[`run_blocked_cv.py`](../run_blocked_cv.py), alongside `@station` and the
default block folds. Out-of-fold predictions land in
`data/model{6,8,9}_yearcv*_predictions.csv`.

---
<!-- NAV -->
[← model9 · Pedotransfer readout](model9.md) · [Index](../README.md) · [Downscaling to 30 m →](downscale.py.md)
<!-- /NAV -->
