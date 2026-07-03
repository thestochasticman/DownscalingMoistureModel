# `model5`: model4 with smoothed soil (a documented tradeoff, not an improvement)

<!-- NAV -->
[← model4 · Climatology + soil](model4.md) · [Index](../README.md) · [model6 · Antecedent meteorology →](model6.md)
<!-- /NAV -->

Source: [`../../emt/model5/model.py`](../../emt/model5/model.py)

Identical estimator and features to [`model4`](model4.md); the only change is that
the SLGA soil rasters are Gaussian-blurred ([`slga.smooth_soil`](slga.py.md)) at
both training-point sampling and map inference. It was built to test whether
smoothing removes the blocky soil map-unit boundaries that mar the model4
downscaled field.

**Result: a genuine tradeoff, and model5 is not recommended.** Smoothing does
clean the map, but at a steep, monotonic cost to cross-validated skill.

## The map improves…

On the leave-Yanco-out demonstration
([`plot_downscale_model5.py`](../plot_downscale_model5.py), 2008-07-31):

| Single-date, 12 stations | model4 (raw soil) | model5 (σ=2 px) |
|---|---|---|
| ubRMSE | 4.90 % | **3.05 %** |
| r | 0.30 | **0.39** |
| bias | +9.01 % | +10.37 % |

The 30 m field is visibly less speckled and the spatial pattern correlates better
with the stations.

## …but the cross-validated skill falls, with no sweet spot

A controlled soil-blur sweep (σ=0 exactly reproduces model4, confirming the
pipeline):

| soil blur σ (px) | LOSO NSE | r | per-station NSE > 0 |
|---|---|---|---|
| **0 (= model4)** | **0.354** | 0.623 | 14/30 |
| 1 | 0.286 | 0.613 | 13/30 |
| 2 (model5 default) | 0.126 | 0.501 | 12/30 |
| 3 | 0.058 | 0.455 | 14/30 |

Even σ=1 — a barely perceptible blur — costs 0.068 NSE, and every further step
degrades it. There is no configuration that keeps the skill and cleans the map.

## Why they cannot be separated

The soil covariate is the **2nd most important predictor**, carrying real
between-station level discrimination. The *same* sharp soil detail that makes the
downscaled map blocky is what tells the model how wet one station is relative to
another. Smoothing removes it from both places at once, so the map artifacts and
the tabular skill are two faces of one signal — you cannot blur one without the
other. (Mechanism: with `max_leaf_nodes=3` the model splits soil on ~1–2
thresholds, so even ~1 % point-value shifts reassign a station's predicted level;
the per-station-window blur is also slightly inconsistent between stations, so the
LOSO drop is an upper bound — but the direction is robust and expected.)

## Status

[`model6`](model6.md) is the recommended model; `model5` is **not** — it is
retained to document the texture-vs-skill tension and, at most, to produce a
presentation-quality map where lower per-station skill is acceptable.

---
<!-- NAV -->
[← model4 · Climatology + soil](model4.md) · [Index](../README.md) · [model6 · Antecedent meteorology →](model6.md)
<!-- /NAV -->
