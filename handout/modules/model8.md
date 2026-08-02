# `model8`: model7 + SLGA soil — the process model at ML parity

<!-- NAV -->
[← model7 · Process model](model7.md) · [Index](../README.md) · [Downscaling to 30 m →](downscale.py.md)
<!-- /NAV -->

Source: [`../../emt/model8/model.py`](../../emt/model8/model.py)

The same calibrated bucket water balance as [`model7`](model7.md) — same five
parameters, same SILO-only forcing, same two-stage fit — with exactly one
change: **SLGA root-zone soil** (clay, sand, AWC, bulk density) joins terrain
in the ridge offset stage. One line of configuration, and the pooled
leave-site-out skill **doubles to parity with the ML models**:

| Metric (37 stations, 2006–2010, leave-site-out) | model7 | **model8** | model6 (36 stn) |
|---|---|---|---|
| Pooled NSE / r | +0.18 / 0.43 | **+0.40 / 0.64** | +0.38 / 0.62 |
| Median per-station NSE | −0.03 | **+0.22** | −0.19 |
| Per-station NSE > 0 | 17/37 | **19/37** | 16/36 |
| Median per-station r | 0.83 | 0.83 | 0.81 |
| Median per-station \|bias\| | 3.98 % | **3.64 %** | 3.85 % |

(Same caveat as model7: the model6 column is the published 36-station
reference, not a same-rows rerun.) Read the table as the completion of
model7's decomposition: the bucket supplies the temporal signal (r 0.83,
unchanged), the soil maps supply the between-site level ranking — and with
both in place a **13-parameter interpretable model matches a tuned
gradient-boosting model pooled (+0.40 vs +0.38) while beating it clearly
per-station (median NSE +0.22 vs −0.19)**.

This is the process-side confirmation of [model6's feature
importance](model6.md#feature-importance), reached from an independent
direction: there the ML model, given 25 covariates, put SLGA soil on top; here
adding the same four soil layers to a physics readout recovers the same skill.
The level problem was information-limited, and the information is soil.

![model8 results](../figures/model8_results.png)

The fitted offset coefficients (%-per-training-sd, ridge α ≈ 22) are textbook
soil physics, which is what makes them credible at held-out sites: **clay
+1.43** (finer soils hold more water), **sand −1.41**, bulk density −0.40,
TWI +0.67, slope −0.71 — and AWC ≈ 0 (see below).

Held-out predicted-vs-observed time series for every station
([`plot_model8_per_station.py`](../plot_model8_per_station.py)):

![model8 per-station held-out time series](../figures/model8_per_station.png)

## What was tested and not defaulted

* **AWC as per-station bucket capacity** — the physically-motivated route
  (`capacity=` on the estimator: SLGA available water capacity scales `smax`,
  so higher-capacity soils genuinely sit wetter). Its LOSO gain is negligible
  (+0.16 vs +0.15 pooled alone; no gain on top of the offsets) because AWC
  barely varies across these 37 stations (10.9 ± 0.7 %). The machinery stays
  in [`model7/model.py`](../../emt/model7/model.py) for regions with real AWC
  contrast.
* **Soil offsets without terrain** scored the same pooled (+0.41) but much
  worse per-station (median NSE +0.05 vs +0.22); the terrain trio earns its
  place alongside soil.

## Data & running

model8 reads the same inputs as model7 plus
`data/process_soil_statics.csv`, all built by
[`emt/model7/build.py`](../../emt/model7/build.py) — the soil step needs a
**TERN API key** in `~/.config/PaddockTS.json` and is skipped with a notice
otherwise (model7 remains fully runnable without it).

```bash
PYTHONPATH=. python -m emt.model7.build     # target + forcing + terrain + soil
PYTHONPATH=. python -m emt.model8.model data/process_target_2006_2010.csv
```

## Status

model8 is the recommended **process-model** configuration; model7 stays as the
covariate-free baseline that isolates what the bucket alone explains. Against
the ML track it is a genuine alternative rather than a diagnostic: pooled
parity with [model6](model6.md), better per-station medians, 13 interpretable
parameters, and no training table — but it has no SMIPS consistency and, like
every model here, no temporal hold-out and no validation outside the
Murrumbidgee. The natural next step is the hybrid flagged in model7: bucket
storage as a model6 feature, or SMIPS assimilated into the bucket.

---
<!-- NAV -->
[← model7 · Process model](model7.md) · [Index](../README.md) · [Downscaling to 30 m →](downscale.py.md)
<!-- /NAV -->
