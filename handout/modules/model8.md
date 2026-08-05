# `model8`: model7 + SLGA soil — the process model at ML parity

<!-- NAV -->
[← model7 · Process model](model7.md) · [Index](../README.md) · [Blocked validation →](blocked_validation.md)
<!-- /NAV -->

Source: [`../../emt/model8/model.py`](../../emt/model8/model.py)

The same calibrated bucket water balance as [`model7`](model7.md) — same five
parameters, same SILO-only forcing, same two-stage fit — with a configured
stack on top, each piece validated under blocked *and* station-out
cross-validation ([blocked validation](blocked_validation.md)):

1. **SLGA root-zone soil** (clay, sand, AWC, bulk density) joins terrain in
   the ridge offset stage — the change that first brought the process track
   to ML parity;
2. **the aridity normal** (mean P/PET from the SILO forcing) joins the
   statics — the level channel that transfers *across* climates;
3. **SLGA AWC as per-station bucket capacity** — higher-capacity soils
   genuinely hold and read out more water;
4. **stratified training weights** (aridity-tertile × block) — so ten
   clustered Yanco stations no longer outvote three scattered dry M-sites.

| 37 stations, 2006–2010 | model7 | **model8 (shipped)** | m8 soil+terrain only | model6 |
|---|---|---|---|---|
| Station-out pooled NSE / r | +0.18 / 0.43 | **+0.41 / 0.64** | +0.40 / 0.64 | +0.38 / 0.62¹ |
| Station-out median stn NSE | −0.03 | +0.13 | **+0.22** | −0.19¹ |
| Blocked pooled NSE | — | **+0.32** | +0.22 | +0.36² |
| Blocked median stn NSE | — | **+0.07** | −0.18 | −0.21² |
| Blocked median block NSE | — | **+0.25** | +0.25 | +0.09² |

¹ published 36-station reference · ² same-rows blocked run (see
[blocked validation](blocked_validation.md)). The bucket supplies the
temporal signal (median station r 0.83, unchanged throughout); soil supplies
the between-site level within a climate; aridity + capacity + weights carry
the level *across* climates. The one metric the pre-stack configuration wins
is the station-out median (+0.22 vs +0.13, the interpolation view — the
aridity term overshoots the best-behaved station of a held-out cluster); the
shipped configuration wins everything else in both harnesses, and the blocked
column is the honest one for a national product.

That both tracks converge on soil — the ML track through [feature
importance](model6.md#feature-importance), the process track through what
unlocks its level ranking — is strong independent evidence the signal is real
and physical, not an artefact of either method.

The station-out figures below are the pre-stack configuration's (the
published reference); the shipped configuration's station-out figures are on
the [blocked-validation page](blocked_validation.md#the-full-stack-at-station-out-in-full):

![model8 results](../figures/model8_results.png)

The shipped fit's offset coefficients (%-per-training-sd) remain textbook
physics, which is what makes them credible at held-out sites: **clay +1.86**
(finer soils hold more water), **sand −1.77**, bulk density −0.44, TWI +0.69,
slope −0.58, elevation −0.35, **aridity +0.61** (wetter climate → wetter
site) — with AWC acting through the bucket capacity (training-mean 10.9 %
normalises the ratio) rather than its (small, −0.42) offset.

Held-out predicted-vs-observed time series for every station, pre-stack
reference ([`plot_model8_per_station.py`](../plot_model8_per_station.py)):

![model8 per-station held-out time series](../figures/model8_per_station.png)

## What was tested along the way

* **AWC as capacity, alone** — negligible (station-out +0.16 vs +0.15 pooled;
  blocked +0.24 with a *worse* station-median), because AWC barely varies
  across these 37 stations (10.9 ± 0.7 %). It earns its place only **stacked**
  with aridity + weights, where the combination is the best transfer
  configuration measured — the story is on the
  [blocked-validation page](blocked_validation.md).
* **Soil offsets without terrain** scored the same pooled (+0.41) but much
  worse per-station (median NSE +0.05 vs +0.22); the terrain trio earns its
  place alongside soil.
* **Climate normals under station-out** *reduce* skill (the
  [model7 negative result](model7.md#per-station-terrain-offsets-two-stage-ridge),
  which replicates on model8) — aridity is in the shipped statics for its
  blocked-transfer gain, accepting the station-out median cost. The
  [reconciliation](blocked_validation.md#reconciling-with-model7s-rejection-of-climate-normals)
  carries both sides.

## Run it for any date

The fitted model ships in the repo (`data/models/model8.joblib`, 6 KB — it is
13 numbers plus the standardisation constants), and
[`emt/model8/predict.py`](../../emt/model8/predict.py) applies it **anywhere in
Australia, for any day**, as a point series or a 30 m map:

```bash
# daily series at a point, any date range
PYTHONPATH=. python -m emt.model8.predict \
    --lat -35.05 --lon 147.5 --start 2025-06-01 --end 2025-06-10

# 30 m map for one day
PYTHONPATH=. python -m emt.model8.predict \
    --bbox 147.30 -35.52 147.62 -35.10 --date 2024-09-15
```

```python
from emt.model8.predict import predict_point, predict_map
s  = predict_point(lat=-35.05, lon=147.5, start="2024-01-01", end="2024-12-31")
ds = predict_map(bbox=(147.30, -35.52, 147.62, -35.10), day="2024-09-15")
ds["sm_pred"]        # 30 m root-zone soil moisture (%)
```

**Why any date works without retraining.** model8 has no lookback *features* —
it has a *state*. To predict day D the bucket is run forward from a spin-up
start to D on SILO rain/PET, so 2006 and last week cost the same and need no
SMIPS at all (unlike [`emt/predict.py`](predict.py.md), which must assemble the
SMIPS lookback for the day). At inference the tool supplies the full stack's
inputs itself: the **aridity normal** is computed from the SILO series it
fetches anyway, and the **capacity ratio** from the SLGA AWC raster over the
fitted training-mean. A station cross-check confirms the inference path
reproduces the training path to machine precision (max |Δ| 0.0 % at K5,
capacity and aridity engaged).

**The spin-up, precisely.** Simulation starts on **1 January, two calendar
years before your start date** — so a June 2025 request simulates from January
2023, about 2.4 years of lead-in. That window exists only to wash out the
arbitrary initial condition (`S = 0.5 · smax`); it is **not** a limit on which
dates you can predict, and it does not need in-situ data. Its only real cost is
fetch time on a cold run, since in map mode every forcing cell pulls that many
years of SILO.

Two years is deliberately generous. Holding the forcing fixed and varying only
the spin-up, at one Murrumbidgee point for 2025-06-01:

| Spin-up | 0.25 yr | 0.5 yr | 1 yr | 2 yr | 4 yr | 10 yr |
|---|---|---|---|---|---|---|
| Storage (mm) | 43.28 | 40.68 | 40.68 | 40.68 | 40.68 | 40.68 |
| Predicted VWC (%) | 17.724 | 17.473 | 17.473 | 17.473 | 17.473 | 17.473 |

Everything from **six months** out is identical to a ten-year spin-up; only the
three-month run drifts (+0.25 %). That is what the fitted recession predicts —
the table was measured with the pre-stack fit (`k = 0.0073/day`, 137-day
e-folding); the shipped full-stack fit has `k = 0.0065/day` (154 days), so the
default keeps roughly a 4× margin. Lower `SPINUP_YEARS` to 1 in
[`predict.py`](../../emt/model8/predict.py) for faster cold starts; it still
leaves a 2× margin.

The map is a genuine downscaling of the same kind the repo does elsewhere: the
**water balance runs on the ~5 km SILO forcing grid** — the scale at which
weather actually varies, with each cell's capacity set by its own SLGA AWC —
and the **30 m structure comes from the per-pixel soil, terrain and aridity
offsets**. Outputs are a GeoTIFF plus a quick-look PNG in the
AOI's PaddockTS query dir; `--step-deg` tunes the forcing grid, `-o` writes an
extra copy.

Kyeamba on 2024-09-15 — 1.6 M pixels from the command above, a date fourteen
years past the training period:

![model8 30 m field over Kyeamba, 2024-09-15](../figures/model8_predict_example.png)

Drainage lines and valley floors sit wet, ridges and the steep north-western
block dry — the TWI and slope terms doing visible work, with SLGA map-unit
boundaries showing as the broader patches. Costs on a first run are the 30
cached SILO cell downloads (~2 years each) plus terrain and soil for the AOI;
later days over the same area reuse all of it.

## Data & running

Retraining (not needed to predict) reads the same inputs as model7 plus
`data/process_soil_statics.csv` and `data/process_climate_statics.csv`, all
built by
[`emt/model7/build.py`](../../emt/model7/build.py) — the soil step needs a
**TERN API key** in `~/.config/PaddockTS.json` and is skipped with a notice
otherwise (model7 remains fully runnable without it).

```bash
PYTHONPATH=. python -m emt.model7.build     # target + forcing + climate + terrain + soil
PYTHONPATH=. python -m emt.model8.model data/process_target_2006_2010.csv
```

## Spatial transfer — read the headline with the blocked numbers

Station-out folds leave a held-out station's cluster neighbours (same ~5 km
forcing cells, same soil map units) in training — an *interpolation*
estimate. Holding out whole spatially independent blocks instead (9 folds:
Yanco, Kyeamba, Adelong, each M-site) is what motivated the shipped stack:
the pre-stack configuration drops to **pooled +0.22, station-median −0.18**
there (much of its between-site level skill was neighbour leakage), while
the shipped configuration holds **pooled +0.32, block-median +0.25,
station-median +0.07**. What remains fails at the climate-envelope *edges*
(the driest station, the wettest cluster) as pure level error with r intact —
a data limitation, not a modelling one. Full experiment, per-block tables and
figures: [Blocked validation](blocked_validation.md).

> **Successor under evaluation.** [model9](model9.md) replaces model8's two
> *global* readout constants with per-site soil hydraulic limits from texture,
> and scores better on blocked transfer (+0.35 vs +0.32 pooled, 8/9 blocks
> positive vs 7/9) with one fewer parameter — but worse at station-out, with
> the gain concentrated in a single block. model8 remains the default.

## Status

model8 is the **recommended model of the process track** ([model7](model7.md)
stays as its covariate-free foundation) and a full peer of
[model6](model6.md): station-out parity pooled, clearly better transfer, 14
interpretable parameters, no training table, and no dependence on SMIPS at
all. Its caveats are the shared ones — no temporal hold-out, no validation
outside the Murrumbidgee — plus the harness distinction above: the
station-out skill is the interpolation figure, the
[blocked](blocked_validation.md) +0.32 the transfer figure. The two tracks
fail differently (see the blocked page's per-block table), which is what makes
the hybrid flagged in model7 (bucket storage as an ML feature, or SMIPS
assimilated into the bucket) the natural next step.

---
<!-- NAV -->
[← model7 · Process model](model7.md) · [Index](../README.md) · [Blocked validation →](blocked_validation.md)
<!-- /NAV -->
