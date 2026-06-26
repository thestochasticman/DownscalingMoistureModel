# Downscaling SMIPS soil moisture to 30 m — approach & results

**Goal.** Take TERN's **SMIPS** profile soil-water field (`TotalBucket`, mm,
~1 km daily) and **downscale it to ~30 m** — the resolution of the Copernicus
DEM — by learning, from in-situ ground truth, how fine-scale terrain redistributes
moisture within each coarse cell.

This handout walks the pipeline module by module, then shows what the rebuilt
model actually does (and where it falls short). Each module has a short write-up
linking to its source; the figures are reproducible from
[`plot_results.py`](plot_results.py).

---

## The pipeline

| stage | module | what it produces |
|---|---|---|
| 1 — ground truth | [`oznet.py`](modules/oznet.py.md) | daily root-zone (0–90 cm) soil moisture per OzNet station — the **target** |
| 2 — study areas | [`queries.py`](modules/queries.py.md) | PaddockTS `Query` windows (per-station + focus catchments) |
| 3a — coarse input | [`smips.py`](modules/smips.py.md) | raw SMIPS `TotalBucket` cube (mm) — **the field being downscaled** |
| 3b — fine predictors | [`covariates.py`](modules/covariates.py.md) | 30 m terrain stack (elevation, slope, northness/eastness, TWI, HLI, accumulation) |
| 4 — training table | [`features.py`](modules/features.py.md) | one row per station-day: target + SMIPS + terrain + seasonality |
| 5 — model | [`model.py`](modules/model.py.md) | Random Forest + **leave-site-out** cross-validation |
| 6 — downscale | [`downscale.py`](modules/downscale.py.md) | apply the model per pixel → a **30 m soil-moisture field** |

The model:

```
sm_rootzone_pct  ~  smips_totalbucket + terrain(...) + doy_sin + doy_cos
```

`lat`/`lon` are intentionally excluded as features (they would leak station
identity); `station` is used only as the spatial CV group.

---

## A data-quality fix that mattered

While validating the pipeline we found TERN's GeoServer **WCS resamples every
request to the requested bounding box**, so a station's SMIPS value depended on
*how much area we asked for*. Station K6 came back as **43.7 / 49.6 / 61.6 mm**
for three different windows — same place, same day. The fix
([`smips.py` → `snap_bbox`](modules/smips.py.md)) aligns every request to the
native ~1 km grid, making sampling window-independent and returning the true
native pixel.

![SMIPS correction](figures/smips_correction.png)

**(a)** Old vs new SMIPS. Three stations sit on the 1:1 line — their old window
already happened to land on the right cell. **Only K6 moved.** **(b)** K6 shifted
**+15.4 mm** on average; the others ≈0. **(c)** Target vs corrected SMIPS: the
four sites form near-horizontal bands — SMIPS varies a lot *within* a site but
barely separates the sites' moisture *levels*. That detail drives the result
below.

---

## What the model does — and where it breaks

The headline metric is **leave-site-out CV**: train on every station but one,
predict the held-out one (the honest test, since at inference the model meets
unseen locations).

![Leave-site-out CV](figures/leave_site_out_cv.png)

- **(a)** Each held-out site is a **tight diagonal sliver** (day-to-day shape is
  tracked) but **offset from the 1:1 line**. K12 is extreme: observed ~33%,
  predicted ~28%.
- **(b)** In numbers: within-site `r ≈ 0.9` everywhere, but **bias** is large
  (K10 +3.5, K12 −4.6). High correlation **+** large bias = the negative pooled
  `r²`.
- **(c)** Feature importance: terrain ≈ 88%, **SMIPS ≈ 0.006**. With `lat`/`lon`
  excluded, the forest has only terrain to set a site's level — and terrain
  doesn't encode it here.

![Per-site time series](figures/per_site_timeseries.png)

Per site, the prediction **follows the temporal shape** but sits at the **wrong
level** — clearest at K12, where the prediction floats well below the
observations.

### Diagnosis
The pipeline is correct; the CV is degenerate because **four stations packed into
~one SMIPS pixel** give the model no cross-site signal. It learns each site's
*dynamics* but cannot place an unseen site's *absolute level*. This is a
data-scope limit, not a code bug.

### Next step
Expand the training table across the catchment — **Yanco + Kyeamba + Adelong**,
more of 2006–2010 — so SMIPS actually varies between training sites. The
[cluster-fetch optimisation](modules/features.py.md) makes that broader,
multi-year build cheap.

---

## Scaling up fixed it — Yanco + Kyeamba + Adelong, 2006–2010

Rebuilding across all three clustered sites — **30 stations, 40,590 station-days**
(SMIPS fetched as just **3 site cubes** via cluster-fetch) — confirms the
approach. The three sites are ~100–150 km apart, so SMIPS finally varies between
training sites, and the model uses it.

![Catchment results](figures/catchment_results.png)

- **(a)** SMIPS now spans the sites (Adelong ≈53, Kyeamba ≈38, Yanco ≈26 mm) —
  the cross-site signal that was missing at Kyeamba alone.
- **(b)** Leave-site-out fit over 30 held-out stations: **pooled r = 0.54,
  r² = +0.16** (was −1.16) — genuinely better than predicting the mean.
- **(c)** Feature importance flips: **SMIPS goes from least-used (0.006) to the
  single most-used feature (0.34)**, then slope (0.27). The downscaling premise —
  SMIPS sets the level, terrain refines it — now holds.
- **(d)** What remains: a per-station **level bias** (e.g. K12 −16%, A5 +11%).
  Bias-removed error per site is good (ubRMSE ≈ 3–4%), so the model tracks
  *dynamics* well; the residual offset is local soil/texture that SMIPS + terrain
  don't capture — a candidate for a soil covariate or per-site effect later.

| metric | Kyeamba-only (4 stn) | catchment (30 stn) |
|---|---|---|
| pooled leave-site-out `r` | −0.45 | **+0.54** |
| pooled leave-site-out `r²` | −1.16 | **+0.16** |
| SMIPS feature importance | 0.006 | **0.34** |
| corr(target, SMIPS) | 0.25 | 0.53 |

---

## Stage 6 — the 30 m product over Yanco

Applying the model per pixel gives the actual deliverable: a **30 m root-zone
soil-moisture map**. To keep the test honest, the model here is trained on
**Kyeamba + Adelong only and Yanco is fully held out** — so the map *and* its
validation are genuine transfer to a catchment the model never saw (the real
downscaling use case). Date: **2008-07-31** (all 12 Yanco stations report).

![Downscaled Yanco](figures/downscale_yanco.png)

- **(a) → (b)/(c)** The blocky ~1 km SMIPS input is sharpened to a 30 m field;
  the zoom **(c)** reveals dendritic drainage structure the coarse input cannot
  resolve. This terrain-driven detail is the entire point of downscaling.
- **(d)** Held-out validation at the 12 Yanco stations:

  | metric | value | reading |
  |---|---|---|
  | RMSE | 11.5 % | dominated by bias |
  | **ubRMSE** | **2.4 %** | bias-removed error is small — the spatial pattern transfers |
  | bias | **+11.3 %** | model trained on wetter uplands over-predicts the semi-arid Yanco plains |
  | r | 0.41 | moderate across 12 stations |

**Interpretation.** Transferring to an unseen catchment, the *relative* structure
is captured well (ubRMSE 2.4 %) but the *absolute level* carries a large regional
bias. This is the same per-station level bias seen in Stage 5, now in its
starkest form (whole region held out): SMIPS + terrain do not encode the local
soil/climate baseline that sets absolute moisture. → see **Future work**.

> **Metrics convention used throughout.** `RMSE` total error; `ubRMSE` =
> √(RMSE²−bias²), the bias-removed error (the standard soil-moisture skill
> number); `bias` = mean(pred−obs); `r` Pearson; `r²` = 1−SS_res/SS_tot (can go
> negative when bias dominates). Headline *generalisation* skill is always the
> **leave-site-out / leave-region-out** value — never an in-sample fit.

---

## Future work

1. **SLGA soil covariate (highest priority).** The dominant remaining error is an
   absolute *level* bias on transfer (Stage 6: +11 % over Yanco). Static soil
   properties — clay/sand fraction, bulk density from the
   [Soil and Landscape Grid of Australia](https://www.clw.csiro.au/aclep/soilandlandscapegrid/)
   (SLGA, ~90 m) — most plausibly encode the baseline that SMIPS + terrain miss.
   They would enter as additional static per-pixel features in
   [`features.py`](modules/features.py.md) / [`downscale.py`](modules/downscale.py.md);
   no architecture change. *Not implemented yet.*
2. **Mass conservation.** Constrain the 30 m field to aggregate back to a coarse
   reference per cell (decompose into cell-mean + terrain anomaly, rebase the
   mean). Needs a coarse reference in % units — see
   [`downscale.py` doc](modules/downscale.py.md#documented-future-work-not-yet-implemented).
3. **Bias correction / mixed effects.** A per-site offset (or quantile mapping to
   SMIPS climatology) would absorb the regional level bias directly.

---

## Reproduce the figures

From the repo root (with the `paddockts` conda env active so `PaddockTS` and the
training-data caches are available):

```bash
PYTHONPATH=. python handout/plot_results.py
```

This loads (or rebuilds) the Kyeamba 2020 Jun–Jul training table, reconstructs
the pre-fix SMIPS values to show the correction, runs the leave-site-out CV, and
overwrites the three PNGs in [`figures/`](figures/).

The catchment and Stage 6 figures come from the expanded table
(`data/train_catchment_2006_2010.csv`):

```bash
PYTHONPATH=. python handout/plot_catchment.py    # catchment_results.png
PYTHONPATH=. python handout/plot_downscale.py    # downscale_yanco.png (30 m map)
```

> The module write-ups in [`modules/`](modules/) summarise each source file; the
> code of record lives in [`../emt/`](../emt/).
