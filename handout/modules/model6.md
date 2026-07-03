# `model6`: model4 + antecedent meteorology (recommended)

<!-- NAV -->
[← model5 · Soil smoothing](model5.md) · [Index](../README.md) · [Downscaling to 30 m →](downscale.py.md)
<!-- /NAV -->

Source: [`../../emt/model6/model.py`](../../emt/model6/model.py)

The same estimator and features as [`model4`](model4.md), extended with **SILO
trailing-window meteorology** ([`antecedent.py`](../../emt/antecedent.py)) — how
much rain has fallen and how the water balance and evaporative demand have run
over the **last week, month and year**:

```
model4 features + rain_7/30/365 + ppet_30/365 + vpd_30 + rain_365_anom
```

| Metric (36 stations, 2006–2010, leave-site-out) | model4 | **model6** |
|---|---|---|
| Pooled NSE / r | 0.368 / 0.628 | **0.393 / 0.644** |
| Median per-station \|bias\| | 3.36 % | **3.16 %** |
| Median per-station NSE | −0.01 | −0.03 |
| Per-station \|bias\| improved | — | **27/36 stations** |

| Function | Role |
|---|---|
| `build_estimator()` | Same regularised `HistGradientBoostingRegressor` as model4 |
| `ensure_features(table)` | model4 features (SMIPS climatology + soil) **+** antecedent meteorology |
| `fit` / `leave_site_out_cv` / `feature_importance` | As in the other model packages |

## Why antecedent meteorology

Soil moisture is set by how much water has recently arrived versus left. The
current SMIPS value and the SMIPS pixel-climatology capture the *state* and the
*long-term level*, but not the **recent accumulation** — chiefly the month-scale
**water balance** `P − PET` (rain minus potential evapotranspiration). Adding it
lowers the residual per-station level **bias** (the binding constraint) from
≈3.9 % to 3.16 %, for a pooled-NSE gain of +0.025. All the features are dynamic
per-pixel time series drawn from SILO (a national ≈5 km daily grid), so they are
computable at every 30 m pixel at inference and cannot memorise station identity
— the same leakage test the other features pass. SILO is fetched one year before
the study start so the 365-day window is complete at every training date.

## Which window matters — month, not year

Permutation importance is led by the **30-day** water balance `ppet_30`, with
`vpd_30` also contributing — the recent-month accumulation the SMIPS state does
not separate. The **year** window is more useful here than on the 30-station
catchment: `rain_365_anom` (this year's rain vs the pixel's normal — a drought
index) is the *second* antecedent feature on the 36-station set, because the dry,
drought-exposed western
[M-sites](../README.md#extending-coverage-regional-sites-30--36-stations) are
where multi-year rainfall deficit carries real signal. Correspondingly the gain
is larger on 36 stations (+0.025 pooled) than on the 30-station catchment
(+0.009), and per-station |bias| falls at 27 of 36 stations.

## Status

model6 is the **recommended** model — it supersedes [`model4`](model4.md) on the
36-station set (higher NSE and r, lower bias) while keeping the same estimator.
model4 remains the reference without the climate features. Inference needs the
SILO antecedent rasters over the AOI in addition to model4's climatology and
soil layers.

---
<!-- NAV -->
[← model5 · Soil smoothing](model5.md) · [Index](../README.md) · [Downscaling to 30 m →](downscale.py.md)
<!-- /NAV -->
