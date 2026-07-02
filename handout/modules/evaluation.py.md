# `evaluation.py`: metrics and leave-site-out cross-validation

<!-- NAV -->
[← Training table](features.py.md) · [Index](../README.md) · [model1 · Random Forest →](model1.md)
<!-- /NAV -->

Source: [`../../emt/evaluation.py`](../../emt/evaluation.py)

The shared scoring harness. It is estimator-agnostic: a model package
([`model1`](model1.md), [`model2`](model2.md)) supplies a feature list and an
estimator factory, and every model is graded here by the same procedure. This is
where the reported skill numbers come from.

| Function | Role |
|---|---|
| `metrics(y_true, y_pred)` | RMSE, ubRMSE, bias, r, NSE (`= r2`), n for one set of paired values |
| `leave_site_out_cv(table, features, estimator_factory, group_col="station")` | Spatial cross-validation; returns pooled and per-site scores plus the out-of-fold predictions |

## How the cross-validation works

The cross-validation is **leave-one-site-out (LOSO)**: a *spatial* split in which
the hold-out unit is a whole **station**, not a random subset of rows and not a
time span. It is implemented with scikit-learn's `LeaveOneGroupOut`, grouping on
the `station` column.

For a table with *N* stations, the procedure runs *N* folds. In each fold:

1. **The held-out unit is one station** — *all* of that station's station-day
   rows form the test set.
2. **The training set is every other station** — all rows from the remaining
   *N* − 1 stations.
3. **A fresh estimator is built per fold** (`estimator_factory()`), so no fitted
   state leaks between folds.
4. The estimator is fit on the training stations and predicts the held-out
   station's rows.

After all *N* folds, every row in the table has a prediction produced by a model
that **never saw that station** during training. These out-of-fold predictions
are what all reported metrics are computed from.

```
Fold 1:  train on  [S2 S3 … SN]   →  predict  S1
Fold 2:  train on  [S1 S3 … SN]   →  predict  S2
   ⋮
Fold N:  train on  [S1 S2 … S(N-1)] → predict  SN
```

### Why hold out by station

Inference occurs at **unobserved locations** — the model is ultimately applied to
every 30 m pixel of an area, none of which has an in-situ sensor. Holding out a
whole station measures exactly that: can the model predict a place it has never
seen? A random row split would leak each station's own level into training (other
days from the same sensor), inflating apparent skill and not testing the
generalisation that matters. Station coordinates (`lat`/`lon`) are excluded from
the features for the same reason; `station` is used *only* as the grouping
variable, never as a predictor (see [`model1`](model1.md#design-decisions)).

## Pooled vs per-station scoring

The same out-of-fold predictions are scored two ways, and the two differ
substantially — both are returned and both are reported.

- **Pooled** (`result["pooled"]`): `metrics()` over *all* station-days at once.
  Because observed values span dry (Yanco) to wet (Adelong) sites, the
  between-site variance enters the NSE denominator, making the pooled figure
  relatively **lenient** — it credits the model for separating dry sites from wet
  ones. Catchment result: pooled NSE +0.15.
- **Per-station** (`result["per_site"]`): `metrics()` applied *within each
  station's own series*, one row per station. This removes the between-site spread
  and is the more **exacting** temporal test. Catchment result: per-station NSE
  negative at 23/30 stations (median −0.56), even though dynamics are tracked well
  (median r 0.75, ubRMSE 3.9%).

The gap between the two is the central result of the project: the model reproduces
**dynamics** but carries a per-station absolute-level **bias**. The per-station
figure is the one to weight (see the
[README](../README.md#per-station-performance)).

## Return value

`leave_site_out_cv` returns a dict:

| Key | Contents |
|---|---|
| `pooled` | `metrics()` over all out-of-fold predictions |
| `per_site` | DataFrame, one row per station, with that station's metrics |
| `predictions` | DataFrame `[station, time, target, pred]` — the raw out-of-fold predictions, for plotting |

## Metrics

`metrics()` returns the standard soil-moisture validation set. Non-finite pairs
are dropped first; fewer than two valid points returns NaNs.

| Metric | Definition |
|---|---|
| `rmse` | √mean((pred − obs)²) |
| `ubrmse` | √(RMSE² − bias²) — bias-removed RMSE, the standard soil-moisture skill statistic |
| `bias` | mean(pred − obs) |
| `r` | Pearson correlation |
| `nse` | Nash-Sutcliffe efficiency, 1 − Σ(pred − obs)² / Σ(obs − mean obs)². NSE = 1 perfect, > 0 more skilful than the observed mean, < 0 worse. Returned again as `r2`; the two are identical. |

## Caveat: no temporal hold-out

The split is **purely spatial**. Within the training stations, all days are used;
there is no forecasting into unseen time. LOSO therefore measures **spatial
transfer** (predicting a new location), not temporal extrapolation (predicting a
future date). The leave-region-out downscaling demonstration
([`downscale.py`](downscale.py.md)) is the same idea at coarser granularity — a
whole catchment is withheld instead of one station.

---

---
<!-- NAV -->
[← Training table](features.py.md) · [Index](../README.md) · [model1 · Random Forest →](model1.md)
<!-- /NAV -->
