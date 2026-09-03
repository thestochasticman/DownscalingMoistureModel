# The neural-network track: MLP, Transformer, and the differentiable bucket

<!-- NAV -->
[← model10 · The hybrid](model10.md) · [Index](../README.md) · [In-situ networks →](insitu_networks.md)
<!-- /NAV -->

Source: [`../../emt/nn/`](../../emt/nn/) (package README:
[`emt/nn/README.md`](../../emt/nn/README.md)) ·
[`../plot_nn_results.py`](../plot_nn_results.py) ·
[`../plot_nn_per_station.py`](../plot_nn_per_station.py)

Three PyTorch models, run through the **same validation ladder** as models
1–10 (`emt/nn/cv.py` reproduces the station / block / year / block×year
folds; every number below is out-of-fold):

1. **nn-mlp** — a residual MLP on [model6](model6.md)'s exact feature set
   (SMIPS + lookback + terrain + soil + antecedent SILO). NSE\* loss
   (per-station variance-normalised squared error, Kratzert et al. 2019),
   AdamW + one-cycle LR, station-grouped early stopping, seed ensembling.
2. **nn-seq** — a Transformer encoder over the previous 365 days of SILO
   forcing plus a static token — **no SMIPS**, the same inputs as
   [model8](model8.md).
3. **nn-hybrid** — [model7](model7.md)'s bucket recurrence rewritten in
   torch (bit-identical to the numba loop) with a small MLP mapping station
   statics to per-station *deviations* of the five bucket parameters,
   sigmoid-bounded to model7's calibration ranges and zero-initialised — at
   step 0 it *is* model7, and every departure must earn its place on a
   station-grouped validation split. The water balance stays enforced; only
   its parameters are learned.

## Results — per-site (leave-station-out) and blocked

| 37 stations, 2006–2010 | nn-mlp | nn-seq | **nn-hybrid (quantile)** | model8 | **mean(hybrid, m8)** |
|---|---|---|---|---|---|
| Station-out pooled NSE / r | +0.38 / 0.62 | +0.36 / 0.61 | +0.35 / 0.63 | +0.41 / 0.64 | **+0.42 / 0.65** |
| Station-out stations NSE>0 | 17/37 | 15/37 | 18/37 | 20/37 | **20/37** |
| Station-out median stn NSE | −0.33 | −0.64 | −0.05 | +0.13 | **+0.15** |
| Blocked pooled NSE | — | — | **+0.354** | +0.322 | **+0.374** |
| Blocked median block NSE | — | — | **+0.30** | +0.25 | **+0.38** |
| Blocked blocks NSE>0 | — | — | 7/9 | 7/9 | 7/9 |

**Post-script — the combination that actually wins.** A learned gating net
over the bases (`emt/nn/stack.py`) is a documented *negative result*: it loses
to equal weighting under both designs (37 sites cannot teach a transferable
who-to-trust map). But widening the plain mean to every validated base beats
the 2-model mean above: **mean(hybrid, model8, model6) blocked = +0.401
pooled, block-median +0.38, median station +0.12, 21/37 positive** — the
repo's best blocked numbers — and mean(hybrid, model8, mlp, seq) station-out
= **+0.450** pooled. Diversity, not learned weighting, is the win.

The single-model story: the MLP matches model6 on identical features; the
Transformer matches the MLP *without SMIPS*; and the hybrid — the only one
whose physics is enforced rather than learned — is the **first model in the
repo to beat model8 on blocked transfer**, the honest test for a national
product. Averaging its out-of-fold predictions with model8's beats every
single model under **both** designs. M7, the station every model 1–10 fails
catastrophically (NSE −13 to −26), comes to **−0.11** station-out.

![nn-track results](../figures/nn_track_results.png)

## What moved the numbers (and what didn't)

* **Noise on the static columns** (MLP): 11 of model6's 25 features are
  constant within a station — a 37-row lookup table through which a net
  memorises station identity. Gaussian noise at 0.3σ on those columns took
  the MLP from +0.33 to +0.38 pooled and 12→17 stations positive. The same
  memorisation shows as train loss collapsing ~20× faster than grouped
  validation improves.
* **Quantile (rank→Gaussian) scaling of the statics** (hybrid): a z-score
  fitted on 37 station rows lets single outliers set a column's scale
  (soil_bdw 3.8σ, elevation 3.2σ). Rank-normalising is the change that took
  blocked pooled +0.24 → **+0.35** — the entire transfer gain, panel (d).
* **model8's stratified weights are redundant under quantile scaling** —
  they help only z-score (masking the same outlier problem at the loss
  instead of the input) and hurt blocked ADELONG (−1.07 vs +0.34).
* **The NSE\* loss** is the per-station form of NSE (pooled NSE is a
  rescaled MSE with the same optimum); it lifts the low-variance dry sites
  to parity and costs nothing pooled.
* **What didn't work**: a static-branch bottleneck (best pooled, worst
  M-sites — the wrong trade), and Huber loss (no gain).

Held-out predicted-vs-observed series for every station
([`plot_nn_per_station.py`](../plot_nn_per_station.py)) — compare the same
figure on the [model8 page](model8.md):

![nn-hybrid per-station held-out time series](../figures/nn_hybrid_per_station.png)

## Reproduce

```bash
# per-site (leave-station-out) and blocked, the recommended configuration
PYTHONPATH=. python -m emt.nn.hybrid cv --design station --loss nse --scale quantile --workers 12
PYTHONPATH=. python -m emt.nn.hybrid cv --design block   --loss nse --scale quantile --workers 5

# the other two models
PYTHONPATH=. python -m emt.nn     cv --design station --loss nse   # MLP
PYTHONPATH=. python -m emt.nn.seq cv --design station --loss nse   # Transformer
```

Writes `data/nn_*_predictions.csv`; `emt/nn/README.md` carries the full
configuration grids and the design rationale.
