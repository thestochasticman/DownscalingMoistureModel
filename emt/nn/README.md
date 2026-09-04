# `emt.nn` — the neural-network track

PyTorch models for the SMIPS → 30 m downscaling problem, evaluated on the same
validation ladder as the rest of the repo.

## Layout

| module | responsibility |
|---|---|
| `config.py` | `DataConfig` (feature list = model6's, target, station, time), `MLPConfig`, `TrainConfig` — every tunable lives here |
| `data.py` | `TabularData`: DataFrame → numpy (features, target, station, time, weight); `Scaler`; station-grouped splits; per-station σ for the NSE\* loss |
| `losses.py` | `mse`, `huber`, `nse` (per-station NSE\*) |
| `mlp.py` | `ResidualMLP`: pre-norm residual blocks, SiLU, dropout |
| `train.py` | `Trainer`: one net, one run — AdamW, one-cycle LR, clipping, bf16 autocast, input noise, early stopping with best-weight restore |
| `model.py` | `MLPModel`: fit/predict on DataFrames, seed ensemble, `save`/`load` (`.pt`) |
| `cv.py` | the ladder: `station` / `block` / `year` / `blockyear` folds, parallel across processes on one GPU; `summarise` |
| `seq.py` | **the sequence model**: `SeqData` (per-station forcing panels + (station, day) samples), `SeqView` (on-GPU window gathering), `SeqTransformer`, `SeqModel`; own CLI |
| `search.py` | resumable random hyper-parameter search scored on the ladder |
| `__main__.py` | CLI |

The package does **not** import the data-fetch stack (PaddockTS); it consumes
the cached feature table `data/model6_features_2006_2010.csv`.

## Use

```bash
PYTHONPATH=. python -m emt.nn cv  --design station --loss nse --workers 8
PYTHONPATH=. python -m emt.nn cv  --design block   --loss mse --hidden 512,256,128 --dropout 0.3
PYTHONPATH=. python -m emt.nn fit --out data/models/nn_mlp.pt
PYTHONPATH=. python -m emt.nn.search --trials 30 --design block
```

Every field of `TrainConfig` / `MLPConfig` is a CLI flag.

```python
from emt.nn import MLPModel, TrainConfig
m = MLPModel(train=TrainConfig(loss="nse", n_ensemble=5)).fit(df)
m.predict(df_new)            # NaN in → NaN out
m.save("data/models/nn_mlp.pt"); MLPModel.load("data/models/nn_mlp.pt")
```

## The sequence model (`emt.nn.seq`)

The neural analogue of the model7/8 bucket: **no SMIPS**, only the SILO forcing
window and the station statics, so it runs for any date.

    sample (station s, day t):
      seq  = forcing[s, t-L+1 .. t]  x  (log1p rain, PET, VPD, doy sin, doy cos)     L = 365
      sta  = statics[s]              (soil x4, terrain x7, aridity)
    tokens = [static token] + L forcing tokens, learned positional embedding
    pre-norm Transformer encoder (d=64, 3 layers, 4 heads) -> readout at token t -> linear

Windows are gathered on the fly from a `(n_stations, T, channels)` forcing
tensor on the GPU; nothing of size samples×L is ever built. `SeqData`/`SeqView`
implement the same protocol as the tabular classes, so `Trainer`, the
ensemble, the losses and the CV ladder are shared unchanged.

```bash
PYTHONPATH=. python -m emt.nn.seq cv  --design station --loss nse --workers 8 \
    --epochs 60 --patience 10 --n_ensemble 2 --batch_size 512
PYTHONPATH=. python -m emt.nn.seq fit --out data/models/nn_seq.pt
```

Cost: ~4.5 s/epoch at batch 256 (2.5 GB); 8 folds in parallel fit on one 32 GB card.

## Preprocessing

`Scaler` does `log1p` on the heavy-tailed columns (`accumulation`, `rain_7`,
`rain_30`, `rain_365`, `slope` — skew 1.5–4, max/median in the hundreds), then
z-scores everything; the target is z-scored too. All statistics come from the
training rows of the fold.

Eleven of the 25 features (terrain + soil) are **constant within a station** —
a 37-row lookup table through which the net can read station identity and
memorise the station mean. Three levers address that, all off by default except
a small static noise:

| lever | where | effect |
|---|---|---|
| `TrainConfig.static_noise` | noise on the static columns only (dynamic columns use `input_noise`) | blurs the station fingerprint |
| `MLPConfig.static_bottleneck` | statics enter via `Linear→SiLU→Dropout(static_dropout)` of this width | limits how much identity can pass |
| `DataConfig.static = ()` | disables both | control |

CLI: `--static_noise 0.3 --static_bottleneck 4`, `--no-log1p`, `--no-static`.

## The loss question

Pooled NSE = 1 − SSE/SST, and SST is a constant of the training set, so
minimising MSE *is* maximising pooled NSE. What the handout reports and cares
about is **per-station** NSE, which normalises by each station's own variance.
`loss="nse"` is that loss (Kratzert et al. 2019's NSE\*):

    err² / (σ_station + ε)²

so a low-variance station (e.g. the dry M-sites) is weighted up to parity with
a high-variance one instead of being drowned out. The station column is used
only to look up σ; it is never a network input.

## Training practices, and why

- **Standardisation** on training rows only; target too (loss in σ units).
- **AdamW** — decoupled weight decay is the correct L2 for Adam.
- **One-cycle schedule** (warm-up → cosine anneal) — warm-up stabilises early
  Adam steps; annealing to ~0 gives a clean convergence with no LR tuning.
- **Gradient clipping** at norm 1 — insurance against rare bad batches.
- **Dropout + LayerNorm + residual blocks** — the standard tabular-MLP recipe.
- **Input noise** — a cheap regulariser that also encodes that covariates
  (SLGA, SILO) carry measurement error.
- **Station-grouped early stopping** — the validation split holds out whole
  stations, so training stops on the transfer skill we report, not on
  memorising site identity from static covariates (train loss falls to ~0.04 σ²
  within epochs; held-out NSE does not follow — this is the whole problem).
- **Seed ensembling** — averaging 3–5 nets is the most reliable variance
  reducer for small tabular networks.
- **Whole table on the GPU**, index-sliced mini-batches — at 50k rows a
  DataLoader is the bottleneck; per-step overhead means batch 1024–2048 is
  5× faster per epoch than 512 with no skill cost.

## Current standings (out-of-fold, 37 stations, 2006–2010)

The recommended configurations, membership decided only by each base's own
validation (blocked pool: bases with blocked pooled ≥ +0.3), median for the
blocked design, mean for station:

| | pooled NSE | stations > 0 | stn median | blocks > 0 | block-median |
|---|---|---|---|---|---|
| **blocked: median(hyb, hybA, m8, m6, m9)** | **+0.417** | **22/37** | +0.11 | **8/9** | **+0.42** |
| **station: mean(hybA, m8, seq-big)** | **+0.483** | 21/37 | +0.16 | 8/9 | +0.38 |
| station: mean(hyb, m8, seq-big) | +0.474 | 21/37 | **+0.23** | 8/9 | +0.31 |
| best single, blocked: nn-hybrid (quantile) | +0.354 | 18/37 | −0.05 | 7/9 | +0.30 |
| best single, station: scaled Transformer | +0.431 | 17/37 | −0.07 | | |
| repo baseline (model8, blocked / station) | +0.322 / +0.408 | 20/37 | +0.07 / +0.13 | 7/9 | +0.25 |

Single models (station-out pooled / blocked pooled):

| model | inputs | station | blocked | note |
|---|---|---|---|---|
| nn-mlp (log1p + static noise 0.3σ) | model6's features (SMIPS) | +0.379 | +0.170 | static noise is the lever (12→17 stn>0); worst transfer |
| nn-seq small (d=64, 3L) | SILO window + statics, no SMIPS | +0.355 | +0.217 | matches the MLP without SMIPS |
| **nn-seq scaled (d=128, 4L, 3-member)** | same | **+0.431** | +0.221 | best single-net interpolation, median r 0.83; scale buys no transfer |
| **nn-hybrid (quantile statics)** | forcing + statics, enforced bucket | +0.346 | **+0.354** | first model to beat model8 blocked; M7 −0.11 |
| nn-hybrid + SMIPS anchor (`--anchor`) | + site SMIPS climatological mean | **+0.387** | +0.339 | **21/37 stn>0, most of any single model**; anchor transfers only partially |
| model6 / model8 / model9 (baselines) | | +0.38 / +0.408 / — | +0.355 / +0.322 / +0.35 | model9 rescued as an ensemble base |

Every model's residual is dominated by per-station LEVEL, not dynamics
(median r 0.77–0.83; oracle per-station de-bias → median ≈ +0.55–0.61 with
35–37/37 positive) — the remaining headroom, and it needs information
(a better satellite anchor, e.g. ESA CCI for this era), not architecture.

## How we got here (findings ledger, chronological)

1. **MLP** at model6 parity on identical features; the lever was Gaussian
   noise at 0.3σ on the 11 station-constant columns (the station-identity
   leak); a static bottleneck traded M-sites for clusters — rejected.
2. **NSE\*** (per-station variance-normalised loss) lifts low-variance dry
   sites at no pooled cost; pooled NSE is a rescaled MSE, so only the
   per-station form changes the optimum.
3. **Transformer** matches the MLP without SMIPS; fixes some level outliers
   (Y7, A4, K8) and gives it back elsewhere — a learned water balance drifts.
   Scaling it (d=128, 4L) buys interpolation (+0.355 → +0.431) and no
   transfer (+0.22 either size): the level wall is informational.
4. **Hybrid** (differentiable model7 bucket, statics → bounded parameter
   deviations, zero-initialised at model7): **quantile scaling of the 37-row
   statics** is what fixed blocked transfer (+0.244 → +0.354 — the entire
   gain; z-score let single stations set column scales, e.g. soil_bdw 3.8σ).
   model8's stratified weights are redundant under it and hurt ADELONG.
5. **Every trained combiner loses to equal weighting** (station/block pooled):
   equal mean +0.450/+0.401 > regime gate +0.429/+0.376 > statics gate
   +0.439/+0.360 > global convex 3-params +0.400/+0.350 > affine ridge
   +0.367/+0.149. Even three fitted numbers per fold lose — the in-sample
   base ranking does not transfer from 37 sites; level freedom is a disaster
   (the convexity constraint protects). See `handout/modules/nn_stack.md`.
6. **SMIPS level anchor**: each site's SMIPS climatological mean as a static.
   Pooled corr with the true site mean is only 0.34, but 0.38–0.61 *within*
   aridity terciles (Simpson's) — the net learns the conditional correction.
   Station-out +0.387 / 21 positive; blocked a wash (the bias structure
   differs by district).
7. **Base diversity + robust combination** (median under blocked) is the
   final recommendation — the "Current standings" table above.
