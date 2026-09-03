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

## Results so far (leave-station-out, 2006–2010, 37 stations, common 47,786 rows)

| model | inputs | pooled NSE | stn > 0 | median stn NSE | median r | oracle de-biased median |
|---|---|---|---|---|---|---|
| nn-mlp (z-score only) | SMIPS + statics + antecedent | +0.329 | 12 | −0.49 | 0.78 | +0.57 |
| nn-mlp B (log1p + static noise) | same | +0.379 | 17 | −0.33 | 0.79 | +0.51 |
| nn-seq Transformer (NSE\*, L=365, 2-member) | SILO forcing window + statics, **no SMIPS** | +0.355 | 15 | −0.64 | 0.77 | +0.53 |
| model6 (boosting) | as nn-mlp | +0.38 | 16/36 | | 0.81 | |
| model8 (process bucket) | as nn-seq | +0.412 | 19 | +0.13 | 0.82 | +0.60 |

The Transformer reaches the MLP's level **without SMIPS** — the same inputs as
model8 — and is the first NN to fix some of the worst level outliers
(Y7 −7.5 → +0.01, K12 −12.5 → −8.2, K8 −3.4 → −0.4, A4 −0.5 → +0.86). It loses
elsewhere (A3, K1, K4, A5) and is 17 wins / 15 losses per station against the
MLP. It is still short of model8, and the deficit is still level: median 66 %
of station MSE is bias; de-biased it is +0.53. Its water balance is learned;
model8's is enforced.

### Hybrid (differentiable bucket) results

| design | model | pooled NSE | stn > 0 | median stn NSE | median \|bias\| | de-biased median |
|---|---|---|---|---|---|---|
| station | **hybrid** | +0.352 | 19/37 | **+0.02** | **3.34** | +0.54 |
| station | model8 | +0.408 | 20/37 | +0.13 | 3.57 | +0.61 |
| block | **hybrid** | +0.244 | 18/37 | −0.00 | 3.83 | +0.54 |
| block | model8 (weighted) | +0.322 | 20/37 | +0.07 | 3.17 | +0.60 |

First untuned configuration (hidden 32, dev_scale 1, 3-member, no stratified
weights). It does not beat model8 pooled, but it is the only model whose
station-out **median** station is positive besides model8, and it has the best
median |bias| and by far the best M-site transfer (M4 +0.54, M5 +0.75, M6 +0.45
blocked; M7 −6.6 where every other model is −13 to −26). Station-wise vs
model8: 12 wins / 20 losses — it fixes model8's worst failures (K2 +3.8,
A5 +2.3, Y7 +0.8, K8 +0.7 NSE) and loses where model8 is already good.
The two models' errors are again complementary.

### Preprocessing attribution and the quantile result

The scaling question ("are we normalising as well as we can?") turned out to be
the lever. Hybrid, all three treatments, both designs:

| variant | station pooled / stn>0 / median | block pooled / block-median / blocks>0 |
|---|---|---|
| z-score | +0.352 / 19 / +0.02 | +0.244 / +0.03 / 5 |
| **quantile** | +0.346 / 18 / −0.05 | **+0.354 / +0.30 / 7** |
| quantile + stratified weights | +0.376 / 17 / −0.19 | +0.249 / +0.22 / 5 |
| model8 (weighted) | +0.408 / 20 / +0.13 | +0.322 / +0.25 / 7 |

- **Quantile scaling of the 37-row statics is what fixed blocked transfer**
  (+0.244 → +0.354, beating model8's +0.322): with rank-normalised inputs no
  station's outlying soil/terrain value can drag the parameter MLP, so the
  physics extrapolates. It also cut M7 — the repo's worst station, −13 to −26
  under every earlier model — to −0.11 station-out / −1.1 blocked.
- **The stratified weights are redundant under quantile scaling** (they help
  only z-score, where they mask the same outlier problem) and hurt blocked
  ADELONG (−1.07 vs +0.34).
- **mean(hybrid-quantile, model8)** from the cached OOF predictions is the best
  result in the repo under BOTH designs: station +0.421 / 20 / +0.15,
  block +0.374 / block-median +0.38. The two models' errors are complementary;
  the average beats each everywhere it matters.

### The stack (`emt.nn.stack`): a negative result, and the diversity win

A learned convex gate (statics + base disagreement → softmax weights over the
base predictions, zero-initialised at the plain mean, fold-disciplined) LOSES
to equal weighting:

| design | plain multi-base mean | gated stack |
|---|---|---|
| station (hybrid, model8, mlp, seq) | **+0.450** pooled | +0.439 |
| block (hybrid, model8, model6) | **+0.401** pooled / stn-med **+0.12** / blk-med **+0.38** | +0.360 / +0.00 / +0.29 |

Thirty-seven sites are too few to learn a transferable who-to-trust map; under
the block design the gate is itself a spatial model and fails like one. What
the exercise surfaced instead is that **base diversity is the win**: the plain
mean over three (blocked) / four (station) validated models is the repo's best
result under both designs — blocked +0.401 pooled, block-median +0.38, median
station +0.12 — with zero trainable parameters. That is the recommended
combination.
