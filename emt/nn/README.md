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
