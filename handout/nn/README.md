# The neural-network track

*A standalone handout: what was built, what it changed, and what it did not.
Reads on its own; the per-model pages in the main handout carry the detail.*

Code: [`emt/nn/`](../../emt/nn/) · Main handout: [`../README.md`](../README.md) ·
Package reference: [`emt/nn/README.md`](../../emt/nn/README.md)

---

## 1. The problem this track inherited

The repository already had two mature tracks — a gradient-boosting downscaler
of the ≈1 km SMIPS product ([model6](../modules/model6.md)) and a calibrated
bucket water balance driven only by SILO rain and PET
([model8](../modules/model8.md)). Both reach station-out pooled NSE ≈ +0.4, and
both fail in the same specific way:

> **The dynamics are solved; the level is not.** Median held-out per-station
> correlation is 0.82, but roughly 60 % of per-station mean-squared error is a
> constant offset. Remove each station's mean — an oracle, not achievable —
> and the median per-station NSE jumps from ≈ 0 to **+0.55…+0.61**, with 35–37
> of 37 stations positive.

Everything below is an attempt on that offset. It is worth stating the
conclusion at the top: **the neural models moved the headline numbers
materially, and none of them dissolved the level term.** What moved skill was
(a) removing a leak of station identity into the inputs, (b) an *observation*
of site level, and (c) combining diverse models. What did not move it was
capacity, and neither did any trained combiner.

Two validation designs are used throughout, both out-of-fold:

| design | what is held out | what it measures |
|---|---|---|
| **station-out** | one station, its cluster neighbours still in training | interpolation beside instrumented sites |
| **blocked** | a whole spatially independent district (YANCO, KYEAMBA, ADELONG, each M-site) | transfer to unobserved country — the honest test for a national product |

## 2. What was built

Four models, all run through the repository's existing validation ladder
([`emt/nn/cv.py`](../../emt/nn/cv.py) reproduces the station / block / year /
block×year folds):

| model | inputs | one-line idea | page |
|---|---|---|---|
| **nn-mlp** | model6's exact features (SMIPS + lookback + terrain + soil + antecedent SILO) | is a neural net better than boosting on identical inputs? | [nn-mlp](../modules/nn_mlp.md) |
| **nn-transformer** | 365 days of SILO rain/PET/VPD + statics, **no SMIPS** | learn the water balance from forcing history | [nn-transformer](../modules/nn_transformer.md) |
| **nn-hybrid** | forcing + statics, **no SMIPS** | keep model7's bucket, learn its parameters per site | [nn-hybrid](../modules/nn_hybrid.md) |
| **nn-stack** | the other models' out-of-fold predictions | learn *where* to trust each model | [nn-stack](../modules/nn_stack.md) |

## 3. Results

37 OzNet stations, 2006–2010, 50,623 station-days. The last three columns are
all from the **blocked** design so they compare like with like.

| | station-out NSE | blocked NSE | blocked stations NSE > 0 | blocked blocks NSE > 0 |
|---|---|---|---|---|
| **Ensemble (recommended)** | **+0.48** | **+0.42** | **22 / 37** | **8 / 9** |
| nn-hybrid + SMIPS level anchor | +0.39 | +0.34 | 19 / 37 | 6 / 9 |
| nn-hybrid (differentiable bucket) | +0.35 | +0.35 | 18 / 37 | 7 / 9 |
| nn-transformer, scaled (d=128, 4 layers) | +0.43 | +0.22 | 16 / 37 | 5 / 9 |
| nn-transformer, small (d=64, 3 layers) | +0.36 | +0.22 | — | — |
| nn-mlp | +0.38 | +0.17 | 16 / 37 | — |
| model8 — previous best (process) | +0.41 | +0.32 | 20 / 37 | 7 / 9 |
| model6 — previous best (ML) | +0.38 | +0.36 | 15 / 37 | 5 / 9 |

Reading the table:

* **nn-hybrid is the first model in the repository to beat model8 on blocked
  transfer** (+0.354 vs +0.322) — the number that matters for a national
  product.
* **The scaled Transformer is the best single network for interpolation**
  (+0.431 station-out, above model8) while its blocked score does not move at
  all. Capacity buys interpolation, not transfer.
* **The MLP transfers worst of everything measured** (+0.17 blocked). Pure ML
  on tabular features is the least robust way to leave the training districts.
* **The ensemble beats every member under both designs.** It is a plain median
  (blocked) or mean (station-out) — no trained weights.

![Neural-network track results](../figures/nn_track_results.png)

<sup>(a) the recommended ensemble's held-out fit; (b) per-station NSE, ensemble
against model8; (c) blocked per-block NSE, model6 → ensemble; (d) what the
statics preprocessing was worth.</sup>

## 4. Five findings

### 4.1 Eleven of the 25 tabular features are station constants — and a net will use them as an ID

Terrain and soil do not vary within a station, so `(elevation, twi, soil_clay,
…)` is a 37-row lookup table. The MLP memorises the station mean through it:
training loss collapses roughly twenty times faster than grouped-validation
skill improves. Gaussian noise at **0.3 σ on the static columns only** (dynamic
columns stay at 0.05 σ) took the MLP from +0.33 to **+0.38** pooled and 12 → 17
stations positive.

A stronger version — squeezing the statics through a 4-dimensional bottleneck —
scored the best pooled number (+0.40) by fitting the big clusters and losing
the isolated M-sites (M2 +0.31 → −0.52). Rejected: that is the wrong trade for
a national product, and a reminder that pooled NSE alone will mislead you here.

### 4.2 Rank-normalising the statics is what fixed blocked transfer

A z-score fitted on **37 station rows** lets a single site set a column's scale
(`soil_bdw` 3.8 σ, `elevation` 3.2 σ, `accumulation` skew 4.4). Replacing it
with a quantile (rank → Gaussian) transform is the entire blocked gain for the
hybrid:

| hybrid variant | station-out | blocked | blocked block-median |
|---|---|---|---|
| z-score statics | +0.352 | +0.244 | +0.03 |
| **quantile statics** | +0.346 | **+0.354** | **+0.30** |
| quantile + model8's stratified weights | +0.376 | +0.249 | +0.22 |

model8's stratified sample weights, which help under z-score, are **redundant
under quantile scaling** — they were compensating at the loss for a problem
better fixed at the input — and they hurt blocked ADELONG (−1.07 vs +0.34).
This was the single largest skill change in the whole track, and it is a
preprocessing decision, not a modelling one.

### 4.3 A per-station loss, because pooled NSE is just MSE

Pooled NSE is a rescaled MSE with the same optimum, so "optimising NSE" pooled
changes nothing. The per-station form does
([Kratzert et al. 2019](https://doi.org/10.5194/hess-23-5089-2019)):

```
loss = Σ  (ŷ − y)² / (σ_station + ε)²
```

It lifts low-variance sites — the dry M-sites — to parity instead of letting
the high-variance Yanco cluster dominate. On the low-variance half of the
network it is worth a mean **+0.46** NSE per station (M7 −20.2 → −14.1, A4 −2.2
→ −0.5) and it is neutral on the high-variance half, at no pooled cost. The
station label is used only to look up σ; it is never a network input.

### 4.4 The level term needs an observation, not a better estimator

The hybrid's statics (soil, terrain, aridity) are *proxies* for site level.
Adding each site's **SMIPS climatological mean** as a static — a
measurement-based level estimate, independent of the in-situ target, so
leakage-safe — gives the most stations-positive of any single model:
**+0.387 station-out, 21 / 37 positive, median |bias| 3.17 %**, against
+0.346 / 18 / −0.05 without it.

There is a subtlety worth recording. Pooled, the SMIPS site mean correlates
with the true site mean at only **0.34** — but *within* aridity terciles at
**0.38 / 0.44 / 0.61** (dry / mid / wet). It is a Simpson's-paradox signal: the
product's site-mean bias varies with climate, and a model given both SMIPS mean
and aridity can learn the conditional correction. That also explains why the
anchor is a wash under blocked validation (+0.339): the correction itself does
not fully transfer between districts.

Relatedly, on where SMIPS helps at all — SMIPS-fed models beat the SMIPS-free
hybrid by a median **+0.47 NSE at mid-aridity stations** and by ≈ 0 at dry and
wet ones. At dry sites the enforced water balance already captures what SMIPS
would add.

### 4.5 Every trained combiner loses to equal weighting

The models fail in different places, so a combiner is the obvious move. Five
were tested, all fold-disciplined (the combiner for a held-out site is trained
only on other sites' out-of-fold predictions):

| combiner | station-out | blocked |
|---|---|---|
| **equal mean / median** | **+0.450** | **+0.401** |
| gate on regime (day-of-year, recent rain / P−PET / VPD) | +0.429 | +0.376 |
| gate on statics (soil, terrain, aridity) | +0.439 | +0.360 |
| global convex weights (3–4 parameters per fold) | +0.400 | +0.350 |
| affine ridge (allowed to correct level) | +0.367 | +0.149 |

The ordering is the finding. **Even three fitted numbers per fold lose**: the
in-sample ranking of the bases does not transfer from 37 sites (the blocked
KYEAMBA fold drives model8's weight to zero, and that fold then pays for it).
Letting the combiner touch level — the affine variant — is a catastrophe,
which vindicates constraining the gate to a convex combination. Under blocked
validation a gate conditioned on site statics is itself a spatial model, and
fails like one.

What the exercise produced instead is the recommended configuration: **a plain
median over diverse, individually-disciplined models**, zero trained
parameters. Diversity is doing the work — model6 earns its place in the
blocked ensemble (+0.417 with it, +0.395 without) despite the weakest solo
block-median of the five, because it is the only pure-ML, SMIPS-fed member and
its errors are the least correlated with the rest.

## 5. The hybrid, in more detail

The differentiable bucket is the track's most interesting object, so it is
worth stating precisely. [model7](../modules/model7.md)'s recurrence, rewritten
in PyTorch (bit-identical to the numba original — maximum absolute difference
0.0 over the full panel), with a small MLP mapping station statics to
**deviations of the five bucket parameters**, sigmoid-bounded to model7's
calibration ranges:

```
θᵢ = lo + (hi − lo)·σ( g + MLP(staticsᵢ) )        θ = (smax, α, k, θr, Δθ)
Sᵢ′ = clip( Sᵢ + P − PET·min(1, Sᵢ/(αᵢ·smaxᵢ)) − kᵢ·Sᵢ , 0, smaxᵢ )
vwcᵢ = θrᵢ + Δθᵢ·Sᵢ/smaxᵢ  (+ a jointly-trained static offset head)
```

Three properties matter:

1. **The water balance stays enforced.** Only its parameters are learned, so
   soil changes capacity, ET stress and recession *inside the physics* —
   where model8 can only shift the readout afterwards.
2. **It starts as model7.** The deviation MLP's last layer is zero-initialised,
   so at step 0 the model *is* model7 at its default parameters; every
   departure has to earn its place on a station-grouped validation split.
3. **The fitted parameters stay physical** — across the 37 stations, smax
   144–171 mm, α 0.67–1.43, offsets within ±1.1 pp.

It also repairs the network's worst station. **M7 sits at NSE −13 to −26 for
every model 1–10; the hybrid brings it to −0.11 station-out** (bias +0.4 pp).
K12 and K14 remain unexplained by any covariate — their −12 to −17 pp offsets
are, on the repository's own reading, a property of the observation network
rather than of the models.

For 30 m maps the parameter-per-site design carries through:
[`emt/nn/spatial.py`](../../emt/nn/spatial.py) gives **every 30 m pixel its own
bucket**, forced by its nearest SILO cell — 1.6 M pixel-buckets × 1070 days ×
3 ensemble members in about four minutes on one GPU. Because the bucket is
sequential in time, `snapshots=` reads any number of dates out of a single
spin-up run, which is why the [nine-date
gallery](../modules/downscale.py.md#the-generated-product) costs one simulation
rather than nine.

## 6. How the models are trained

Shared by all of them ([`emt/nn/train.py`](../../emt/nn/train.py)):

* **AdamW** (decoupled weight decay) with a **one-cycle** schedule — warm-up
  then cosine anneal — and gradient-norm clipping at 1.0.
* **Station-grouped early stopping**: the validation split holds out whole
  stations, so training stops on transfer skill rather than on memorisation,
  with best weights restored.
* **Seed ensembling** (3–4 members) — the most reliable variance reducer for
  networks this small.
* **bf16 autocast**, the whole training set resident on the GPU with
  index-sliced mini-batches; at 50 k rows a DataLoader is the bottleneck.
* Standardisation fitted on training rows only; `log1p` on the heavy-tailed
  columns; quantile scaling available for the statics (§4.2).

Architectures: the MLP is pre-norm residual blocks (256-256-128, SiLU, dropout
0.15); the Transformer is a pre-norm encoder (d = 64 or 128, 3–4 layers, 4
heads) over a static token plus 365 forcing tokens with learned positional
embeddings, read out at the target day; the hybrid's parameter net is a single
32-unit hidden layer, and it trains full-batch because one forward pass
simulates the entire panel.

## 7. Running it

```bash
# cross-validation, any design: station | block | year | blockyear
PYTHONPATH=. python -m emt.nn        cv --design station --loss nse            # MLP
PYTHONPATH=. python -m emt.nn.seq    cv --design station --loss nse --d_model 128 --n_layers 4
PYTHONPATH=. python -m emt.nn.hybrid cv --design block   --loss nse --scale quantile
PYTHONPATH=. python -m emt.nn.hybrid cv --design station --loss nse --scale quantile --anchor
PYTHONPATH=. python -m emt.nn.stack  cv --design block                          # the combiner

# fit on everything, and make a 30 m map for any date (no SMIPS)
PYTHONPATH=. python -m emt.nn.hybrid fit --scale quantile
PYTHONPATH=. python -m emt.nn.spatial --bbox 147.30 -35.52 147.62 -35.10 --date 2008-08-05
```

Every run writes out-of-fold predictions to `data/nn_*_predictions.csv`, which
is what the figures and the ensembles read — no number in this handout is
recomputed by hand. The fitted hybrid ships as `data/models/nn_hybrid_q.pt`
(35 KB of tensors and dataclasses; unlike the scikit-learn artefacts it does
not go stale between library versions).

## 8. Limitations, and what would move it next

* **All numbers are Murrumbidgee.** 37 stations, one catchment, 2006–2010. The
  blocked design is the best available proxy for national transfer, not a
  substitute for national validation.
* **Year and block×year folds are not yet run** for the neural models; the
  ladder rows exist in `emt/nn/cv.py` but have only been executed for station
  and block.
* **The ensemble has no single-call tool.** The members are combined in
  [`plot_downscale_gallery_best.py`](../plot_downscale_gallery_best.py); a
  wrapper would need to run three model families for one AOI.
* **The level term is still the headroom**, and the evidence says it is
  informational: capacity does not touch it (§3), and no combiner does (§4.5),
  but an *observation* does (§4.4). The next thing to try is a proper satellite
  anchor for this era — **ESA CCI soil moisture** covers 2006–2010 where SMAP
  (2015−) does not — used the way the SMIPS climatological mean was, or as the
  coarse-% reference in the mass-conservation rebasing the
  [downscaling page](../modules/downscale.py.md#future-work-mass-conservation)
  describes.
* **Giving the Transformer SMIPS** is designed and pre-registered but not run
  ([nn-transformer](../modules/nn_transformer.md#not-yet-tested-giving-it-smips)),
  with its expected outcome stated in advance: better interpolation, little
  blocked gain, and a probable loss of ensemble diversity.

---

*Per-model detail: [nn-mlp](../modules/nn_mlp.md) ·
[nn-transformer](../modules/nn_transformer.md) ·
[nn-hybrid](../modules/nn_hybrid.md) · [nn-stack](../modules/nn_stack.md).
Main handout: [`../README.md`](../README.md).*
