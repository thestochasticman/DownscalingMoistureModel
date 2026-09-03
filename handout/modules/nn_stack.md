# nn-stack: a learned combiner — a negative result, and the diversity win

<!-- NAV -->
[← nn-hybrid](nn_hybrid.md) · [Index](../README.md) · [In-situ networks →](insitu_networks.md)
<!-- /NAV -->

Source: [`../../emt/nn/stack.py`](../../emt/nn/stack.py)

The repo's models fail in *different places* (the hybrid fixes K2/A5/Y7/K8
exactly where model8 fails), so a combiner is the obvious next step. The
stack is the careful version: a small net maps a sample's statics and the
bases' disagreement to **softmax weights over the base predictions** — a
convex combination, no intercept, no free scale, so it can only choose
between models, never shift a site's level. Zero-initialised at the plain
mean; fold-disciplined like everything here (the gate for a held-out site is
trained only on other sites' out-of-fold base predictions).

## Results: every learned combiner loses to equal weighting

Four combiners of increasing restraint, all fold-disciplined
([`run_stack_variants.py`](../run_stack_variants.py) for the linear two):

| pooled NSE | station (4 bases) | block (3 bases) |
|---|---|---|
| **equal mean** | **+0.450** | **+0.401** |
| gate on regime (when: doy, recent rain/P−PET/VPD) | +0.429 | +0.376 |
| gate on statics (where: soil/terrain/aridity) | +0.439 | +0.360 |
| global convex weights (3–4 params per fold) | +0.400 | +0.350 |
| affine ridge (allowed to correct level) | +0.367 | +0.149 |

The ordering is the finding. **Even three fitted numbers per fold lose**: the
training rows systematically misestimate the held-out optimum (the blocked
KYEAMBA fold drives model8's weight to 0.0; the weights chase whichever base
looked good in-sample). Freeing the combiner to touch level (affine) is a
disaster — vindicating the convexity constraint. Conditioning on regime
rather than site is the least-bad learned variant, and still short. With 37
sites the in-sample base ranking simply does not transfer; like
[model5](model5.md) and [model10](model10.md), this page records the negative
result — in its strongest form — so it is not re-attempted.

## What the exercise surfaced: base diversity

The plain, parameter-free mean over the diverse validated bases is the
**best result in the repo under both designs**:

| | pooled NSE | stations NSE>0 | median stn | block-median |
|---|---|---|---|---|
| blocked: mean(hybrid, model8, model6) | **+0.401** | 21/37 | **+0.12** | **+0.38** |
| station: mean(hybrid, model8, mlp, seq) | **+0.450** | 18/37 | −0.02 | +0.28 |
| previous best single (model8 / blocked) | +0.322 | 20/37 | +0.07 | +0.25 |

Against the paper's headline "transfers at pooled NSE ≈ +0.32", the diversity
mean moves the blocked number to **+0.40** with zero trained parameters.
That is the recommended combination.

## Reproduce

```bash
PYTHONPATH=. python -m emt.nn.stack cv --design block  --workers 9
PYTHONPATH=. python -m emt.nn.stack cv --design station --workers 12
```
