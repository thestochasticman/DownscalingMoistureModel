# `model10`: the hybrid — a negative result, with one instructive exception

<!-- NAV -->
[← Temporal validation](temporal_validation.md) · [Index](../README.md) · [nn-mlp →](nn_mlp.md)
<!-- /NAV -->

Source: [`../../emt/model10/model.py`](../../emt/model10/model.py)

The two tracks fail in different places. Under blocked validation
[model6](model6.md) collapses at M2 (−3.39) where [model8](model8.md) scores
+0.76; model8 fails at Adelong where model6 nearly survives. That
complementarity has been the standing argument for a hybrid since
[model7](model7.md).

model10 is the cheap direction of it: model6's feature set plus
**`bucket_storage`** — the fitted model8 water balance's state, in mm, on the
day. The storage comes from SILO rain and PET through the fitted bucket, so it
is a national, backward-looking covariate like every other: it carries no
in-situ information and cannot memorise station identity. Against the target it
is a comparable predictor to SMIPS itself (r 0.45 against 0.52), while being
available on days and in places SMIPS is not.

Same 47,786 usable rows as model6, so the comparison is exactly same-rows.

## Results

| blocked 9-fold | pooled | block-median | blocks NSE>0 | station-median |
|---|---|---|---|---|
| model6 | +0.355 | +0.092 | 5/9 | −0.21 |
| **model10** | **+0.367** | +0.036 | **6/9** | **−0.13** |
| model8 (process) | +0.322 | **+0.249** | **7/9** | **+0.07** |

Block × year: model6 +0.340, model10 +0.345, model8 +0.273.

**This is a negative result.** Pooled improves by 0.012 and the station-median
by 0.08, but the **block-median falls** (+0.092 → +0.036). The bucket state
helps model6 slightly where model6 was already adequate and does not repair
what is broken. It does not approach model8's block-level transfer.

## The exception, and what it teaches

At **M2** — the block where model6 collapses and model8 excels — the hybrid
recovers half the failure:

| block | model6 | model10 | model8 |
|---|---|---|---|
| **M2** | −3.39 | **−1.66** | +0.76 |
| ADELONG | −0.33 | −0.41 | −1.03 |
| M7 | −3.32 | −4.68 | −4.42 |

So the complementarity is real and the mechanism works — but only there.
Adelong is unchanged and M7 is worse.

The reading: **one feature among twenty-six is too weak a coupling.** The
bucket's value is its *level* discipline, and a 127-leaf boosting model with
twenty-five other predictors will not defer to it; the evidence is that the
gain appears in pooled and station-level views (dynamics, which model6 already
had) and not in the block-median (level, which is what was broken).

The other direction — **assimilating SMIPS into the bucket as an observation**,
leaving the physics in charge of the level — is the harder and more promising
one, and remains open.

## Status

Evaluated, not shipped. Applying model10 to a map needs bucket storage per
pixel; [`emt.model8.predict`](../../emt/model8/predict.py) already computes it
on the SILO forcing grid, but wiring that into
[`emt.downscale`](downscale.py.md) was not done, because the result does not
warrant it.

```bash
PYTHONPATH=. python handout/run_blocked_cv.py m10 m10@blockyear
```

---
<!-- NAV -->
[← Temporal validation](temporal_validation.md) · [Index](../README.md) · [nn-mlp →](nn_mlp.md)
<!-- /NAV -->
