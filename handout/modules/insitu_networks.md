# In-situ networks — the contract, and where more data would come from

<!-- NAV -->
[← nn-stack](nn_stack.md) · [Index](../README.md) · [Data quality →](qc.md)
<!-- /NAV -->

Source: [`../../emt/insitu/base.py`](../../emt/insitu/base.py) ·
[`../../emt/insitu/oznet.py`](../../emt/insitu/oznet.py)

[Blocked validation](blocked_validation.md) established that skill collapses at
the edges of the sampled climate envelope, and that the effective sample is
**9 independent locations**, not 37 stations. Both point the same way: the
binding constraint is the in-situ data, not the model. This page is the survey
of what more is available, and the layer that lets a second network in.

## The contract

Everything downstream — [`build_dataset`](features.py.md),
[`model7/build.py`](model7.md), every harness in `handout/` — consumes one
table shape. Until recently that shape lived implicitly inside the OzNet
loader, which is why the project had exactly one network.
[`emt/insitu/base.py`](../../emt/insitu/base.py) states it:

| column | meaning |
|---|---|
| `site` | grouping label; the spatial CV block derives from it |
| `station` | **unique across all networks** — it is the CV grouping key |
| `time` | midnight-normalised day |
| `sm_rootzone_pct` | daily mean volumetric water content, **per cent** |
| `n_layers` | how many sensor depths formed the integral |

plus `station, lat, lon` from [`coordinates.py`](../../emt/insitu/coordinates.py).
`check_target` validates conformance, `assert_disjoint` catches station-code
collisions between networks, `depth_weighted_mean` does the mechanical part of
reconciling sensor depths, and `combine` merges networks with both checks
applied.

**Adding a network is a parser plus a depth decision.** The parser is
mechanical. The depth decision is not — see below.

## Two things the contract caught immediately

**A range check does not catch the unit error it is for.** A table in fractions
has median ≈ 0.2, comfortably inside 0–100 %. The check that works is on the
**median**; the range check is kept only for the impossible cases.

**The target is not depth-homogeneous.** `Y3`'s root zone is a **single sensor
layer**, where every other station uses three. Under blocked validation Y3
reads NSE −0.34 with bias **+4.27 %** — the model too wet, which is what a
shallow reference produces, since it dries faster than a profile mean. With
n = 1 this is a hypothesis, not a result. But it raises the possibility that
part of the between-site level error this project has spent its effort on is a
**measurement** inconsistency rather than a modelling failure, and it sets the
standard any new network must meet.

(A third check fired on 27 rows above 55 % VWC. Those are genuine: a contiguous
twenty-day run at M6 during the record November 2010 flooding. Inundation at
the edge of sensor validity, not a defect.)

## The survey

| source | sites | depth | verdict |
|---|---|---|---|
| **OzNet, unused years** | 38 profile stations, **2001–2025** | 0–90 cm — identical | **best value, blocked on bandwidth** |
| OzNet YA/YB grid | 24, Yanco, 3 km/9 km lattice | **0–5 cm surface** | unusable as a target |
| CosmOZ | ~19 national | **10–30 cm**, varies with wetness | deprioritised — depth mismatch |
| OzFlux | ~23 national, tropics to arid | profile sensors | **best second network** |

**The largest gain is data already reachable.** The archive holds 2,202 files
across 38 core profile stations spanning **2001–2025**; the committed tables
use 2006–2010 — 33 % of the files, 186 of 615 station-years. Same instruments,
same depths, same parser: no reconciliation problem at all. It would take the
[temporal validation](temporal_validation.md) from five years with one anomaly
to twenty-five spanning the Millennium Drought, the 2010–12 La Niña, the
2017–19 drought and the 2020–22 La Niña.

**The dense array exists but measures the wrong thing.** The YA/YB grid is
exactly the geometry needed to check a 30 m *spatial pattern* — the one claim
in this project that has never been tested. But at 0–5 cm it cannot be a
root-zone target; using it would replicate the Y3 problem 24 times. It was
excluded only *incidentally* (no coordinate page → dropped for want of lat/lon)
until [`site_of`](../../emt/build_dataset.py) was changed to reject anything
failing `^[YKAM]\d+$`. Note `"YA1"[:1] == "Y"`, so the old prefix map would
have filed them under YANCO.

**CosmOZ is a worse fit than first assumed.** Its effective depth is typically
**10–30 cm** and *varies with soil wetness* — the Y3 problem in a harder form,
across 19 sites. **OzFlux** carries profile sensors and reaches the tropics and
arid interior the Murrumbidgee lacks, and is the better second network.

## Blocked: the OzNet server is 9.5 KB/s

The temporal extension is **not** blocked on method, credentials or permission.
It is blocked on bandwidth. Measured directly, one file:

```
https://www.oznet.org.au/.../k2_04_su_sm.xls
HTTP 200   2.48 MB in 266.2 s   =   0.0093 MB/s
```

The 736 cached files are exactly the 2006–2010 window; roughly **1,460 files
remain**, averaging ~2.6 MB. At the measured rate that is **on the order of
100 hours** of continuous downloading — during which a 28-minute run retrieved
six files.

**What would unblock it**, in order of preference:

1. A **bulk archive** from the OzNet custodians (Monash/Melbourne) — one
   request, and the whole problem disappears.
2. **ISMN** ([International Soil Moisture Network](https://ismn.earth)), which
   aggregates OzNet with standardised metadata and may serve faster. Needs
   registration, which is why it has not been tried here.
3. Leaving the download running for days. `download_oznet` caches per file, so
   it resumes; nothing is wasted. This is the fallback, not a plan.

The forcing side is **done**: `data/process_forcing_2000_2025.csv`, 351,389
rows, 37 stations, 2000-01-01 → 2025-12-31, built by
[`extend_forcing_2000_2025.py`](../extend_forcing_2000_2025.py). SILO is fast;
only OzNet is slow.

## What extending the forcing already showed

The aridity static is **a property of the window, not the site**. Recomputed
over 2000–2025 rather than 2005–2010 the normals move by up to 0.63 sd, and the
window [`predict.py`](../../emt/model8/predict.py) used at inference differs
from the training window by up to **1.11 sd** — about 0.7 percentage points of
level, arising purely from which years were fetched. Fixed on the
`aridity-reference-window` branch by pinning a reference climatology; the
refit costs nothing in skill (station-out +0.408 → +0.395, blocked +0.322 →
+0.319, block-median +0.249 → +0.252), so it is justified by removing an
arbitrary dependence rather than by accuracy.

## Status

The contract is in place and the surface-grid exclusion is explicit and
regression-checked. The temporal extension is specified, scripted
([`extend_target_2001_2025.py`](../extend_target_2001_2025.py)) and waiting on
bandwidth. OzFlux is next once that resolves — with its depth compatibility
checked *before* any parser is written, which is the lesson CosmOZ taught.

---
<!-- NAV -->
[← nn-stack](nn_stack.md) · [Index](../README.md) · [Data quality →](qc.md)
<!-- /NAV -->
