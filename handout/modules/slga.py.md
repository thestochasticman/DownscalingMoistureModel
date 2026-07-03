# `slga.py`: soil covariates (SLGA, ~90 m)

<!-- NAV -->
[← Terrain covariates](covariates.py.md) · [Index](../README.md) · [Training table →](features.py.md)
<!-- /NAV -->

Source: [`../../emt/slga.py`](../../emt/slga.py)

Loads static soil covariates from the Soil and Landscape Grid of Australia
(SLGA v2, TERN), ~90 m national soil-property grids. Built to test whether soil
supplies the absolute-moisture baseline that SMIPS and terrain miss.

> **Status: used in the recommended model — but the result was conditional.**
> Added *alone* (to models 1–3), soil covariates did **not** improve
> leave-site-out skill: `soil_sand` became a near-unique per-station identifier
> and hurt generalisation. But once the SMIPS pixel-climatology supplies a
> legitimate level anchor, the same four soil covariates *help* — so soil is a
> feature of [`model4`](model4.md) and [`model6`](model6.md). The lesson: soil is
> useful as *texture* on top of a level signal, not as the level itself.

| Function | Role |
|---|---|
| `soil_covariates(query)` | Root-zone (0–100 cm) soil covariates for `query.bbox` as an `xr.Dataset` |

`SOIL_VARS = (soil_clay, soil_sand, soil_awc, soil_bdw)`: clay and sand fraction,
available water capacity, and bulk density. Each is the depth-weighted mean over
the five SLGA slices spanning 0–100 cm (weights = slice thickness), matching the
0–90 cm root-zone target.

## Why a project-local loader

PaddockTS provides `download_slga_soils`, but it hardcodes a single SLGA release
date in the COG URL (`..._20210902.tif`). That date is correct for clay and sand
but returns 404 for AWC (released `20210614`) and bulk density (`20230607`): each
SLGA attribute is published on its own date. This module resolves the actual
filename per attribute from the TERN datastore directory listing (robust to date
changes), while reusing PaddockTS's TERN API-key authentication and attribute
codes.

Requires a TERN API key (`tern_api_key` in `~/.config/PaddockTS.json`).

## Use

The covariates are static per location and would be sampled at each station and
reprojected per pixel for downscaling. That integration was implemented, tested,
and reverted (the experiment above); the loader remains callable on its own for
re-evaluation when more sites are available.

---
<!-- NAV -->
[← Terrain covariates](covariates.py.md) · [Index](../README.md) · [Training table →](features.py.md)
<!-- /NAV -->
