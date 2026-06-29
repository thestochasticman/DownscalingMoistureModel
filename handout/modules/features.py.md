# `features.py`: training-table assembly (Stage 4)

Source: [`../../emt/features.py`](../../emt/features.py)

Assembles the training table. For each OzNet station-day it joins the coarse
SMIPS value, the static terrain covariates, and seasonality terms to the target.

| Function | Role |
|---|---|
| `build_training_table(coords, oznet_daily, start, end)` | Full table across stations |
| `station_features(…)` | Feature records for one station |
| `add_temporal_features(df)` | Cyclic day-of-year terms `doy_sin`, `doy_cos` |
| `_partition_clusters`, `_cluster_smips` | Cluster-fetch retrieval (below) |

## Columns

| Group | Columns |
|---|---|
| Target | `sm_rootzone_pct` (OzNet 0–90 cm, %) |
| Coarse | `smips_totalbucket` (mm, at the station pixel) |
| Terrain | `elevation, slope, northness, eastness, twi, hli, accumulation` |
| Temporal | `doy_sin, doy_cos` |
| Identifiers | `station, site, lat, lon` (`station` is the cross-validation group) |

## Cluster-fetch retrieval

A per-station retrieval issues one SMIPS WCS request for each station, with
substantial overlap between nearby stations. `build_training_table` instead
groups stations of a site whose combined extent is within
`MAX_CLUSTER_DEG = 0.7°`, retrieves one shared SMIPS cube for the group
(`_cluster_smips`), and samples each station from it. Dispersed groups fall back
to per-station retrieval. For the three-catchment build this reduces SMIPS
retrieval from 30 station requests to 3 site-level cubes.

The substitution is exact because grid alignment in
[`smips.py`](smips.py.md) (`snap_bbox`) makes sampling independent of the request
window, so a shared cube and a per-station cube return identical values (verified:
maximum absolute difference 0.000 mm).

## Output

Long-format table, one record per station-day, consumed by
[`model1`](model1.md).
