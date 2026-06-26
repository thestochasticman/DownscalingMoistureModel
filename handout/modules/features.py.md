# `features.py` — the training table (Stage 4)

Source: [`../../emt/features.py`](../../emt/features.py)

Joins everything into the model's training table: for every OzNet station-day,
attach the coarse SMIPS value, the static terrain covariates, and seasonality.

| function | role |
|---|---|
| `build_training_table(coords, oznet_daily, start, end)` | the full table across all stations |
| `station_features(…)` | feature rows for one station (target + SMIPS + terrain + time) |
| `add_temporal_features(df)` | cyclic `doy_sin`, `doy_cos` |
| `_partition_clusters`, `_cluster_smips` | the cluster-fetch optimisation (below) |

## Columns produced
| group | columns |
|---|---|
| target | `sm_rootzone_pct` (OzNet 0–90 cm %) |
| coarse | `smips_totalbucket` (mm, at the station pixel) |
| terrain | `elevation, slope, northness, eastness, twi, hli, accumulation` |
| temporal | `doy_sin, doy_cos` |
| id/space | `station, site, lat, lon` (`station` is the spatial-CV group) |

## Cluster fetch (the optimisation)
A naive build issues one SMIPS WCS request *per station*. `build_training_table`
instead groups stations of one `site` whose envelope is tighter than
`MAX_CLUSTER_DEG = 0.7°` and fetches **one shared SMIPS cube** for the whole
group (`_cluster_smips`), sampling each station from it. Wider / scattered
groups fall back to per-station cubes.

This is only safe because [`smips.py`](smips.py.md)'s `snap_bbox` makes sampling
window-independent — so the cluster cube and a per-station cube return the
**identical** value. Validated on K5/K6/K7: max abs difference **0.000 mm**.

## Output
Long-format table, one row per station-day → consumed by
[`model.py`](model.py.md). Cached at `data/train_kyeamba_2020JJ.csv`.
