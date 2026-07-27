#!/usr/bin/env python3
"""Compare original vs OzNet-retrained model6 under dense local calibration.

The comparison uses one dense point/date model-input table and two global model6
baselines:

* original tracked model6 artefact from ``HEAD:data/models/model6.joblib``;
* current working ``data/models/model6.joblib`` after the OzNet 255-leaf retrain.

For each baseline the script evaluates:

* global model predictions at dense points;
* local residual calibration fitted to all dense points;
* point-group held-out local residual calibration, where whole points are held
  out using GroupKFold.

It can also write GeoTIFF maps for every dense validation date, applying each
global model and its local residual corrector over the full AOI grid.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from attrs import evolve


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DENSE_ROOT = Path("/Volumes/Dmitry_work/borevitz_projects/model6_dense_validation_spiking")
DEFAULT_OUTPUT_NAME = "model6_global_retrain_local_calibration"
DEFAULT_SILO_EMAIL = "dmitry.grishin@anu.edu.au"
NODATA = -9999.0


def process_only_silo_email(email: str | None) -> str:
    """Supply a SILO fallback email without rewriting PaddockTS config."""
    if not email:
        return "no fallback email supplied"
    os.environ.setdefault("PADDOCKTS_EMAIL", email)
    try:
        import PaddockTS.config as pts_config
    except Exception as exc:  # pragma: no cover - only relevant if PaddockTS missing
        return f"PaddockTS config unavailable ({type(exc).__name__}: {exc})"

    if getattr(pts_config.config, "email", None):
        return "using existing PaddockTS SILO email/config"
    pts_config.config = evolve(pts_config.config, email=email)
    return f"using process-only SILO fallback email {email}"


def metrics(y_true, y_pred) -> dict:
    from emt.evaluation import metrics as _metrics

    return _metrics(y_true, y_pred)


def fmt(value, digits: int = 3) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    return "NA" if not np.isfinite(v) else f"{v:.{digits}f}"


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.{digits}f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def extract_original_model(out_dir: Path) -> Path:
    """Write the tracked original model6 artefact into ``out_dir``."""
    out = out_dir / "models" / "model6_original_from_git_HEAD.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        subprocess.run(
            ["git", "show", "HEAD:data/models/model6.joblib"],
            cwd=REPO_ROOT,
            check=True,
            stdout=f,
        )
    return out


def copy_retrained_model(out_dir: Path) -> Path:
    src = REPO_ROOT / "data/models/model6.joblib"
    dst = out_dir / "models" / "model6_retrained_oznet_current.joblib"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def fit_residual_corrector(
    train: pd.DataFrame,
    features: list[str],
    pred_col: str,
    method: str,
    knn_neighbors: int,
):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cols = [*features, pred_col]
    sub = train.dropna(subset=[pred_col, "obs_sm_pct"]).copy()
    X = sub[cols].replace([np.inf, -np.inf], np.nan)
    y = sub["obs_sm_pct"].to_numpy(float) - sub[pred_col].to_numpy(float)
    if method == "knn":
        est = KNeighborsRegressor(
            n_neighbors=max(1, min(int(knn_neighbors), len(sub))),
            weights="uniform",
        )
    elif method == "ridge":
        est = Ridge(alpha=1.0)
    else:
        raise ValueError(f"unknown calibration method: {method}")
    corrector = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    corrector.fit(X, y)
    return corrector, cols


def apply_corrector(
    df: pd.DataFrame,
    corrector,
    corrector_cols: list[str],
    pred_col: str,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    correction = corrector.predict(df[corrector_cols].replace([np.inf, -np.inf], np.nan))
    return np.clip(df[pred_col].to_numpy(float) + correction, clip_min, clip_max)


def groupkfold_calibrated_predictions(
    df: pd.DataFrame,
    features: list[str],
    pred_col: str,
    method: str,
    knn_neighbors: int,
    n_splits: int,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    groups = df["point"].astype(str)
    n_groups = groups.nunique()
    n_splits = max(2, min(int(n_splits), int(n_groups)))
    cv = GroupKFold(n_splits=n_splits)
    preds = np.full(len(df), np.nan, dtype=float)
    for train_idx, test_idx in cv.split(df, df["obs_sm_pct"], groups):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        corrector, cols = fit_residual_corrector(train, features, pred_col, method, knn_neighbors)
        preds[test_idx] = apply_corrector(test, corrector, cols, pred_col, clip_min, clip_max)
    return preds


def baseline_result(
    name: str,
    model,
    df: pd.DataFrame,
    features: list[str],
    method: str,
    knn_neighbors: int,
    n_splits: int,
    clip_min: float,
    clip_max: float,
):
    work = df.copy()
    pred_col = f"{name}_global_sm_pct"
    X = work[features].replace([np.inf, -np.inf], np.nan)
    ok = np.isfinite(X).all(axis=1).to_numpy()
    work[pred_col] = np.nan
    work.loc[ok, pred_col] = model.predict(X.loc[ok, features])

    corrector, corrector_cols = fit_residual_corrector(work, features, pred_col, method, knn_neighbors)
    work[f"{name}_local_insample_sm_pct"] = apply_corrector(
        work, corrector, corrector_cols, pred_col, clip_min, clip_max
    )
    work[f"{name}_local_groupkfold_sm_pct"] = groupkfold_calibrated_predictions(
        work,
        features,
        pred_col,
        method,
        knn_neighbors,
        n_splits,
        clip_min,
        clip_max,
    )
    out_metrics = {
        "global": metrics(work["obs_sm_pct"], work[pred_col]),
        "local_insample": metrics(work["obs_sm_pct"], work[f"{name}_local_insample_sm_pct"]),
        "local_groupkfold": metrics(work["obs_sm_pct"], work[f"{name}_local_groupkfold_sm_pct"]),
    }
    return {
        "name": name,
        "pred_col": pred_col,
        "corrector": corrector,
        "corrector_cols": corrector_cols,
        "point_predictions": work[
            [
                "point",
                "date",
                "lon",
                "lat",
                "obs_sm_pct",
                pred_col,
                f"{name}_local_insample_sm_pct",
                f"{name}_local_groupkfold_sm_pct",
            ]
        ].copy(),
        "metrics": out_metrics,
    }


def metric_rows(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for label, result in results.items():
        for mode, m in result["metrics"].items():
            rows.append(
                {
                    "baseline": label,
                    "prediction": mode,
                    "rmse": m["rmse"],
                    "ubrmse": m["ubrmse"],
                    "bias": m["bias"],
                    "r": m["r"],
                    "nse_r2": m["nse"],
                    "n": m["n"],
                }
            )
    return pd.DataFrame(rows)


def comparison_rows(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    pairs = [
        ("global", "global"),
        ("local_insample", "local_insample"),
        ("local_groupkfold", "local_groupkfold"),
    ]
    for original_mode, retrained_mode in pairs:
        o = results["original"]["metrics"][original_mode]
        r = results["retrained"]["metrics"][retrained_mode]
        rows.append(
            {
                "comparison": f"retrained - original ({original_mode})",
                "delta_rmse": r["rmse"] - o["rmse"],
                "delta_ubrmse": r["ubrmse"] - o["ubrmse"],
                "delta_abs_bias": abs(r["bias"]) - abs(o["bias"]),
                "delta_r": r["r"] - o["r"],
                "delta_nse_r2": r["nse"] - o["nse"],
            }
        )
    return pd.DataFrame(rows)


def output_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "models": root / "models",
        "point": root / "point_calibration",
        "maps": root / "maps",
        "figures": root / "figures",
    }
    map_subdirs = [
        "global_original",
        "local_original",
        "global_retrained",
        "local_retrained",
        "global_retrained_minus_original",
        "local_retrained_minus_original",
        "local_gain_original",
        "local_gain_retrained",
        "multiband",
    ]
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    for sub in map_subdirs:
        (dirs["maps"] / sub).mkdir(parents=True, exist_ok=True)
    return dirs


def write_single_band(path: Path, array: np.ndarray, profile: dict, description: str) -> None:
    import rasterio

    prof = profile.copy()
    prof.update(driver="GTiff", count=1, dtype="float32", nodata=NODATA, compress="deflate", predictor=3)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.where(np.isfinite(array), array, NODATA).astype("float32"), 1)
        dst.set_band_description(1, description)


def write_multiband(path: Path, arrays: list[np.ndarray], profile: dict, descriptions: list[str]) -> None:
    import rasterio

    prof = profile.copy()
    prof.update(
        driver="GTiff",
        count=len(arrays),
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
    )
    with rasterio.open(path, "w", **prof) as dst:
        for idx, (arr, desc) in enumerate(zip(arrays, descriptions), 1):
            dst.write(np.where(np.isfinite(arr), arr, NODATA).astype("float32"), idx)
            dst.set_band_description(idx, desc)


def write_figure(day: str, arrays: dict[str, np.ndarray], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sm_arrays = [
        arrays["global_original"],
        arrays["local_original"],
        arrays["global_retrained"],
        arrays["local_retrained"],
    ]
    combined = np.concatenate([a[np.isfinite(a)] for a in sm_arrays if np.isfinite(a).any()])
    vmin, vmax = np.nanpercentile(combined, [2, 98]) if combined.size else (0, 60)
    diff_arrays = [
        arrays["global_retrained_minus_original"],
        arrays["local_retrained_minus_original"],
    ]
    diff_vals = np.concatenate([a[np.isfinite(a)] for a in diff_arrays if np.isfinite(a).any()])
    dlim = float(np.nanpercentile(np.abs(diff_vals), 98)) if diff_vals.size else 1.0
    dlim = max(dlim, 0.5)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = [
        ("Original global", arrays["global_original"], "YlGnBu", vmin, vmax, "SM (%)"),
        ("Original + local calibration", arrays["local_original"], "YlGnBu", vmin, vmax, "SM (%)"),
        ("Local gain on original", arrays["local_gain_original"], "coolwarm", -dlim, dlim, "%"),
        ("Retrained global", arrays["global_retrained"], "YlGnBu", vmin, vmax, "SM (%)"),
        ("Retrained + local calibration", arrays["local_retrained"], "YlGnBu", vmin, vmax, "SM (%)"),
        ("Retrained-local minus original-local", arrays["local_retrained_minus_original"], "coolwarm", -dlim, dlim, "%"),
    ]
    for ax, (title, arr, cmap, lo, hi, label) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75, label=label)
    fig.suptitle(f"Global retrain/local calibration comparison — {day}", y=1.02)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_maps(
    *,
    df: pd.DataFrame,
    features: list[str],
    original_model,
    retrained_model,
    results: dict[str, dict],
    dense_root: Path,
    dirs: dict[str, Path],
    clip_min: float,
    clip_max: float,
) -> pd.DataFrame:
    from PaddockTS.query import Query
    import rasterio

    from soilmoisture_points_validation.make_sm_comparison_maps import (
        _open_template,
        bbox_from_predictions,
        build_dynamic_layers,
        build_static_layers,
        grid_dataframe,
        query_stub,
    )

    validation_dir = dense_root / "soilmoisture_points_validation"
    dates = sorted(df["date"].astype(str).unique())
    bbox = bbox_from_predictions(df)
    first_tif = validation_dir / "tifs" / f"soil_moisture_{dates[0]}.tif"
    template = _open_template(first_tif)
    q = Query(
        bbox=list(bbox),
        start=pd.Timestamp(dates[0]).date(),
        end=pd.Timestamp(dates[-1]).date(),
        stub=query_stub(bbox, dates[0], dates[-1]),
    )
    static_layers = build_static_layers(q, template)
    dynamic_by_day = build_dynamic_layers(q, dates, bbox, template)

    rows = []
    for i, day in enumerate(dates, 1):
        print(f"writing comparison maps {i}/{len(dates)}: {day}", flush=True)
        src_tif = validation_dir / "tifs" / f"soil_moisture_{day}.tif"
        with rasterio.open(src_tif) as src:
            profile = src.profile.copy()
            legacy = src.read(1).astype("float32")
            if src.nodata is not None:
                legacy = np.where(legacy == src.nodata, np.nan, legacy)

        grid = grid_dataframe(features, static_layers, dynamic_by_day[day], legacy)
        X = grid[features].replace([np.inf, -np.inf], np.nan)
        valid = np.isfinite(legacy.ravel()) & np.isfinite(X).all(axis=1).to_numpy()

        arrs = {name: np.full(legacy.size, np.nan, dtype="float32") for name in [
            "global_original",
            "global_retrained",
            "local_original",
            "local_retrained",
        ]}

        if valid.any():
            X_valid = X.loc[valid, features]
            orig_global = original_model.predict(X_valid).astype("float32")
            retr_global = retrained_model.predict(X_valid).astype("float32")
            arrs["global_original"][valid] = np.clip(orig_global, clip_min, clip_max)
            arrs["global_retrained"][valid] = np.clip(retr_global, clip_min, clip_max)

            for label, pred_vals, out_key in [
                ("original", arrs["global_original"][valid], "local_original"),
                ("retrained", arrs["global_retrained"][valid], "local_retrained"),
            ]:
                tmp = X_valid.copy()
                pred_col = results[label]["pred_col"]
                tmp[pred_col] = pred_vals
                corrected = apply_corrector(
                    tmp,
                    results[label]["corrector"],
                    results[label]["corrector_cols"],
                    pred_col,
                    clip_min,
                    clip_max,
                )
                arrs[out_key][valid] = corrected.astype("float32")

        arrays = {k: v.reshape(legacy.shape) for k, v in arrs.items()}
        arrays["global_retrained_minus_original"] = arrays["global_retrained"] - arrays["global_original"]
        arrays["local_retrained_minus_original"] = arrays["local_retrained"] - arrays["local_original"]
        arrays["local_gain_original"] = arrays["local_original"] - arrays["global_original"]
        arrays["local_gain_retrained"] = arrays["local_retrained"] - arrays["global_retrained"]

        file_map = {
            "global_original": f"global_original_model6_{day}.tif",
            "local_original": f"local_calibrated_original_{day}.tif",
            "global_retrained": f"global_retrained_model6_{day}.tif",
            "local_retrained": f"local_calibrated_retrained_{day}.tif",
            "global_retrained_minus_original": f"global_retrained_minus_original_{day}.tif",
            "local_retrained_minus_original": f"local_retrained_minus_original_{day}.tif",
            "local_gain_original": f"local_gain_original_{day}.tif",
            "local_gain_retrained": f"local_gain_retrained_{day}.tif",
        }
        for key, filename in file_map.items():
            write_single_band(dirs["maps"] / key / filename, arrays[key], profile, key)

        band_keys = [
            "global_original",
            "local_original",
            "global_retrained",
            "local_retrained",
            "global_retrained_minus_original",
            "local_retrained_minus_original",
            "local_gain_original",
            "local_gain_retrained",
        ]
        write_multiband(
            dirs["maps"] / "multiband" / f"model6_global_retrain_local_calibration_{day}.tif",
            [arrays[k] for k in band_keys],
            profile,
            band_keys,
        )
        write_figure(day, arrays, dirs["figures"] / f"model6_global_retrain_local_calibration_{day}.png")

        row = {"date": day, "valid_pixels": int(np.isfinite(arrays["global_original"]).sum())}
        for key in band_keys:
            arr = arrays[key]
            row[f"{key}_mean"] = float(np.nanmean(arr))
            row[f"{key}_min"] = float(np.nanmin(arr))
            row[f"{key}_max"] = float(np.nanmax(arr))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(dirs["root"] / "map_summary.csv", index=False)
    return summary


def write_report(
    *,
    out_dir: Path,
    feature_table: Path,
    dense_root: Path,
    method: str,
    knn_neighbors: int,
    n_splits: int,
    original_params: dict,
    retrained_params: dict,
    metric_table: pd.DataFrame,
    delta_table: pd.DataFrame,
    legacy_pred_diff: dict,
    map_summary: pd.DataFrame | None,
    silo_status: str,
) -> None:
    global_delta = delta_table[delta_table["comparison"].str.contains("(global)", regex=False)]
    group_delta = delta_table[delta_table["comparison"].str.contains("(local_groupkfold)", regex=False)]

    def judgement(delta_nse: float, delta_rmse: float) -> str:
        if delta_nse > 0.01 and delta_rmse < -0.05:
            return "improved"
        if delta_nse < -0.01 and delta_rmse > 0.05:
            return "worsened"
        return "roughly unchanged"

    gd = global_delta.iloc[0] if len(global_delta) else None
    cd = group_delta.iloc[0] if len(group_delta) else None
    global_judgement = judgement(float(gd["delta_nse_r2"]), float(gd["delta_rmse"])) if gd is not None else "unknown"
    local_judgement = judgement(float(cd["delta_nse_r2"]), float(cd["delta_rmse"])) if cd is not None else "unknown"

    report = f"""# model6 global retrain vs dense local calibration

Generated: {datetime.now().isoformat(timespec="seconds")}

Output folder: `{out_dir}`

Dense input table: `{feature_table}`

This comparison uses the same dense point/date model-input table for both
baselines:

- **Original global model6**: extracted from `HEAD:data/models/model6.joblib`.
- **Retrained global model6**: current working `data/models/model6.joblib`, fitted
  on the locally rebuilt OzNet training table with a larger leaf cap.
- **Local calibration**: `{method}` residual corrector using model6 inputs plus
  the relevant global prediction. The primary less-optimistic score is
  point-group held-out GroupKFold with `{n_splits}` folds.

SILO configuration for any map-input generation: {silo_status}.

## Short answer

- Global retraining changed dense-site global-only performance: **{global_judgement}**.
- After dense local residual calibration, retraining changed point-group held-out
  local calibration performance: **{local_judgement}**.

## Global model parameters

The original global model is still available from git and was copied into
`models/model6_original_from_git_HEAD.joblib`.

| parameter | original global | retrained global |
|---|---:|---:|
| max_leaf_nodes | {original_params.get("max_leaf_nodes")} | {retrained_params.get("max_leaf_nodes")} |
| min_samples_leaf | {original_params.get("min_samples_leaf")} | {retrained_params.get("min_samples_leaf")} |
| max_iter | {original_params.get("max_iter")} | {retrained_params.get("max_iter")} |
| max_features | {original_params.get("max_features")} | {retrained_params.get("max_features")} |
| learning_rate | {original_params.get("learning_rate")} | {retrained_params.get("learning_rate")} |
| l2_regularization | {original_params.get("l2_regularization")} | {retrained_params.get("l2_regularization")} |

Existing dense-validation `pred_sm_pct` vs original model re-prediction:

```json
{json.dumps(legacy_pred_diff, indent=2)}
```

## Point-level metrics

Bias follows the EMT convention: prediction minus observation.

{markdown_table(metric_table)}

## Retrained minus original deltas

Positive delta NSE/R² is good; negative delta RMSE/ubRMSE/abs_bias is good.

{markdown_table(delta_table)}

## Outputs

- `models/` — original-from-git and current retrained model artefact copies plus JSON params.
- `point_calibration/point_predictions_comparison.csv` — dense point predictions for both global baselines and local calibration variants.
- `point_calibration/metrics_summary.csv` — point-level metric table.
- `point_calibration/retrained_minus_original_deltas.csv` — metric deltas.
- `maps/` — GeoTIFF maps for the global and locally calibrated predictions, plus difference rasters.
- `figures/` — quick-look PNG comparison panels per date.
- `map_summary.csv` — map-level mean/min/max for each raster layer.

## Interpretation note

The local in-sample calibration rows tell us how strongly the dense data can
calibrate the site if all local observations are allowed into the residual
surface. The point-group held-out rows are the better test of whether that local
calibration transfers to unseen point locations inside the same dense site.
"""
    if map_summary is not None and not map_summary.empty:
        report += "\n## Map-level mean differences\n\n"
        report += markdown_table(
            map_summary[
                [
                    "date",
                    "global_retrained_minus_original_mean",
                    "local_retrained_minus_original_mean",
                    "local_gain_original_mean",
                    "local_gain_retrained_mean",
                    "valid_pixels",
                ]
            ]
        )
        report += "\n"
    (out_dir / "README.md").write_text(report)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--feature-table", type=Path)
    parser.add_argument("--out-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--method", choices=["knn", "ridge"], default="knn")
    parser.add_argument("--knn-neighbors", type=int, default=12)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=60.0)
    parser.add_argument("--silo-email", default=DEFAULT_SILO_EMAIL)
    parser.add_argument("--skip-maps", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
    silo_status = process_only_silo_email(args.silo_email)
    print(f"SILO: {silo_status}", flush=True)

    from emt.model6 import model as model6

    feature_table = args.feature_table or (
        args.dense_root / "Validation_2stage" / "stage1_dense_unseen_validation" / "point_date_model_inputs.csv"
    )
    out_dir = args.dense_root / args.out_name
    dirs = output_dirs(out_dir)

    df = pd.read_csv(feature_table).replace([np.inf, -np.inf], np.nan)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    required = ["point", "date", "lon", "lat", "obs_sm_pct", *model6.FEATURES]
    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"dense rows for comparison: {len(df)} ({df['point'].nunique()} points, {df['date'].nunique()} dates)", flush=True)

    original_model_path = extract_original_model(out_dir)
    retrained_model_path = copy_retrained_model(out_dir)
    original_model = joblib.load(original_model_path)
    retrained_model = joblib.load(retrained_model_path)
    original_params = original_model.get_params()
    retrained_params = retrained_model.get_params()
    (dirs["models"] / "model_params_original.json").write_text(json.dumps(original_params, indent=2, sort_keys=True, default=str) + "\n")
    (dirs["models"] / "model_params_retrained.json").write_text(json.dumps(retrained_params, indent=2, sort_keys=True, default=str) + "\n")

    features = list(model6.FEATURES)
    results = {
        "original": baseline_result(
            "original",
            original_model,
            df,
            features,
            args.method,
            args.knn_neighbors,
            args.n_splits,
            args.clip_min,
            args.clip_max,
        ),
        "retrained": baseline_result(
            "retrained",
            retrained_model,
            df,
            features,
            args.method,
            args.knn_neighbors,
            args.n_splits,
            args.clip_min,
            args.clip_max,
        ),
    }

    point_pred = df[["point", "date", "lon", "lat", "obs_sm_pct", "pred_sm_pct"]].copy()
    for label, result in results.items():
        point_pred = point_pred.merge(
            result["point_predictions"],
            on=["point", "date", "lon", "lat", "obs_sm_pct"],
            how="left",
        )
    point_pred.to_csv(dirs["point"] / "point_predictions_comparison.csv", index=False)

    metric_table = metric_rows(results)
    delta_table = comparison_rows(results)
    metric_table.to_csv(dirs["point"] / "metrics_summary.csv", index=False)
    delta_table.to_csv(dirs["point"] / "retrained_minus_original_deltas.csv", index=False)

    legacy_diff = {}
    if "pred_sm_pct" in df.columns:
        orig_pred = results["original"]["point_predictions"]["original_global_sm_pct"].to_numpy(float)
        legacy = df["pred_sm_pct"].to_numpy(float)
        ok = np.isfinite(orig_pred) & np.isfinite(legacy)
        legacy_diff = {
            "n_compared": int(ok.sum()),
            "mean_existing_minus_original_reprediction": float(np.nanmean(legacy[ok] - orig_pred[ok])),
            "max_abs_existing_minus_original_reprediction": float(np.nanmax(np.abs(legacy[ok] - orig_pred[ok]))),
        }
        (dirs["point"] / "existing_pred_vs_original_reprediction.json").write_text(
            json.dumps(legacy_diff, indent=2, sort_keys=True) + "\n"
        )

    joblib.dump(results["original"]["corrector"], dirs["models"] / f"local_{args.method}_original_corrector.joblib")
    joblib.dump(results["retrained"]["corrector"], dirs["models"] / f"local_{args.method}_retrained_corrector.joblib")

    map_summary = None
    if not args.skip_maps:
        map_summary = generate_maps(
            df=df,
            features=features,
            original_model=original_model,
            retrained_model=retrained_model,
            results=results,
            dense_root=args.dense_root,
            dirs=dirs,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
        )

    write_report(
        out_dir=out_dir,
        feature_table=feature_table,
        dense_root=args.dense_root,
        method=args.method,
        knn_neighbors=args.knn_neighbors,
        n_splits=args.n_splits,
        original_params=original_params,
        retrained_params=retrained_params,
        metric_table=metric_table,
        delta_table=delta_table,
        legacy_pred_diff=legacy_diff,
        map_summary=map_summary,
        silo_status=silo_status,
    )
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
