"""Dense-point validation and local-data spiking analysis for EMT model6.

This script implements the two-stage validation plan:

1. independent dense-point validation + model-input terrain/input bias mapping;
2. sensitivity to local training-data "spiking" through residual calibration
   experiments using increasing amounts of local observations.

Outputs are written outside the repo by default:

    /Volumes/Dmitry_work/borevitz_projects/model6_dense_validation_spiking

The analysis deliberately uses model6 inputs sampled from the prediction system
itself (terrain, SLGA soil, SMIPS lookbacks, antecedent SILO and seasonality),
not the auxiliary terrain columns in the field CSV.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_VALIDATION_DIR = Path(
    "/Volumes/Dmitry_work/borevitz_projects/Data/soilmoisture_points_validation"
)
DEFAULT_INPUT_CSV = Path(
    "/Volumes/Dmitry_work/borevitz_projects/Data/soilmoisture_points_coordinates.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Volumes/Dmitry_work/borevitz_projects/model6_dense_validation_spiking"
)
NODATA = -9999.0
METEOROLOGY_INPUTS = [
    "rain_7",
    "rain_30",
    "rain_365",
    "ppet_30",
    "ppet_365",
    "vpd_30",
    "rain_365_anom",
]


def repo_branch() -> str | None:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    branch = result.stdout.strip()
    return branch or None


def require_emt_branch(allow_non_emt: bool = False) -> None:
    branch = repo_branch()
    if allow_non_emt or branch in (None, "EMT"):
        return
    raise SystemExit(
        f"This workflow is intended for the EMT branch; current branch is {branch!r}. "
        "Switch with `git switch EMT`, or pass --allow-non-emt."
    )


def fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except Exception:
        return "NA"
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"

    def cell(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "root": out_dir,
        "stage1": out_dir / "stage1_dense_unseen_validation",
        "stage1_figures": out_dir / "stage1_dense_unseen_validation" / "figures",
        "stage2": out_dir / "stage2_local_spiking_sensitivity",
        "stage2_figures": out_dir / "stage2_local_spiking_sensitivity" / "figures",
        "rasters": out_dir / "stage1_dense_unseen_validation" / "rasters",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_validation(validation_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(validation_dir / "predictions.csv")
    point_metrics = pd.read_csv(validation_dir / "metrics_per_point.csv")
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.date.astype(str)
    predictions["residual_obs_minus_pred"] = predictions["obs_sm_pct"] - predictions["pred_sm_pct"]
    return predictions, point_metrics


def bbox_from_predictions(predictions: pd.DataFrame, padding_deg: float = 0.002):
    return (
        float(predictions["lon"].min() - padding_deg),
        float(predictions["lat"].min() - padding_deg),
        float(predictions["lon"].max() + padding_deg),
        float(predictions["lat"].max() + padding_deg),
    )


def query_stub(bbox, start_day: str, end_day: str) -> str:
    b = "_".join(f"{v:.3f}" for v in bbox)
    return f"denseval_{b}_{start_day}_{end_day}".replace(".", "p").replace("-", "m")


def sample_model6_inputs(
    predictions: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    cache_csv: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Sample model6 input features for every point/date row."""
    from PaddockTS.query import Query
    from rasterio.enums import Resampling

    from emt.antecedent import antecedent_grid, antecedent_day_layers
    from emt.covariates import TERRAIN_VARS, sample_points, terrain_covariates
    from emt.model6 import model as model6
    from emt.slga import SOIL_VARS, soil_covariates
    from emt.smips import smips_cube

    if cache_csv.exists() and not force:
        cached = pd.read_csv(cache_csv)
        missing = [c for c in model6.FEATURES if c not in cached.columns]
        if not missing:
            print(f"using cached model-input table: {cache_csv}", flush=True)
            return cached

    dates = sorted(predictions["date"].unique())
    start_day = pd.Timestamp(dates[0]).date()
    end_day = pd.Timestamp(dates[-1]).date()
    q = Query(
        bbox=list(bbox),
        start=start_day,
        end=end_day,
        stub=query_stub(bbox, dates[0], dates[-1]),
    )

    print("sampling static terrain inputs ...", flush=True)
    terr = terrain_covariates(q)
    grid = terr["elevation"]

    print("sampling static SLGA soil inputs ...", flush=True)
    soil = soil_covariates(q)
    soil_on_grid = {}
    for var in SOIL_VARS:
        da = soil[var]
        if da.rio.crs is None:
            da = da.rio.write_crs(4326)
        soil_on_grid[var] = da.rio.reproject_match(grid, resampling=Resampling.nearest)

    print("fetching one SMIPS cube for all dates/lookbacks ...", flush=True)
    smips_start = (pd.Timestamp(dates[0]) - pd.Timedelta(days=365)).date()
    smips = smips_cube(smips_start, end_day, bbox, var="totalbucket").sortby("time")

    print("fetching one antecedent SILO grid for all dates ...", flush=True)
    ante = antecedent_grid(q, start_day, end_day)

    point_static: dict[str, dict] = {}
    for point, group in predictions.groupby("point"):
        lon = float(group["lon"].iloc[0])
        lat = float(group["lat"].iloc[0])
        vals = {"point": point, "lon": lon, "lat": lat}
        terr_pt = sample_points(terr, lon, lat)
        for var in TERRAIN_VARS:
            vals[var] = float(terr_pt[var].values)
        for var, da in soil_on_grid.items():
            vals[var] = float(sample_points(da, lon, lat).values)
        point_static[point] = vals

    frames = []
    for i, day in enumerate(dates, 1):
        print(f"  model-input date {i}/{len(dates)}: {day}", flush=True)
        ts = pd.Timestamp(day)
        upto = smips.sel(time=slice(None, ts))
        today = upto.isel(time=-1).rio.write_crs(4326)
        smips_layers = {
            "smips_totalbucket": today,
            "smips_7d": upto.isel(time=slice(-7, None)).mean("time").rio.write_crs(4326),
            "smips_30d": upto.isel(time=slice(-30, None)).mean("time").rio.write_crs(4326),
            "smips_365d": upto.isel(time=slice(-365, None)).mean("time").rio.write_crs(4326),
        }
        smips_layers["smips_anom"] = smips_layers["smips_totalbucket"] - smips_layers["smips_365d"]

        aligned = {}
        for name, da in smips_layers.items():
            aligned[name] = da.rio.reproject_match(grid, resampling=Resampling.nearest)
        for name, da in antecedent_day_layers(ante, ts.date()).items():
            if da.rio.crs is None:
                da = da.rio.write_crs(4326)
            aligned[name] = da.rio.reproject_match(grid, resampling=Resampling.nearest)

        doy = ts.dayofyear
        doy_sin = float(np.sin(2 * np.pi * doy / 365.25))
        doy_cos = float(np.cos(2 * np.pi * doy / 365.25))

        rows = []
        for row in predictions[predictions["date"] == day].itertuples(index=False):
            vals = dict(point_static[row.point])
            vals.update(
                {
                    "point": row.point,
                    "date": day,
                    "obs_sm_pct": float(row.obs_sm_pct),
                    "pred_sm_pct": float(row.pred_sm_pct),
                    "residual_obs_minus_pred": float(row.residual_obs_minus_pred),
                    "residual_pred_minus_obs": float(row.residual_pred_minus_obs),
                    "source_row": int(row.source_row),
                    "measurement_time": getattr(row, "measurement_time", None),
                    "water_mm": getattr(row, "water_mm", np.nan),
                    "doy_sin": doy_sin,
                    "doy_cos": doy_cos,
                }
            )
            for name, da in aligned.items():
                vals[name] = float(sample_points(da, vals["lon"], vals["lat"]).values)
            rows.append(vals)
        frames.append(pd.DataFrame(rows))

    out = pd.concat(frames, ignore_index=True)
    ordered = [
        "point",
        "date",
        "lon",
        "lat",
        "obs_sm_pct",
        "pred_sm_pct",
        "residual_obs_minus_pred",
        "residual_pred_minus_obs",
        "source_row",
        "measurement_time",
        "water_mm",
        *model6.FEATURES,
    ]
    out = out[[c for c in ordered if c in out.columns]]
    out.to_csv(cache_csv, index=False)
    print(f"wrote {cache_csv}", flush=True)
    return out


def compute_metrics(y_true, y_pred) -> dict:
    from emt.evaluation import metrics

    return metrics(y_true, y_pred)


def stage1_diagnostics(
    feature_rows: pd.DataFrame,
    point_metrics: pd.DataFrame,
    stage1_dir: Path,
    rasters_dir: Path,
    template_tif: Path,
    radius_m: float,
) -> dict:
    from emt.model6 import model as model6

    static_or_mean = (
        feature_rows.groupby("point", as_index=False)
        .agg(
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            observed_mean=("obs_sm_pct", "mean"),
            predicted_mean=("pred_sm_pct", "mean"),
            residual_mean=("residual_obs_minus_pred", "mean"),
            **{f"{c}_mean": (c, "mean") for c in model6.FEATURES},
        )
    )
    point_full = point_metrics.merge(static_or_mean, on="point", how="left")
    point_full["abs_bias"] = point_full["bias"].abs()
    point_full["quality_class"] = point_full["nse"].map(quality_class)
    point_full["rank_nse"] = point_full["nse"].rank(ascending=False, method="min")
    point_full.to_csv(stage1_dir / "point_metrics_with_model_inputs.csv", index=False)

    feature_rows.to_csv(stage1_dir / "point_date_model_inputs.csv", index=False)

    corr = feature_correlations(point_full, list(model6.FEATURES))
    corr.to_csv(stage1_dir / "model_input_quality_correlations.csv", index=False)

    bins = feature_bin_metrics(feature_rows, model6.FEATURES)
    bins.to_csv(stage1_dir / "model_input_bin_metrics.csv", index=False)

    rasterise_point_metrics(
        point_full,
        template_tif,
        rasters_dir / "point_quality_nse_r2.tif",
        rasters_dir / "point_quality_bias.tif",
        rasters_dir / "point_quality_rmse.tif",
        rasters_dir / "point_quality_metrics_multiband.tif",
        radius_m=radius_m,
    )
    make_stage1_figures(point_full, corr, stage1_dir / "figures")

    best = point_full.nlargest(10, "nse")[
        ["point", "nse", "r", "bias", "rmse", "lon", "lat", "quality_class"]
    ]
    worst = point_full.nsmallest(10, "nse")[
        ["point", "nse", "r", "bias", "rmse", "lon", "lat", "quality_class"]
    ]
    return {
        "point_metrics": point_full,
        "correlations": corr,
        "bins": bins,
        "best": best,
        "worst": worst,
    }


def quality_class(nse: float) -> str:
    if not np.isfinite(nse):
        return "missing"
    if nse >= 0.5:
        return "strong"
    if nse > 0:
        return "positive"
    if nse > -1:
        return "poor"
    return "very_poor"


def feature_correlations(point_full: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    rows = []
    targets = ["nse", "r", "bias", "abs_bias", "rmse", "ubrmse"]
    feature_cols = [f"{c}_mean" for c in model_features if f"{c}_mean" in point_full.columns]
    for feature in feature_cols:
        for target in targets:
            sub = point_full[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 5 or sub[feature].std() == 0 or sub[target].std() == 0:
                continue
            rows.append(
                {
                    "model_input": feature.removesuffix("_mean"),
                    "quality_metric": target,
                    "pearson_r": float(sub[feature].corr(sub[target], method="pearson")),
                    "spearman_r": float(sub[feature].corr(sub[target], method="spearman")),
                    "n_points": int(len(sub)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_pearson_r"] = out["pearson_r"].abs()
    return out.sort_values(["quality_metric", "abs_pearson_r"], ascending=[True, False])


def feature_bin_metrics(feature_rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        vals = feature_rows[feature].replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() < 10 or vals.nunique(dropna=True) < 3:
            continue
        try:
            bins = pd.qcut(vals, q=3, labels=["low", "mid", "high"], duplicates="drop")
        except ValueError:
            continue
        tmp = feature_rows.assign(bin=bins).dropna(subset=["bin"])
        for level, group in tmp.groupby("bin", observed=False):
            m = compute_metrics(group["obs_sm_pct"], group["pred_sm_pct"])
            rows.append(
                {
                    "model_input": feature,
                    "bin": str(level),
                    "value_min": float(group[feature].min()),
                    "value_median": float(group[feature].median()),
                    "value_max": float(group[feature].max()),
                    **m,
                }
            )
    return pd.DataFrame(rows)


def rasterise_point_metrics(
    point_full: pd.DataFrame,
    template_tif: Path,
    nse_tif: Path,
    bias_tif: Path,
    rmse_tif: Path,
    multi_tif: Path,
    radius_m: float = 45.0,
) -> None:
    import rasterio
    from pyproj import Transformer

    bands = ["nse", "r2", "r", "bias", "rmse", "ubrmse", "rank_nse"]
    with rasterio.open(template_tif) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
    if crs is None:
        raise ValueError(f"template has no CRS: {template_tif}")

    arrays = {name: np.full((height, width), np.nan, dtype="float32") for name in bands}
    nearest_dist = np.full((height, width), np.inf, dtype="float32")
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = transformer.transform(point_full["lon"].values, point_full["lat"].values)
    pixel_size = max(abs(transform.a), abs(transform.e))
    radius_px = max(0, int(math.ceil(radius_m / pixel_size)))

    for i, row in point_full.reset_index(drop=True).iterrows():
        x, y = float(xs[i]), float(ys[i])
        col_f, row_f = ~transform * (x, y)
        row0 = int(math.floor(row_f))
        col0 = int(math.floor(col_f))
        for rr in range(max(0, row0 - radius_px), min(height - 1, row0 + radius_px) + 1):
            for cc in range(max(0, col0 - radius_px), min(width - 1, col0 + radius_px) + 1):
                cx, cy = rasterio.transform.xy(transform, rr, cc, offset="center")
                dist = math.hypot(float(cx) - x, float(cy) - y)
                if dist > radius_m or dist >= float(nearest_dist[rr, cc]):
                    continue
                nearest_dist[rr, cc] = dist
                for band in bands:
                    value = row.get(band, np.nan)
                    arrays[band][rr, cc] = np.nan if pd.isna(value) else float(value)

    def write(path: Path, band_names: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        prof = profile.copy()
        prof.update(
            driver="GTiff",
            count=len(band_names),
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
            predictor=3,
        )
        with rasterio.open(path, "w", **prof) as dst:
            for idx, name in enumerate(band_names, 1):
                arr = arrays[name]
                dst.write(np.where(np.isfinite(arr), arr, NODATA).astype("float32"), idx)
                dst.set_band_description(idx, name)

    write(nse_tif, ["nse"])
    write(bias_tif, ["bias"])
    write(rmse_tif, ["rmse"])
    write(multi_tif, bands)


def make_stage1_figures(point_full: pd.DataFrame, corr: pd.DataFrame, fig_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"stage1 figures skipped: {type(exc).__name__}: {exc}", flush=True)
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    c = corr[corr["quality_metric"] == "nse"].copy()
    if not c.empty:
        c = c.sort_values("abs_pearson_r", ascending=False).head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(c["model_input"], c["pearson_r"], color=np.where(c["pearson_r"] >= 0, "#3b82f6", "#ef4444"))
        ax.axvline(0, color="0.3", linewidth=0.8)
        ax.set(xlabel="Pearson r with point NSE/R²", title="Model-input association with point prediction quality")
        fig.tight_layout()
        fig.savefig(fig_dir / "nse_model_input_correlations.png", dpi=160)
        plt.close(fig)

    plot_features = [f for f in ["twi_mean", "slope_mean", "hli_mean", "soil_awc_mean"] if f in point_full.columns]
    if plot_features:
        fig, axes = plt.subplots(1, len(plot_features), figsize=(4 * len(plot_features), 3.5), squeeze=False)
        for ax, feature in zip(axes.ravel(), plot_features):
            ax.scatter(point_full[feature], point_full["nse"], c=point_full["bias"], cmap="coolwarm", s=35, alpha=0.85)
            ax.axhline(0, color="0.3", linewidth=0.8, linestyle="--")
            ax.set(xlabel=feature.removesuffix("_mean"), ylabel="point NSE/R²")
        fig.suptitle("Where model6 transfers well or poorly in model-input space", y=1.03)
        fig.tight_layout()
        fig.savefig(fig_dir / "nse_vs_selected_model_inputs.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


@dataclass
class SelectionContext:
    point_static: pd.DataFrame
    feature_matrix: pd.DataFrame
    xy: pd.DataFrame


def selection_context(feature_rows: pd.DataFrame) -> SelectionContext:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from pyproj import Transformer

    static_features = [
        c
        for c in [
            "elevation",
            "slope",
            "northness",
            "eastness",
            "twi",
            "hli",
            "accumulation",
            "soil_clay",
            "soil_sand",
            "soil_awc",
            "soil_bdw",
        ]
        if c in feature_rows.columns
    ]
    point_static = (
        feature_rows.groupby("point", as_index=False)
        .agg(lon=("lon", "mean"), lat=("lat", "mean"), **{c: (c, "mean") for c in static_features})
        .set_index("point")
    )
    mat = point_static[static_features].replace([np.inf, -np.inf], np.nan)
    X = SimpleImputer(strategy="median").fit_transform(mat)
    X = StandardScaler().fit_transform(X)
    feature_matrix = pd.DataFrame(X, index=point_static.index, columns=static_features)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32755", always_xy=True)
    x, y = transformer.transform(point_static["lon"].values, point_static["lat"].values)
    xy = pd.DataFrame({"x": x, "y": y}, index=point_static.index)
    return SelectionContext(point_static=point_static, feature_matrix=feature_matrix, xy=xy)


def select_points(
    target: str,
    candidates: list[str],
    k: int,
    strategy: str,
    ctx: SelectionContext,
    rng: np.random.Generator,
) -> list[str]:
    if k <= 0:
        return []
    if k >= len(candidates):
        return list(candidates)

    if strategy == "random":
        return list(rng.choice(candidates, size=k, replace=False))

    if strategy == "nearest":
        t = ctx.xy.loc[target]
        dist = ((ctx.xy.loc[candidates]["x"] - t["x"]) ** 2 + (ctx.xy.loc[candidates]["y"] - t["y"]) ** 2) ** 0.5
        return list(dist.sort_values().head(k).index)

    if strategy == "terrain_similar":
        t = ctx.feature_matrix.loc[target]
        diff = ctx.feature_matrix.loc[candidates] - t
        dist = (diff**2).sum(axis=1) ** 0.5
        return list(dist.sort_values().head(k).index)

    if strategy == "terrain_stratified":
        # Greedy maximin spread in model-input space. This samples the local
        # input-space envelope rather than only the target's nearest neighbours.
        X = ctx.feature_matrix.loc[candidates].copy()
        selected = [str(((X - X.median()).pow(2).sum(axis=1)).idxmin())]
        while len(selected) < k:
            remaining = [p for p in candidates if p not in selected]
            dist_to_selected = []
            for p in remaining:
                d = ((X.loc[selected] - X.loc[p]) ** 2).sum(axis=1).pow(0.5).min()
                dist_to_selected.append((p, float(d)))
            selected.append(max(dist_to_selected, key=lambda x: x[1])[0])
        return selected

    raise ValueError(f"unknown strategy: {strategy}")


def fit_corrected_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    method: str,
) -> np.ndarray:
    if method == "bias_only":
        correction = float(train["residual_obs_minus_pred"].mean()) if len(train) else 0.0
        return test["pred_sm_pct"].to_numpy(dtype=float) + correction

    if method == "ridge_residual":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        train_sub = train.dropna(subset=["residual_obs_minus_pred"]).copy()
        if train_sub.empty:
            return test["pred_sm_pct"].to_numpy(dtype=float)
        X_train = train_sub[feature_cols + ["pred_sm_pct"]].replace([np.inf, -np.inf], np.nan)
        y_train = train_sub["residual_obs_minus_pred"].to_numpy(dtype=float)
        X_test = test[feature_cols + ["pred_sm_pct"]].replace([np.inf, -np.inf], np.nan)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
        model.fit(X_train, y_train)
        return test["pred_sm_pct"].to_numpy(dtype=float) + model.predict(X_test)

    raise ValueError(f"unknown calibration method: {method}")


def stage2_spatial_spiking(
    feature_rows: pd.DataFrame,
    stage2_dir: Path,
    repeats: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    from emt.model6 import model as model6

    rng = np.random.default_rng(seed)
    points = sorted(feature_rows["point"].unique())
    ctx = selection_context(feature_rows)
    feature_cols = list(model6.FEATURES)
    spike_sizes = [0, 1, 3, 5, 10, 20, 40]
    methods = ["bias_only", "ridge_residual"]
    strategies = ["nearest", "terrain_similar", "terrain_stratified", "random"]

    target_results = []
    prediction_rows = []
    for ti, target in enumerate(points, 1):
        test = feature_rows[feature_rows["point"] == target].copy()
        base = compute_metrics(test["obs_sm_pct"], test["pred_sm_pct"])
        base_record = {"target_point": target, "strategy": "baseline", "method": "none", "spike_points": 0, "replicate": 0}
        base_record.update({f"baseline_{k}": v for k, v in base.items()})
        base_record.update(base)
        target_results.append(base_record)
        if ti % 10 == 0:
            print(f"  spatial spiking target {ti}/{len(points)}", flush=True)

        candidates = [p for p in points if p != target]
        for k in spike_sizes:
            if k == 0:
                continue
            for strategy in strategies:
                reps = repeats if strategy == "random" else 1
                for rep in range(reps):
                    selected = select_points(target, candidates, k, strategy, ctx, rng)
                    train = feature_rows[feature_rows["point"].isin(selected)].copy()
                    for method in methods:
                        pred_corr = fit_corrected_predictions(train, test, feature_cols, method)
                        m = compute_metrics(test["obs_sm_pct"], pred_corr)
                        record = {
                            "target_point": target,
                            "strategy": strategy,
                            "method": method,
                            "spike_points": k,
                            "spike_observations": int(len(train)),
                            "replicate": rep,
                            "selected_points": ";".join(selected),
                        }
                        record.update(m)
                        record.update({f"baseline_{kk}": vv for kk, vv in base.items()})
                        record["delta_nse"] = record["nse"] - base["nse"]
                        record["delta_rmse"] = record["rmse"] - base["rmse"]
                        record["delta_abs_bias"] = abs(record["bias"]) - abs(base["bias"])
                        target_results.append(record)
                        for row, pc in zip(test.itertuples(index=False), pred_corr):
                            prediction_rows.append(
                                {
                                    "target_point": target,
                                    "date": row.date,
                                    "strategy": strategy,
                                    "method": method,
                                    "spike_points": k,
                                    "replicate": rep,
                                    "obs_sm_pct": row.obs_sm_pct,
                                    "baseline_pred_sm_pct": row.pred_sm_pct,
                                    "corrected_pred_sm_pct": float(pc),
                                }
                            )

    target_df = pd.DataFrame(target_results)
    pred_df = pd.DataFrame(prediction_rows)
    target_df.to_csv(stage2_dir / "spatial_spiking_target_results.csv", index=False)
    pred_df.to_csv(stage2_dir / "spatial_spiking_predictions.csv", index=False)

    learning = summarize_spiking(target_df)
    learning.to_csv(stage2_dir / "spatial_spiking_learning_curve.csv", index=False)
    return {"target_results": target_df, "predictions": pred_df, "learning": learning}


def summarize_spiking(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = results.groupby(["strategy", "method", "spike_points"], dropna=False, sort=True)
    for (strategy, method, k), g in grouped:
        if strategy == "baseline":
            continue
        rows.append(
            {
                "strategy": strategy,
                "method": method,
                "spike_points": int(k),
                "n_target_fits": int(len(g)),
                "median_nse": float(g["nse"].median()),
                "mean_nse": float(g["nse"].mean()),
                "positive_nse_fits": int((g["nse"] > 0).sum()),
                "median_rmse": float(g["rmse"].median()),
                "median_abs_bias": float(g["bias"].abs().median()),
                "median_delta_nse": float(g["delta_nse"].median()),
                "mean_delta_nse": float(g["delta_nse"].mean()),
                "median_delta_rmse": float(g["delta_rmse"].median()),
                "median_delta_abs_bias": float(g["delta_abs_bias"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "strategy", "spike_points"])


def stage2_temporal_self_spiking(feature_rows: pd.DataFrame, stage2_dir: Path) -> dict[str, pd.DataFrame]:
    from emt.model6 import model as model6

    feature_cols = list(model6.FEATURES)
    methods = ["bias_only", "ridge_residual"]
    rows = []
    pred_rows = []
    for point, group in feature_rows.groupby("point"):
        g = group.sort_values("date").reset_index(drop=True)
        for n_train_dates in [1, 2, 3, 4, 5]:
            if len(g) <= n_train_dates + 1:
                continue
            train = g.iloc[:n_train_dates].copy()
            test = g.iloc[n_train_dates:].copy()
            base = compute_metrics(test["obs_sm_pct"], test["pred_sm_pct"])
            for method in methods:
                pred_corr = fit_corrected_predictions(train, test, feature_cols, method)
                m = compute_metrics(test["obs_sm_pct"], pred_corr)
                record = {
                    "point": point,
                    "method": method,
                    "training_dates": n_train_dates,
                    "training_observations": len(train),
                    "test_observations": len(test),
                    **m,
                    **{f"baseline_{k}": v for k, v in base.items()},
                }
                record["delta_nse"] = record["nse"] - base["nse"]
                record["delta_rmse"] = record["rmse"] - base["rmse"]
                record["delta_abs_bias"] = abs(record["bias"]) - abs(base["bias"])
                rows.append(record)
                for row, pc in zip(test.itertuples(index=False), pred_corr):
                    pred_rows.append(
                        {
                            "point": point,
                            "date": row.date,
                            "method": method,
                            "training_dates": n_train_dates,
                            "obs_sm_pct": row.obs_sm_pct,
                            "baseline_pred_sm_pct": row.pred_sm_pct,
                            "corrected_pred_sm_pct": float(pc),
                        }
                    )
    result = pd.DataFrame(rows)
    pred = pd.DataFrame(pred_rows)
    result.to_csv(stage2_dir / "temporal_self_spiking_point_results.csv", index=False)
    pred.to_csv(stage2_dir / "temporal_self_spiking_predictions.csv", index=False)

    learning_rows = []
    for (method, n_dates), g in result.groupby(["method", "training_dates"]):
        learning_rows.append(
            {
                "method": method,
                "training_dates": int(n_dates),
                "n_points": int(len(g)),
                "median_nse": float(g["nse"].median()),
                "mean_nse": float(g["nse"].mean()),
                "positive_nse_points": int((g["nse"] > 0).sum()),
                "median_rmse": float(g["rmse"].median()),
                "median_abs_bias": float(g["bias"].abs().median()),
                "median_delta_nse": float(g["delta_nse"].median()),
                "mean_delta_nse": float(g["delta_nse"].mean()),
                "median_delta_rmse": float(g["delta_rmse"].median()),
                "median_delta_abs_bias": float(g["delta_abs_bias"].median()),
            }
        )
    learning = pd.DataFrame(learning_rows).sort_values(["method", "training_dates"])
    learning.to_csv(stage2_dir / "temporal_self_spiking_learning_curve.csv", index=False)
    return {"point_results": result, "predictions": pred, "learning": learning}


def make_stage2_figures(spatial_learning: pd.DataFrame, temporal_learning: pd.DataFrame, fig_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"stage2 figures skipped: {type(exc).__name__}: {exc}", flush=True)
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    for method in sorted(spatial_learning["method"].dropna().unique()):
        sub = spatial_learning[spatial_learning["method"] == method]
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy, g in sub.groupby("strategy"):
            g = g.sort_values("spike_points")
            ax.plot(g["spike_points"], g["median_delta_nse"], marker="o", label=strategy)
        ax.axhline(0, color="0.3", linewidth=0.8, linestyle="--")
        ax.set(
            xlabel="local spike points supplied (target point held out)",
            ylabel="median ΔNSE/R² vs unspiked model6",
            title=f"Spatial local-data spiking sensitivity ({method})",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"spatial_spiking_delta_nse_{method}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, g in temporal_learning.groupby("method"):
        g = g.sort_values("training_dates")
        ax.plot(g["training_dates"], g["median_delta_nse"], marker="o", label=method)
    ax.axhline(0, color="0.3", linewidth=0.8, linestyle="--")
    ax.set(
        xlabel="early observations supplied at the same point",
        ylabel="median ΔNSE/R² on later dates",
        title="Temporal self-spiking sensitivity",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "temporal_self_spiking_delta_nse.png", dpi=160)
    plt.close(fig)


def write_summary_report(
    out_dir: Path,
    validation_dir: Path,
    stage1: dict,
    spatial: dict[str, pd.DataFrame],
    temporal: dict[str, pd.DataFrame],
    pooled_metrics: dict,
    metadata: dict,
) -> None:
    point_full = stage1["point_metrics"]
    corr = stage1["correlations"]
    spatial_learning = spatial["learning"]
    temporal_learning = temporal["learning"]

    best = stage1["best"]
    worst = stage1["worst"]
    top_corr_nse = (
        corr[corr["quality_metric"] == "nse"]
        .sort_values("abs_pearson_r", ascending=False)
        .head(10)
        [["model_input", "pearson_r", "spearman_r", "n_points"]]
        if not corr.empty
        else pd.DataFrame()
    )
    met_corr_nse = (
        corr[
            (corr["quality_metric"] == "nse")
            & corr["model_input"].isin(METEOROLOGY_INPUTS)
        ]
        .sort_values("abs_pearson_r", ascending=False)
        [["model_input", "pearson_r", "spearman_r", "n_points"]]
        if not corr.empty
        else pd.DataFrame()
    )
    met_corr_bias = (
        corr[
            (corr["quality_metric"].isin(["bias", "abs_bias"]))
            & corr["model_input"].isin(METEOROLOGY_INPUTS)
        ]
        .sort_values(["quality_metric", "abs_pearson_r"], ascending=[True, False])
        [["model_input", "quality_metric", "pearson_r", "spearman_r", "n_points"]]
        if not corr.empty
        else pd.DataFrame()
    )
    best_spatial = (
        spatial_learning.sort_values("median_delta_nse", ascending=False)
        .head(12)
        if not spatial_learning.empty
        else pd.DataFrame()
    )
    best_temporal = (
        temporal_learning.sort_values("median_delta_nse", ascending=False)
        .head(8)
        if not temporal_learning.empty
        else pd.DataFrame()
    )

    positive = int((point_full["nse"] > 0).sum())
    very_poor = int((point_full["nse"] <= -1).sum())
    report = f"""# Model6 dense-point validation and local spiking analysis

Output folder: `{out_dir}`

Source validation folder: `{validation_dir}`

Git branch: `{metadata.get("git_branch")}`

## Executive summary

The analysis is implemented in two stages.

1. **Dense unseen-site validation:** model6 was evaluated against the high-density
   point dataset without using those points for model fitting. The validation
   rasters and tables map where the model transfers well or poorly across the
   site. Bias diagnostics use the model's own inputs — terrain, SLGA soil,
   SMIPS lookbacks, antecedent weather and seasonality — rather than the
   auxiliary terrain columns in the field CSV.
2. **Local-data spiking sensitivity:** local observations were supplied in
   increasing amounts to residual-calibration experiments to quantify how much
   local information improves predictions at held-out locations or later dates.

## Stage 1 — unseen dense-point validation

Pooled model6 skill against the dense point dataset:

| metric | value |
|---|---:|
| NSE / R² | {fmt(pooled_metrics.get("nse"))} |
| Pearson r | {fmt(pooled_metrics.get("r"))} |
| RMSE | {fmt(pooled_metrics.get("rmse"))} |
| ubRMSE | {fmt(pooled_metrics.get("ubrmse"))} |
| bias | {fmt(pooled_metrics.get("bias"))} |
| n | {pooled_metrics.get("n")} |

Per-point summary:

- positive NSE/R² points: {positive}/{len(point_full)}
- very poor NSE/R² points (≤ -1): {very_poor}/{len(point_full)}
- median |bias|: {fmt(point_full["bias"].abs().median(), 2)} %

### Best points by NSE/R²

{markdown_table(best)}

### Worst points by NSE/R²

{markdown_table(worst)}

### Strongest model-input associations with point NSE/R²

{markdown_table(top_corr_nse)}

### Antecedent meteorology associations with point NSE/R²

{markdown_table(met_corr_nse)}

### Antecedent meteorology associations with bias and |bias|

{markdown_table(met_corr_bias)}

Interpretation: this is an exploratory bias screen. Strong associations indicate
where model6 may be systematically over- or under-performing in its own input
space, but they are not causal by themselves.

## Stage 2 — sensitivity to local training-data spiking

Two local spiking experiments were run:

1. **Spatial spiking:** for each target point, the target point was held out.
   Increasing numbers of other local points were supplied as calibration data.
   Points were selected by four strategies: nearest in space, most similar in
   model-input terrain/soil space, stratified coverage of model-input space, and
   random selection.
2. **Temporal self-spiking:** for each point, the first few observations at that
   same location were supplied as calibration data and later dates were held out.

The implemented spiking mechanism is residual calibration of the shipped model6
predictions, not full OzNet+local retraining, because the canonical OzNet
training table was not present in this checkout. The feature table produced here
can be appended to a rebuilt OzNet table for a full retraining experiment.

### Best spatial spiking settings by median ΔNSE/R²

{markdown_table(best_spatial[["strategy", "method", "spike_points", "median_delta_nse", "median_delta_rmse", "median_delta_abs_bias", "positive_nse_fits"]])}

For random selection, `positive_nse_fits` counts repeated target/replicate fits;
for deterministic strategies, it is equivalent to the number of target points.
The ridge residual corrector is useful for sensitivity testing but can be
unstable with very small temporal spike counts; the bias-only corrector is the
more conservative low-data benchmark.

### Best temporal self-spiking settings by median ΔNSE/R²

{markdown_table(best_temporal[["method", "training_dates", "median_delta_nse", "median_delta_rmse", "median_delta_abs_bias", "positive_nse_points"]])}

## Research novelty

High-novelty components:

- The dense point dataset supports sub-grid validation of a downscaled soil
  moisture product, rather than only sparse station validation.
- Mapping NSE/R², bias and RMSE at dense point locations exposes where a
  national gridded model succeeds or fails within a single high-density terrain
  mosaic.
- The local-data spiking curves quantify the marginal value of adding local
  measurements, including whether spatial proximity or model-input similarity is
  the better guide for calibration sampling.
- Temporal self-spiking estimates how many local visits are needed before a
  specific point becomes locally reliable.

Comparable-to-existing-study components:

- Use of held-out independent observations and standard soil-moisture metrics
  such as RMSE, ubRMSE, bias, correlation and NSE/R².
- Use of terrain, soil, coarse soil-moisture products and antecedent weather as
  predictors for statistical downscaling.
- Cross-validation concepts that hold out space or time to assess transfer.

## Full utilisation of the dense spatial point dataset

The dataset is used as:

1. an independent validation target;
2. a spatial bias map of point-wise model quality;
3. a model-input-space diagnostic for terrain/soil/antecedent bias;
4. a source of controlled local calibration spikes;
5. a basis for learning curves that estimate the minimum local sampling density
   needed to improve prediction at specific locations.

## Caveat

Model6 was trained on OzNet root-zone soil moisture, while the dense point CSV
appears to represent shallower measurements. Treat these results as an external
terrain-transfer and calibration-sensitivity diagnostic unless measurement depth
is reconciled.
"""
    (out_dir / "summary_report.md").write_text(report)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def write_metadata(out_dir: Path, metadata: dict) -> None:
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_existing_stage_outputs(paths: dict[str, Path]) -> tuple[dict, dict, dict]:
    """Load existing Stage 1/2 tables for a report-only refresh."""
    point_full = pd.read_csv(paths["stage1"] / "point_metrics_with_model_inputs.csv")
    corr = pd.read_csv(paths["stage1"] / "model_input_quality_correlations.csv")
    stage1 = {
        "point_metrics": point_full,
        "correlations": corr,
        "best": point_full.nlargest(10, "nse")[
            ["point", "nse", "r", "bias", "rmse", "lon", "lat", "quality_class"]
        ],
        "worst": point_full.nsmallest(10, "nse")[
            ["point", "nse", "r", "bias", "rmse", "lon", "lat", "quality_class"]
        ],
    }
    spatial = {
        "learning": pd.read_csv(paths["stage2"] / "spatial_spiking_learning_curve.csv")
    }
    temporal = {
        "learning": pd.read_csv(paths["stage2"] / "temporal_self_spiking_learning_curve.csv")
    }
    return stage1, spatial, temporal


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense-point validation and local spiking analysis.")
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--allow-non-emt", action="store_true")
    parser.add_argument("--radius-m", type=float, default=45.0)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="refresh summary_report.md from existing Stage 1/2 output tables",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_emt_branch(args.allow_non_emt)
    paths = ensure_dirs(args.output_dir)

    predictions, point_metrics = load_validation(args.validation_dir)
    pooled = read_json(args.validation_dir / "metrics_pooled.json")
    bbox = bbox_from_predictions(predictions)
    template_tif = sorted((args.validation_dir / "tifs").glob("soil_moisture_*.tif"))[0]

    metadata = {
        "git_branch": repo_branch(),
        "repo_root": str(REPO_ROOT),
        "validation_dir": str(args.validation_dir),
        "input_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "bbox_wsen_epsg4326": bbox,
        "template_tif": str(template_tif),
        "random_repeats": args.random_repeats,
        "seed": args.seed,
    }

    if args.report_only:
        stage1, spatial, temporal = load_existing_stage_outputs(paths)
        write_summary_report(args.output_dir, args.validation_dir, stage1, spatial, temporal, pooled, metadata)
        write_metadata(args.output_dir, metadata)
        print(f"wrote {args.output_dir / 'summary_report.md'}", flush=True)
        print(f"wrote {args.output_dir / 'metadata.json'}", flush=True)
        return 0

    feature_rows = sample_model6_inputs(
        predictions,
        bbox,
        cache_csv=paths["stage1"] / "point_date_model_inputs.csv",
        force=args.force_features,
    )
    stage1 = stage1_diagnostics(
        feature_rows,
        point_metrics,
        paths["stage1"],
        paths["rasters"],
        template_tif=template_tif,
        radius_m=args.radius_m,
    )

    print("running Stage 2 spatial local-data spiking ...", flush=True)
    spatial = stage2_spatial_spiking(
        feature_rows,
        paths["stage2"],
        repeats=args.random_repeats,
        seed=args.seed,
    )
    print("running Stage 2 temporal self-spiking ...", flush=True)
    temporal = stage2_temporal_self_spiking(feature_rows, paths["stage2"])
    make_stage2_figures(spatial["learning"], temporal["learning"], paths["stage2_figures"])

    write_summary_report(args.output_dir, args.validation_dir, stage1, spatial, temporal, pooled, metadata)
    write_metadata(args.output_dir, metadata)
    print(f"wrote {args.output_dir / 'summary_report.md'}", flush=True)
    print(f"wrote {args.output_dir / 'metadata.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
