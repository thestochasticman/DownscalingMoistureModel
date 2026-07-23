"""Rasterise per-point validation quality onto the model prediction grid.

Outputs:

* ``point_quality_nse_r2.tif`` -- one-band GeoTIFF of per-point NSE/R².
* ``point_quality_metrics.tif`` -- multiband GeoTIFF with NSE/R², r, bias, RMSE,
  ubRMSE, sample count and NSE rank.
* ``point_quality_with_terrain.csv`` -- per-point metrics joined to coordinates
  and source terrain attributes for bias diagnostics.
* ``point_quality_report.md`` -- best/worst points and strongest terrain
  correlations with NSE.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

DEFAULT_VALIDATION_DIR = Path(
    "/Volumes/Dmitry_work/borevitz_projects/Data/soilmoisture_points_validation"
)
DEFAULT_INPUT = Path(
    "/Volumes/Dmitry_work/borevitz_projects/Data/soilmoisture_points_coordinates.csv"
)
NODATA = -9999.0
METRIC_BANDS = ["nse", "r2", "r", "bias", "rmse", "ubrmse", "n", "rank_nse"]


def first_template_tif(validation_dir: Path) -> Path:
    tifs = sorted((validation_dir / "tifs").glob("soil_moisture_*.tif"))
    if not tifs:
        raise FileNotFoundError(f"no prediction TIFFs found under {validation_dir / 'tifs'}")
    return tifs[0]


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


def load_point_quality(validation_dir: Path, input_csv: Path) -> pd.DataFrame:
    metrics = pd.read_csv(validation_dir / "metrics_per_point.csv")
    predictions = pd.read_csv(validation_dir / "predictions.csv")

    coords = (
        predictions.groupby("point", as_index=False)
        .agg(
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            observed_mean=("obs_sm_pct", "mean"),
            predicted_mean=("pred_sm_pct", "mean"),
            residual_mean=("residual_pred_minus_obs", "mean"),
            n_predictions=("pred_sm_pct", "count"),
        )
    )
    out = metrics.merge(coords, on="point", how="left")
    out["rank_nse"] = out["nse"].rank(ascending=False, method="min").astype("Int64")
    out["quality_class"] = out["nse"].map(quality_class)

    source = pd.read_csv(input_csv)
    if "Point_number" in source.columns:
        source["point"] = source["Point_number"].astype(str).str.strip()
        source_numeric = source.select_dtypes(include=[np.number]).copy()
        terrain_cols = [
            c
            for c in source_numeric.columns
            if c
            not in {
                "Soil_moisture",
                "Water_mm",
                "x_3577",
                "y_3577",
            }
        ]
        if terrain_cols:
            terrain = source.groupby("point")[terrain_cols].mean(numeric_only=True).reset_index()
            out = out.merge(terrain, on="point", how="left")

    return out.sort_values("rank_nse").reset_index(drop=True)


def rasterise_metrics(
    point_quality: pd.DataFrame,
    template_tif: Path,
    radius_m: float,
) -> tuple[dict[str, np.ndarray], dict]:
    with rasterio.open(template_tif) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    if crs is None:
        raise ValueError(f"template TIFF has no CRS: {template_tif}")

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = transformer.transform(
        point_quality["lon"].to_numpy(dtype=float),
        point_quality["lat"].to_numpy(dtype=float),
    )

    arrays = {name: np.full((height, width), np.nan, dtype="float32") for name in METRIC_BANDS}
    nearest_dist = np.full((height, width), np.inf, dtype="float32")
    pixel_size = max(abs(transform.a), abs(transform.e))
    radius_px = max(0, int(math.ceil(radius_m / pixel_size)))

    for i, row in point_quality.reset_index(drop=True).iterrows():
        x = float(xs[i])
        y = float(ys[i])
        col_f, row_f = ~transform * (x, y)
        row0 = int(math.floor(row_f))
        col0 = int(math.floor(col_f))

        r_min = max(0, row0 - radius_px)
        r_max = min(height - 1, row0 + radius_px)
        c_min = max(0, col0 - radius_px)
        c_max = min(width - 1, col0 + radius_px)

        for rr in range(r_min, r_max + 1):
            for cc in range(c_min, c_max + 1):
                cx, cy = rasterio.transform.xy(transform, rr, cc, offset="center")
                dist = math.hypot(float(cx) - x, float(cy) - y)
                if dist > radius_m or dist >= float(nearest_dist[rr, cc]):
                    continue
                nearest_dist[rr, cc] = dist
                for metric in METRIC_BANDS:
                    value = row.get(metric, np.nan)
                    arrays[metric][rr, cc] = np.nan if pd.isna(value) else float(value)

    return arrays, profile


def write_tif(path: Path, arrays: list[np.ndarray], profile: dict, descriptions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=len(arrays),
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        for band_index, (array, description) in enumerate(zip(arrays, descriptions), start=1):
            dst.write(np.where(np.isfinite(array), array, NODATA).astype("float32"), band_index)
            dst.set_band_description(band_index, description)


def correlations(point_quality: pd.DataFrame) -> pd.DataFrame:
    skip = {
        "rmse",
        "ubrmse",
        "bias",
        "r",
        "nse",
        "r2",
        "n",
        "rank_nse",
        "lon",
        "lat",
        "observed_mean",
        "predicted_mean",
        "residual_mean",
        "n_predictions",
    }
    rows = []
    for col in point_quality.select_dtypes(include=[np.number]).columns:
        if col in skip:
            continue
        sub = point_quality[["nse", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 5 or sub[col].std() == 0 or sub["nse"].std() == 0:
            continue
        rows.append(
            {
                "terrain_variable": col,
                "pearson_r_with_nse": float(sub["nse"].corr(sub[col])),
                "n": int(len(sub)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["terrain_variable", "pearson_r_with_nse", "n"])
    out = pd.DataFrame(rows)
    out["abs_r"] = out["pearson_r_with_nse"].abs()
    return out.sort_values("abs_r", ascending=False).drop(columns="abs_r")


def write_report(path: Path, point_quality: pd.DataFrame, corrs: pd.DataFrame, radius_m: float) -> None:
    best = point_quality.nlargest(10, "nse")[
        ["point", "nse", "r", "bias", "rmse", "n", "lon", "lat", "quality_class"]
    ]
    worst = point_quality.nsmallest(10, "nse")[
        ["point", "nse", "r", "bias", "rmse", "n", "lon", "lat", "quality_class"]
    ]
    corr_table = corrs.head(12)

    path.write_text(
        "# Point prediction-quality raster report\n\n"
        f"Rasterised point metric radius: {radius_m:.1f} m. If point buffers overlap, "
        "the nearest point wins for each pixel.\n\n"
        "Primary GeoTIFF: `point_quality_nse_r2.tif`.\n\n"
        "Multiband GeoTIFF: `point_quality_metrics.tif` with bands: "
        + ", ".join(METRIC_BANDS)
        + ".\n\n"
        "## Best points by NSE/R²\n\n"
        + markdown_table(best)
        + "\n\n## Worst points by NSE/R²\n\n"
        + markdown_table(worst)
        + "\n\n## Strongest source-terrain correlations with NSE/R²\n\n"
        + (
            markdown_table(corr_table)
            if len(corr_table)
            else "No numeric terrain correlations available."
        )
        + "\n"
    )


def markdown_table(df: pd.DataFrame) -> str:
    """Small markdown table writer that avoids pandas' optional tabulate dependency."""
    if df.empty:
        return "_No rows._"

    def cell(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        return str(value)

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create point-quality GeoTIFFs from validation metrics.")
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--template-tif", type=Path, default=None)
    parser.add_argument(
        "--radius-m",
        type=float,
        default=45.0,
        help="radius burned around each point; use ~0 for exact single-pixel points",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template_tif = args.template_tif or first_template_tif(args.validation_dir)
    point_quality = load_point_quality(args.validation_dir, args.input_csv)
    arrays, profile = rasterise_metrics(point_quality, template_tif, args.radius_m)

    primary = args.validation_dir / "point_quality_nse_r2.tif"
    multiband = args.validation_dir / "point_quality_metrics.tif"
    csv_out = args.validation_dir / "point_quality_with_terrain.csv"
    corrs_out = args.validation_dir / "point_quality_terrain_correlations.csv"
    report = args.validation_dir / "point_quality_report.md"

    write_tif(primary, [arrays["nse"]], profile, ["nse_r2"])
    write_tif(multiband, [arrays[name] for name in METRIC_BANDS], profile, METRIC_BANDS)
    point_quality.to_csv(csv_out, index=False)
    corrs = correlations(point_quality)
    corrs.to_csv(corrs_out, index=False)
    write_report(report, point_quality, corrs, args.radius_m)

    print(f"wrote {primary}")
    print(f"wrote {multiband}")
    print(f"wrote {csv_out}")
    print(f"wrote {corrs_out}")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
