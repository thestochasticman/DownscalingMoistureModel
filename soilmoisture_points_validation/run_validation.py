"""Validate model6 GeoTIFF predictions against soil-moisture point observations.

The workflow is intentionally folder-local and reproducible:

    python soilmoisture_points_validation/run_validation.py

It reads the field CSV, computes one site bbox, uses ``emt.predict`` to generate
one GeoTIFF per sampling date, samples those rasters at the observed points, and
writes the same metrics used by the handout's evaluation module.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INPUT = Path(
    "/Volumes/Dmitry_work/borevitz_projects/Data/soilmoisture_points_coordinates.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"

DATE_COL = "Date"
POINT_COL = "Point_number"
OBS_COL = "Soil_moisture"
TIME_COL = "Time"
LON_COL = "x_3577"
LAT_COL = "y_3577"


def _repo_branch() -> str | None:
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
    branch = _repo_branch()
    if allow_non_emt or branch in (None, "EMT"):
        return
    raise SystemExit(
        f"This validation workflow is intended for the EMT branch; current branch "
        f"is {branch!r}. Switch with `git switch EMT`, or pass --allow-non-emt."
    )


def _check_columns(df: pd.DataFrame) -> None:
    required = [DATE_COL, POINT_COL, OBS_COL, LON_COL, LAT_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"input CSV is missing required columns: {missing}")


def load_observations(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return valid point observations and excluded source rows."""
    df = pd.read_csv(path)
    _check_columns(df)

    dates = pd.to_datetime(df[DATE_COL], errors="coerce")
    obs = pd.to_numeric(df[OBS_COL], errors="coerce")
    lon = pd.to_numeric(df[LON_COL], errors="coerce")
    lat = pd.to_numeric(df[LAT_COL], errors="coerce")

    valid = (
        dates.notna()
        & obs.notna()
        & lon.between(-180, 180)
        & lat.between(-90, 90)
    )

    exclude_reason = pd.Series(
        np.select(
            [
                dates.isna(),
                obs.isna(),
                ~(lon.between(-180, 180) & lat.between(-90, 90)),
            ],
            [
                "missing_or_invalid_date",
                "missing_or_invalid_soil_moisture",
                "missing_or_invalid_coordinates",
            ],
            default="unknown",
        ),
        index=df.index,
    )
    excluded = df.loc[~valid].copy()
    excluded["exclude_reason"] = exclude_reason.loc[~valid]

    out = pd.DataFrame(
        {
            "point": df.loc[valid, POINT_COL].astype(str).str.strip(),
            "date": dates.loc[valid].dt.date.astype(str),
            "obs_sm_pct": obs.loc[valid].astype(float),
            "lon": lon.loc[valid].astype(float),
            "lat": lat.loc[valid].astype(float),
            "source_row": df.index[valid].astype(int),
        }
    )
    if TIME_COL in df.columns:
        out["measurement_time"] = df.loc[valid, TIME_COL].astype(str)
    if "Water_mm" in df.columns:
        out["water_mm"] = pd.to_numeric(df.loc[valid, "Water_mm"], errors="coerce")

    sort_cols = ["date", "point"]
    if "measurement_time" in out.columns:
        sort_cols.append("measurement_time")
    return out.sort_values(sort_cols).reset_index(drop=True), excluded


def bbox_from_points(rows: pd.DataFrame, padding_deg: float) -> tuple[float, float, float, float]:
    west = float(rows["lon"].min() - padding_deg)
    south = float(rows["lat"].min() - padding_deg)
    east = float(rows["lon"].max() + padding_deg)
    north = float(rows["lat"].max() + padding_deg)
    if not (west < east and south < north):
        raise ValueError(f"invalid bbox derived from point coordinates: {(west, south, east, north)}")
    return west, south, east, north


def tif_path(tif_dir: Path, day: str) -> Path:
    return tif_dir / f"soil_moisture_{day}.tif"


def load_model_once(model_name: str):
    try:
        from emt.persist import load_model
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import EMT model persistence. Run this from the "
            "DownscalingMoistureModel repo root with the package installed/editable."
        ) from exc

    try:
        model = load_model(model_name)
    except ModuleNotFoundError as exc:
        if exc.name == "_loss":
            raise SystemExit(
                "Could not load data/models/model6.joblib because scikit-learn's "
                "pickle internals do not match this environment. The shipped model "
                "was saved with scikit-learn 1.8.0; run:\n\n"
                "  conda activate paddockts\n"
                '  conda install -c conda-forge "scikit-learn=1.8.0"\n'
            ) from exc
        raise
    if model is None:
        raise SystemExit(f"No trained model found at data/models/{model_name}.joblib")
    return model


def generate_tifs(
    dates: Iterable[str],
    bbox: tuple[float, float, float, float],
    tif_dir: Path,
    model_name: str,
    overwrite: bool,
) -> list[Path]:
    """Generate missing date rasters with emt.predict and return all paths."""
    try:
        from emt.predict import predict
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import `emt.predict`. Make sure this repo is on the latest "
            "EMT branch and run from the repository root."
        ) from exc

    tif_dir.mkdir(parents=True, exist_ok=True)
    days = list(dates)
    needed = [d for d in days if overwrite or not tif_path(tif_dir, d).exists()]
    model = load_model_once(model_name) if needed else None

    paths: list[Path] = []
    for i, day in enumerate(days, 1):
        out = tif_path(tif_dir, day)
        paths.append(out)
        if out.exists() and not overwrite:
            print(f"[{i}/{len(days)}] using existing {out}")
            continue

        print(f"[{i}/{len(days)}] generating {out.name} for bbox={bbox}")
        ds = predict(
            bbox,
            date.fromisoformat(day),
            model=model,
            model_name=model_name,
            verbose=True,
            save=False,
            plot=False,
        )
        ds["sm_pred"].rio.to_raster(out)
        print(f"    wrote {out}")
    return paths


def _sample_one_tif(tif: Path, points: pd.DataFrame) -> np.ndarray:
    import rasterio
    from pyproj import Transformer

    with rasterio.open(tif) as src:
        xs = points["lon"].to_numpy(dtype=float)
        ys = points["lat"].to_numpy(dtype=float)
        if src.crs is not None and src.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = transformer.transform(xs, ys)

        values: list[float] = []
        for sample in src.sample(zip(xs, ys), masked=True):
            value = sample[0]
            if np.ma.is_masked(value):
                values.append(np.nan)
                continue
            value = float(value)
            if src.nodata is not None and math.isclose(value, float(src.nodata)):
                values.append(np.nan)
            else:
                values.append(value)
        return np.asarray(values, dtype=float)


def sample_predictions(rows: pd.DataFrame, tif_dir: Path) -> pd.DataFrame:
    out = rows.copy()
    out["pred_sm_pct"] = np.nan
    out["tif"] = ""

    for day, idx in out.groupby("date", sort=True).groups.items():
        tif = tif_path(tif_dir, str(day))
        if not tif.exists():
            raise FileNotFoundError(
                f"missing TIFF for {day}: {tif}. Generate it first or omit --sample-only."
            )
        points = out.loc[idx]
        out.loc[idx, "pred_sm_pct"] = _sample_one_tif(tif, points)
        out.loc[idx, "tif"] = str(tif)
    out["residual_pred_minus_obs"] = out["pred_sm_pct"] - out["obs_sm_pct"]
    return out


def grouped_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    from emt.evaluation import metrics

    rows = []
    for key, group in df.groupby(group_col, dropna=False, sort=True):
        item = {group_col: key}
        item.update(metrics(group["obs_sm_pct"], group["pred_sm_pct"]))
        rows.append(item)
    return pd.DataFrame(rows)


def compute_metrics(predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    from emt.evaluation import metrics

    valid = predictions.dropna(subset=["obs_sm_pct", "pred_sm_pct"]).copy()
    if valid.empty:
        raise ValueError("no finite prediction/observation pairs were available for metrics")

    pooled = metrics(valid["obs_sm_pct"], valid["pred_sm_pct"])
    per_point = grouped_metrics(valid, "point")
    per_date = grouped_metrics(valid, "date")
    return pooled, per_point, per_date


def _round_for_json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def fmt(value: float, digits: int = 3) -> str:
    try:
        value = float(value)
    except Exception:
        return "NA"
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def print_summary(pooled: dict, per_point: pd.DataFrame, predictions: pd.DataFrame) -> None:
    finite = predictions["pred_sm_pct"].notna().sum()
    total = len(predictions)
    positive_nse = int((per_point["nse"] > 0).sum()) if "nse" in per_point else 0
    median_abs_bias = float(per_point["bias"].abs().median()) if len(per_point) else np.nan

    print("\nValidation summary")
    print("------------------")
    print(f"Sampled predictions: {finite}/{total}")
    print(f"Pooled NSE / r: {fmt(pooled['nse'])} / {fmt(pooled['r'])}")
    print(
        "RMSE / ubRMSE / bias: "
        f"{fmt(pooled['rmse'])} / {fmt(pooled['ubrmse'])} / {fmt(pooled['bias'])} %"
    )
    print(
        f"Per-point NSE > 0: {positive_nse}/{len(per_point)}; "
        f"median |bias| {fmt(median_abs_bias, 2)} %"
    )


def write_report(
    path: Path,
    input_csv: Path,
    bbox: tuple[float, float, float, float],
    dates: list[str],
    source_rows: int,
    used_rows: int,
    excluded_rows: int,
    pooled: dict,
    per_point: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    positive_nse = int((per_point["nse"] > 0).sum()) if "nse" in per_point else 0
    median_abs_bias = float(per_point["bias"].abs().median()) if len(per_point) else np.nan
    sampled = int(predictions["pred_sm_pct"].notna().sum())

    body = f"""# Soil-moisture point validation report

Input CSV: `{input_csv}`

Generated/sampled dates: {", ".join(dates)}

BBox used: `{bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}` (`W S E N`, EPSG:4326)

Rows:

- source rows: {source_rows}
- rows with usable date, observation and coordinates: {used_rows}
- excluded rows: {excluded_rows}
- sampled prediction/observation pairs: {sampled}

## Handout-style summary

| Skill | model6 on soil-moisture point CSV |
|---|---:|
| Pooled NSE / r | {fmt(pooled["nse"])} / {fmt(pooled["r"])} |
| RMSE / ubRMSE / bias | {fmt(pooled["rmse"])} / {fmt(pooled["ubrmse"])} / {fmt(pooled["bias"])} % |
| Median per-point \\|bias\\| | {fmt(median_abs_bias, 2)} % |
| Per-point NSE > 0 | {positive_nse}/{len(per_point)} |

## Caveat

The handout model predicts OzNet-style root-zone soil moisture, while this CSV
appears to contain shallower point measurements. These scores are therefore best
read as an external terrain-transfer diagnostic rather than a strict root-zone
validation unless the field measurement depth is reconciled.
"""
    path.write_text(body)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model6 TIFFs for the point CSV dates and validate them at the points."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="model6")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("W", "S", "E", "N"),
        help="optional lon/lat bbox override; defaults to point extent plus padding",
    )
    parser.add_argument(
        "--padding-deg",
        type=float,
        default=0.002,
        help="bbox padding in degrees when --bbox is not supplied",
    )
    parser.add_argument(
        "--overwrite-tifs",
        action="store_true",
        help="regenerate TIFFs even if outputs already exist",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="do not generate TIFFs; sample existing outputs/tifs/*.tif only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse the CSV, print date/bbox summary, then stop before imports/downloads",
    )
    parser.add_argument(
        "--allow-non-emt",
        action="store_true",
        help="skip the safety check that the current Git branch is EMT",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_emt_branch(args.allow_non_emt)

    rows, excluded = load_observations(args.input_csv)
    if rows.empty:
        raise SystemExit("No usable rows found in the input CSV.")

    bbox = tuple(args.bbox) if args.bbox else bbox_from_points(rows, args.padding_deg)
    dates = sorted(rows["date"].unique())

    print(f"Input CSV: {args.input_csv}")
    print(f"Usable rows: {len(rows)}")
    print(f"Excluded rows: {len(excluded)}")
    print(f"Unique points: {rows['point'].nunique()}")
    print(f"Unique dates: {len(dates)} ({', '.join(dates)})")
    print(f"BBox W S E N: {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

    if args.dry_run:
        print("\nDry run only; no TIFFs generated and no metrics computed.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tif_dir = args.output_dir / "tifs"
    excluded.to_csv(args.output_dir / "excluded_rows.csv", index=False)

    if not args.sample_only:
        generate_tifs(dates, bbox, tif_dir, args.model, args.overwrite_tifs)

    predictions = sample_predictions(rows, tif_dir)
    pooled, per_point, per_date = compute_metrics(predictions)

    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    per_point.to_csv(args.output_dir / "metrics_per_point.csv", index=False)
    per_date.to_csv(args.output_dir / "metrics_per_date.csv", index=False)
    with (args.output_dir / "metrics_pooled.json").open("w") as f:
        json.dump({k: _round_for_json(v) for k, v in pooled.items()}, f, indent=2)

    write_report(
        args.output_dir / "report.md",
        args.input_csv,
        bbox,
        dates,
        source_rows=len(rows) + len(excluded),
        used_rows=len(rows),
        excluded_rows=len(excluded),
        pooled=pooled,
        per_point=per_point,
        predictions=predictions,
    )
    print_summary(pooled, per_point, predictions)
    print(f"\nWrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
