"""Aggregate WeatherLink Drill & Drop depth rows to profile-mean rows.

Model6 predicts an OzNet-style root-zone soil-moisture percentage. WeatherLink
Drill & Drop data arrive as one value per depth. This helper turns the generic
depth-level CSV into a second generic CSV with one profile-average observation
per lsid/date, which is generally the fairer validation target for model6.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def profile_mean(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = ["Date", "Point_number", "Soil_moisture", "x_3577", "y_3577", "lsid", "depth_cm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"input CSV is missing required columns: {missing}")

    df["Soil_moisture"] = pd.to_numeric(df["Soil_moisture"], errors="coerce")
    df["depth_cm"] = pd.to_numeric(df["depth_cm"], errors="coerce")
    df = df.dropna(subset=["Date", "Soil_moisture", "x_3577", "y_3577", "lsid"])
    group_cols = ["Date", "lsid"]
    optional_first = [
        "station_id",
        "station_name",
        "node_name",
        "product_name",
        "conversion_mode",
        "raw_units",
    ]

    agg = {
        "Soil_moisture": "mean",
        "x_3577": "mean",
        "y_3577": "mean",
        "depth_cm": ["min", "max", "count"],
    }
    for col in optional_first:
        if col in df.columns:
            agg[col] = "first"

    out = df.groupby(group_cols, as_index=False, sort=True).agg(agg)
    out.columns = [
        "_".join(str(x) for x in col if str(x))
        if isinstance(col, tuple)
        else str(col)
        for col in out.columns
    ]
    out = out.rename(
        columns={
            "Date_": "Date",
            "lsid_": "lsid",
            "Soil_moisture_mean": "Soil_moisture",
            "x_3577_mean": "x_3577",
            "y_3577_mean": "y_3577",
            "depth_cm_min": "depth_min_cm",
            "depth_cm_max": "depth_max_cm",
            "depth_cm_count": "n_depths",
        }
    )
    for col in optional_first:
        if f"{col}_first" in out.columns:
            out = out.rename(columns={f"{col}_first": col})
    out["Point_number"] = out["lsid"].map(lambda x: f"wl_{x}_profile_mean")
    out["Time"] = "daily_profile_mean"
    ordered = [
        "Date",
        "Time",
        "Point_number",
        "Soil_moisture",
        "x_3577",
        "y_3577",
        "lsid",
        "node_name",
        "station_id",
        "station_name",
        "product_name",
        "depth_min_cm",
        "depth_max_cm",
        "n_depths",
        "conversion_mode",
        "raw_units",
    ]
    return out[[c for c in ordered if c in out.columns]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a profile-mean generic WeatherLink validation CSV.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = profile_mean(args.input_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"wrote {args.output_csv}")
    print(f"rows: {len(out)}; profiles: {out['Point_number'].nunique()}; dates: {out['Date'].nunique()}")
    print(f"depth counts: {sorted(out['n_depths'].dropna().astype(int).unique())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
