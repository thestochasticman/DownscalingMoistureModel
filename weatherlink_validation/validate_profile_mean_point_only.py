"""Profile-mean WeatherLink validation without generating full daily TIFFs.

Preferred WeatherLink validation mode for Drill & Drop profiles:
individual depths are averaged to one daily profile mean, then model6 is
evaluated at the profile coordinate by sampling the same model inputs used by
the raster pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weatherlink_validation import download_weatherlink_soil_moisture as downloader  # noqa: E402
from weatherlink_validation.make_profile_mean_csv import profile_mean  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "weatherlink_validation" / "outputs"


def fmt(value: float, digits: int = 3) -> str:
    try:
        value = float(value)
    except Exception:
        return "NA"
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def safe_json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def bbox_from_rows(rows: pd.DataFrame, padding_deg: float = 0.002) -> tuple[float, float, float, float]:
    return (
        float(rows["lon"].min() - padding_deg),
        float(rows["lat"].min() - padding_deg),
        float(rows["lon"].max() + padding_deg),
        float(rows["lat"].max() + padding_deg),
    )


def generic_to_model_rows(generic: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(generic["Date"], errors="coerce")
    obs = pd.to_numeric(generic["Soil_moisture"], errors="coerce")
    lon = pd.to_numeric(generic["x_3577"], errors="coerce")
    lat = pd.to_numeric(generic["y_3577"], errors="coerce")
    ok = dates.notna() & obs.notna() & lon.notna() & lat.notna()
    out = pd.DataFrame(
        {
            "point": generic.loc[ok, "Point_number"].astype(str),
            "date": dates.loc[ok].dt.date.astype(str),
            "obs_sm_pct": obs.loc[ok].astype(float),
            "lon": lon.loc[ok].astype(float),
            "lat": lat.loc[ok].astype(float),
            "source_row": generic.index[ok].astype(int),
            "measurement_time": generic.loc[ok, "Time"].astype(str)
            if "Time" in generic.columns
            else "daily_profile_mean",
        }
    )
    out["pred_sm_pct"] = np.nan
    out["residual_obs_minus_pred"] = np.nan
    out["residual_pred_minus_obs"] = np.nan
    return out.sort_values(["date", "point"]).reset_index(drop=True)


def predict_point_model6(rows: pd.DataFrame, output_dir: Path, force_features: bool) -> pd.DataFrame:
    from emt.model6 import model as model6
    from emt.persist import load_model
    from soilmoisture_points_validation.dense_validation_and_spiking import sample_model6_inputs

    model = load_model("model6")
    if model is None:
        raise SystemExit("No trained model6 found at data/models/model6.joblib")

    feature_rows = sample_model6_inputs(
        rows,
        bbox_from_rows(rows),
        cache_csv=output_dir / "point_date_model_inputs.csv",
        force=force_features,
    )
    feature_rows["pred_sm_pct"] = model.predict(feature_rows[list(model6.FEATURES)])
    feature_rows["residual_pred_minus_obs"] = feature_rows["pred_sm_pct"] - feature_rows["obs_sm_pct"]
    feature_rows["residual_obs_minus_pred"] = feature_rows["obs_sm_pct"] - feature_rows["pred_sm_pct"]
    return feature_rows


def compute_metrics(y_true, y_pred) -> dict:
    from emt.evaluation import metrics

    return metrics(y_true, y_pred)


def window_metrics(pred: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    rows = []
    pred = pred.sort_values("date").copy()
    pred["date_dt"] = pd.to_datetime(pred["date"])
    end = pred["date_dt"].max()
    for days in windows:
        start = end - pd.Timedelta(days=days - 1)
        sub = pred[pred["date_dt"].between(start, end)]
        if len(sub) < 2:
            continue
        rows.append(
            {
                "window": f"last_{days}_days",
                "start_date": sub["date"].min(),
                "end_date": sub["date"].max(),
                **compute_metrics(sub["obs_sm_pct"], sub["pred_sm_pct"]),
            }
        )
    if len(pred) >= 2:
        rows.append(
            {
                "window": "all",
                "start_date": pred["date"].min(),
                "end_date": pred["date"].max(),
                **compute_metrics(pred["obs_sm_pct"], pred["pred_sm_pct"]),
            }
        )
    return pd.DataFrame(rows)


def cumulative_metrics(pred: pd.DataFrame, min_n: int = 14) -> pd.DataFrame:
    pred = pred.sort_values("date").reset_index(drop=True)
    rows = []
    for i in range(min_n, len(pred) + 1):
        sub = pred.iloc[:i]
        rows.append({"n_days": i, "end_date": sub["date"].iloc[-1], **compute_metrics(sub["obs_sm_pct"], sub["pred_sm_pct"])})
    return pd.DataFrame(rows)


def temporal_spiking(pred: pd.DataFrame, training_sizes: list[int]) -> pd.DataFrame:
    pred = pred.sort_values("date").reset_index(drop=True)
    rows = []
    for n_train in training_sizes:
        if len(pred) <= n_train + 1:
            continue
        train = pred.iloc[:n_train]
        test = pred.iloc[n_train:]
        baseline = compute_metrics(test["obs_sm_pct"], test["pred_sm_pct"])
        correction = float((train["obs_sm_pct"] - train["pred_sm_pct"]).mean())
        corrected = test["pred_sm_pct"] + correction
        m = compute_metrics(test["obs_sm_pct"], corrected)
        row = {
            "method": "bias_only",
            "training_dates": n_train,
            "train_start_date": train["date"].min(),
            "train_end_date": train["date"].max(),
            "test_start_date": test["date"].min(),
            "test_end_date": test["date"].max(),
            "test_observations": len(test),
            "correction_pct": correction,
            **m,
            **{f"baseline_{k}": v for k, v in baseline.items()},
        }
        row["delta_nse"] = row["nse"] - baseline["nse"]
        row["delta_rmse"] = row["rmse"] - baseline["rmse"]
        row["delta_abs_bias"] = abs(row["bias"]) - abs(baseline["bias"])
        rows.append(row)
    return pd.DataFrame(rows)


def make_figures(pred: pd.DataFrame, windows: pd.DataFrame, cumulative: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"figures skipped: {type(exc).__name__}: {exc}", flush=True)
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = pred.sort_values("date").copy()
    p["date_dt"] = pd.to_datetime(p["date"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(p["date_dt"], p["obs_sm_pct"], label="WeatherLink profile mean", linewidth=1.8)
    ax.plot(p["date_dt"], p["pred_sm_pct"], label="model6 point prediction", linewidth=1.5)
    ax.set(ylabel="Soil moisture (%)", title="Profile-mean WeatherLink vs model6")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "timeseries_obs_vs_model6.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(p["obs_sm_pct"], p["pred_sm_pct"], s=25, alpha=0.8)
    lo = min(p["obs_sm_pct"].min(), p["pred_sm_pct"].min())
    hi = max(p["obs_sm_pct"].max(), p["pred_sm_pct"].max())
    ax.plot([lo, hi], [lo, hi], color="0.3", linestyle="--", linewidth=1)
    ax.set(xlabel="Observed profile mean (%)", ylabel="model6 prediction (%)", title="Observed vs predicted")
    fig.tight_layout()
    fig.savefig(fig_dir / "scatter_obs_vs_model6.png", dpi=160)
    plt.close(fig)

    w = windows[windows["window"] != "all"].copy()
    if not w.empty:
        w["days"] = w["window"].str.extract(r"(\d+)").astype(int)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(w["days"], w["nse"], marker="o")
        ax.axhline(0, color="0.3", linewidth=0.8, linestyle="--")
        ax.set(xlabel="Trailing window length (days)", ylabel="NSE/R²", title="Does NSE improve with longer windows?")
        fig.tight_layout()
        fig.savefig(fig_dir / "nse_by_trailing_window.png", dpi=160)
        plt.close(fig)

    if not cumulative.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(cumulative["n_days"], cumulative["nse"], linewidth=1.5)
        ax.axhline(0, color="0.3", linewidth=0.8, linestyle="--")
        ax.set(xlabel="Days included from start", ylabel="Cumulative NSE/R²", title="Cumulative NSE as record length increases")
        fig.tight_layout()
        fig.savefig(fig_dir / "cumulative_nse.png", dpi=160)
        plt.close(fig)


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                vals.append("")
            elif isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.{digits}f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, lsid: str, profile_csv: Path, pred: pd.DataFrame, pooled: dict, windows: pd.DataFrame, spiking: pd.DataFrame) -> None:
    body = f"""# WeatherLink profile-mean point-only validation

Sensor lsid: `{lsid}`  
Profile-mean CSV: `{profile_csv}`  
Date range: {pred["date"].min()} to {pred["date"].max()}  
Observations: {len(pred)}

Only the profile-mean Drill & Drop value is used. Individual depths are not used
as validation points.

## Pooled model6 skill

| Metric | Value |
|---|---:|
| NSE/R² | {fmt(pooled.get("nse"))} |
| Pearson r | {fmt(pooled.get("r"))} |
| RMSE | {fmt(pooled.get("rmse"))} % |
| ubRMSE | {fmt(pooled.get("ubrmse"))} % |
| Bias | {fmt(pooled.get("bias"))} % |
| n | {pooled.get("n")} |

## Trailing-window comparison

{markdown_table(windows[["window", "start_date", "end_date", "n", "nse", "r", "rmse", "bias"]])}

## Temporal self-spiking

Spatial spiking is skipped because this is one profile. Temporal self-spiking
uses the first N profile-mean observations to estimate a simple local bias
correction, then evaluates later observations.

{markdown_table(spiking[["training_dates", "test_observations", "correction_pct", "nse", "baseline_nse", "delta_nse", "rmse", "baseline_rmse", "delta_rmse", "bias"]])}

## Interpretation note

If NSE improves in longer windows, the short-window negative NSE was likely
being driven by low observed variance over the 20-day winter subset. If NSE
remains negative despite stronger correlation or low RMSE, model6 is capturing
some level/direction information but not the full local temporal dynamics.

Figures:

- `figures/timeseries_obs_vs_model6.png`
- `figures/scatter_obs_vs_model6.png`
- `figures/nse_by_trailing_window.png`
- `figures/cumulative_nse.png`
"""
    meta_cols = [c for c in ["node_name", "lsid", "depth_min_cm", "depth_max_cm", "n_depths"] if c in pred.columns]
    if meta_cols:
        body += "\n## Profile metadata\n\n" + markdown_table(pred[meta_cols].drop_duplicates()) + "\n"
    (output_dir / "summary_report.md").write_text(body)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile-mean point-only WeatherLink model6 validation.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--station-id", default="149046")
    parser.add_argument("--lsid", default="591644")
    parser.add_argument("--timezone", default="Australia/Sydney")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"profile_mean_{args.lsid}_{args.start_date}_{args.end_date}"
    download_dir = output_dir / "download"
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    depth_csv = download_dir / f"weatherlink_{args.lsid}_depth_generic_model6_validation.csv"
    profile_csv = download_dir / f"weatherlink_{args.lsid}_profile_mean_generic_model6_validation.csv"

    if not args.skip_download:
        dl_args = argparse.Namespace(
            start_date=downloader.parse_day(args.start_date),
            end_date=downloader.parse_day(args.end_date),
            station_id=[args.station_id],
            lsid=[args.lsid],
            timezone=args.timezone,
            api_key=None,
            api_secret=None,
            env_file=downloader.DEFAULT_ENV_FILE,
            demo=False,
            output_dir=download_dir,
            basename=f"weatherlink_{args.lsid}",
            sensor_locations_csv=None,
            field_regex=downloader.DEFAULT_FIELD_REGEX,
            value_mode="auto",
            daily_agg="median",
            keep_unconverted=False,
            list_only=False,
        )
        paths = downloader.build_dataset(dl_args)
        Path(paths["generic_csv"]).replace(depth_csv)

    depth = pd.read_csv(depth_csv)
    profile = profile_mean(depth_csv)
    profile.to_csv(profile_csv, index=False)

    rows = generic_to_model_rows(profile)
    if rows.empty:
        raise SystemExit("No usable profile-mean rows after filtering/downloading.")

    pred = predict_point_model6(rows, output_dir, args.force_features)
    meta = profile.drop_duplicates("Point_number")
    keep_meta = [
        c for c in profile.columns
        if c not in pred.columns and c not in {"Date", "Time", "Point_number", "Soil_moisture", "x_3577", "y_3577"}
    ]
    if keep_meta:
        pred = pred.merge(meta[["Point_number", *keep_meta]], left_on="point", right_on="Point_number", how="left")
        pred = pred.drop(columns=["Point_number"], errors="ignore")

    pooled = compute_metrics(pred["obs_sm_pct"], pred["pred_sm_pct"])
    windows = window_metrics(pred, [20, 45, 60, 90, 120, 180, 270, 365])
    cumulative = cumulative_metrics(pred)
    spiking = temporal_spiking(pred, [1, 2, 3, 5, 10, 20, 30, 60, 90, 120])

    pred.to_csv(output_dir / "profile_mean_predictions.csv", index=False)
    windows.to_csv(output_dir / "window_metrics.csv", index=False)
    cumulative.to_csv(output_dir / "cumulative_metrics.csv", index=False)
    spiking.to_csv(output_dir / "temporal_self_spiking.csv", index=False)
    with (output_dir / "metrics_pooled.json").open("w") as f:
        json.dump({k: safe_json(v) for k, v in pooled.items()}, f, indent=2)
    make_figures(pred, windows, cumulative, output_dir)
    write_report(output_dir, args.lsid, profile_csv, pred, pooled, windows, spiking)

    print("\nProfile-mean point-only validation complete")
    print("-------------------------------------------")
    print(f"Sensor lsid: {args.lsid}")
    print(f"Date range: {pred['date'].min()} to {pred['date'].max()}")
    print(f"Observations: {len(pred)}")
    print(f"Pooled NSE / r: {fmt(pooled['nse'])} / {fmt(pooled['r'])}")
    print(f"RMSE / bias: {fmt(pooled['rmse'])} / {fmt(pooled['bias'])} %")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
