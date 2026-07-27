"""End-to-end WeatherLink validation and local spiking wrapper.

This script:

1. downloads WeatherLink soil-moisture records and writes a generic point/date
   CSV;
2. runs the existing model6 TIFF generation + point validation workflow; and
3. runs the existing two-stage terrain-bias/spiking workflow when enough points
   and dates are available.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weatherlink_validation import download_weatherlink_soil_moisture as downloader  # noqa: E402
from weatherlink_validation.weatherlink_client import WeatherLinkError  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "weatherlink_validation" / "outputs"


def repo_branch() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def require_emt_branch(allow_non_emt: bool = False) -> None:
    branch = repo_branch()
    if allow_non_emt or branch in (None, "EMT"):
        return
    raise SystemExit(
        f"This workflow is intended for the EMT branch; current branch is {branch!r}. "
        "Switch with `git switch EMT`, or pass --allow-non-emt."
    )


def validate_generic_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Generic validation CSV does not exist: {path}")
    df = pd.read_csv(path)
    required = ["Date", "Point_number", "Soil_moisture", "x_3577", "y_3577"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Generic validation CSV is missing required columns: {missing}")
    finite = pd.to_numeric(df["Soil_moisture"], errors="coerce").notna().sum()
    if finite == 0:
        raise SystemExit(
            "The WeatherLink CSV has no finite Soil_moisture values in percent. "
            "For Davis centibar sensors, supply calibration parameters and rerun "
            "with --value-mode centibar_vg, or use --value-mode relative_wetness "
            "only for a relative diagnostic."
        )
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download WeatherLink soil data, validate model6, and run dense-style spiking analysis."
    )
    # Reuse downloader's parser for credential/download options by accepting the
    # same names here.
    parser.add_argument("--start-date", required=False)
    parser.add_argument("--end-date", required=False)
    parser.add_argument("--station-id", action="append")
    parser.add_argument("--lsid", action="append")
    parser.add_argument("--timezone")
    parser.add_argument("--api-key")
    parser.add_argument("--api-secret")
    parser.add_argument("--env-file", type=Path, default=downloader.DEFAULT_ENV_FILE)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--sensor-locations-csv", type=Path)
    parser.add_argument("--field-regex", default=downloader.DEFAULT_FIELD_REGEX)
    parser.add_argument(
        "--value-mode",
        choices=["auto", "percent", "centibar_vg", "relative_wetness", "centibar_as_percent"],
        default="auto",
    )
    parser.add_argument(
        "--daily-agg",
        choices=["median", "mean", "last", "min", "max", "none"],
        default="median",
    )
    parser.add_argument("--keep-unconverted", action="store_true")
    parser.add_argument("--list-only", action="store_true")

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--basename", default="weatherlink_soil_moisture")
    parser.add_argument("--generic-csv", type=Path, help="Use an existing generic CSV instead of downloading.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-model-validation", action="store_true")
    parser.add_argument("--skip-spiking", action="store_true")
    parser.add_argument("--overwrite-tifs", action="store_true")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--allow-non-emt", action="store_true")
    args = parser.parse_args(argv)
    if not args.skip_download and not args.generic_csv and not args.list_only:
        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date are required unless --skip-download, --generic-csv or --list-only is used")
    return args


def run_downloader(args: argparse.Namespace) -> Path:
    if args.generic_csv:
        return args.generic_csv
    if args.skip_download:
        return args.output_dir / f"{args.basename}_generic_model6_validation.csv"

    dl_args = argparse.Namespace(
        start_date=downloader.parse_day(args.start_date) if args.start_date else None,
        end_date=downloader.parse_day(args.end_date) if args.end_date else None,
        station_id=args.station_id,
        lsid=args.lsid,
        timezone=args.timezone,
        api_key=args.api_key,
        api_secret=args.api_secret,
        env_file=args.env_file,
        demo=args.demo,
        output_dir=args.output_dir / "download",
        basename=args.basename,
        sensor_locations_csv=args.sensor_locations_csv,
        field_regex=args.field_regex,
        value_mode=args.value_mode,
        daily_agg=args.daily_agg,
        keep_unconverted=args.keep_unconverted,
        list_only=args.list_only,
    )
    paths = downloader.build_dataset(dl_args)
    if args.list_only:
        raise SystemExit(0)
    return paths["generic_csv"]


def run_model_validation(args: argparse.Namespace, generic_csv: Path) -> Path:
    from soilmoisture_points_validation import run_validation

    validation_dir = args.output_dir / "model6_point_validation"
    argv = [
        "--input-csv",
        str(generic_csv),
        "--output-dir",
        str(validation_dir),
    ]
    if args.overwrite_tifs:
        argv.append("--overwrite-tifs")
    if args.sample_only:
        argv.append("--sample-only")
    if args.allow_non_emt:
        argv.append("--allow-non-emt")
    rc = run_validation.main(argv)
    if rc != 0:
        raise SystemExit(rc)
    return validation_dir


def run_spiking(args: argparse.Namespace, generic_csv: Path, validation_dir: Path, df: pd.DataFrame) -> Path | None:
    n_points = df["Point_number"].astype(str).nunique()
    n_dates = pd.to_datetime(df["Date"], errors="coerce").dt.date.nunique()
    if n_points < 2 or n_dates < 3:
        print(
            f"\nSkipping spiking: found {n_points} point(s) and {n_dates} date(s); "
            "spatial/temporal spiking needs at least 2 points and 3 dates.",
            flush=True,
        )
        return None

    from soilmoisture_points_validation import dense_validation_and_spiking

    spiking_dir = args.output_dir / "Validation_2stage"
    argv = [
        "--validation-dir",
        str(validation_dir),
        "--input-csv",
        str(generic_csv),
        "--output-dir",
        str(spiking_dir),
        "--random-repeats",
        str(args.random_repeats),
    ]
    if args.force_features:
        argv.append("--force-features")
    if args.allow_non_emt:
        argv.append("--allow-non-emt")
    rc = dense_validation_and_spiking.main(argv)
    if rc != 0:
        raise SystemExit(rc)
    return spiking_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_emt_branch(args.allow_non_emt)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        generic_csv = run_downloader(args)
    except WeatherLinkError as exc:
        print(f"WeatherLink error: {exc}", file=sys.stderr)
        return 2
    df = validate_generic_csv(generic_csv)

    validation_dir = None
    if not args.skip_model_validation:
        validation_dir = run_model_validation(args, generic_csv)

    spiking_dir = None
    if not args.skip_spiking:
        if validation_dir is None:
            validation_dir = args.output_dir / "model6_point_validation"
        spiking_dir = run_spiking(args, generic_csv, validation_dir, df)

    print("\nWeatherLink model6 workflow complete")
    print("------------------------------------")
    print(f"Generic CSV: {generic_csv}")
    if validation_dir:
        print(f"Model6 validation: {validation_dir}")
    if spiking_dir:
        print(f"Two-stage spiking analysis: {spiking_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
