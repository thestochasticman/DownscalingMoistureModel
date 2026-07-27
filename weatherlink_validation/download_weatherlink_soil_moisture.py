"""Download Davis WeatherLink v2 soil-moisture data and normalize it for model6.

The output CSV is intentionally shaped like the existing dense-point CSV reader:

    Date, Time, Point_number, Soil_moisture, x_3577, y_3577

where ``Soil_moisture`` must be in volumetric percent if the file is used for
handout-style RMSE/NSE validation against model6. Davis #6440 soil-moisture
sensors commonly report soil-water tension in centibars, so this script keeps
the raw value and only writes ``Soil_moisture`` when a valid conversion mode is
selected.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weatherlink_validation.weatherlink_client import (  # noqa: E402
    WeatherLinkClient,
    WeatherLinkError,
    credentials_from_env,
)


DEFAULT_ENV_FILE = REPO_ROOT / ".secrets" / "weatherlink.env"
DEFAULT_OUTPUT = REPO_ROOT / "weatherlink_validation" / "outputs"
DEFAULT_FIELD_REGEX = r"(?i)(moist.*soil|soil.*moist)"
CENTIBAR_UNITS = {"centibar", "centibars", "cb", "cbar", "kpa"}


@dataclass
class Metadata:
    stations: list[dict[str, Any]]
    sensors: list[dict[str, Any]]
    nodes: list[dict[str, Any]]
    catalog: dict[str, Any]


def parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD date, got {value!r}") from exc


def nested_items(obj: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key), value
            yield from nested_items(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from nested_items(value)


def first_value(obj: dict[str, Any], keys: Iterable[str]) -> Any:
    lowered = {str(k).lower(): v for k, v in obj.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def station_timezone(station: dict[str, Any], override: str | None) -> ZoneInfo:
    if override:
        return ZoneInfo(override)
    for key in [
        "time_zone",
        "timezone",
        "tz",
        "iana_time_zone",
        "station_timezone",
    ]:
        value = station.get(key)
        if isinstance(value, str) and value:
            try:
                return ZoneInfo(value)
            except Exception:
                continue
    return ZoneInfo("UTC")


def unix_window_for_local_days(start_day: date, end_day: date, tz: ZoneInfo) -> tuple[int, int]:
    start_local = datetime.combine(start_day, time.min, tzinfo=tz)
    end_local = datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=tz)
    return int(start_local.timestamp()), int(end_local.timestamp())


def chunk_windows(start_ts: int, end_ts: int, max_seconds: int = 86_400) -> Iterable[tuple[int, int]]:
    cursor = int(start_ts)
    final = int(end_ts)
    while cursor < final:
        nxt = min(cursor + max_seconds, final)
        yield cursor, nxt
        cursor = nxt


def local_record_date(ts: int, tz: ZoneInfo) -> tuple[str, str]:
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(tz)
    day = dt.date()
    if dt.timetz().hour == 0 and dt.timetz().minute == 0 and dt.timetz().second == 0:
        # WeatherLink archive timestamps are interval end-times; a midnight
        # record belongs to the previous local day.
        day = day - timedelta(days=1)
    return day.isoformat(), dt.strftime("%H:%M:%S")


def list_from_response(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    return []


def fetch_metadata(client: WeatherLinkClient, station_ids: list[str] | None) -> Metadata:
    stations = list_from_response(client.stations(station_ids), "stations")
    sensors = list_from_response(client.sensors(), "sensors")
    try:
        nodes = list_from_response(client.nodes(), "nodes")
    except WeatherLinkError:
        nodes = []
    try:
        catalog = client.sensor_catalog()
    except WeatherLinkError:
        catalog = {}
    return Metadata(stations=stations, sensors=sensors, nodes=nodes, catalog=catalog)


def metadata_summary(metadata: Metadata) -> pd.DataFrame:
    station_by_id = {str(s.get("station_id")): s for s in metadata.stations}
    rows = []
    for sensor in metadata.sensors:
        station_id = str(
            first_value(sensor, ["station_id", "parent_station_id", "owner_station_id"]) or ""
        )
        station = station_by_id.get(station_id, {})
        rows.append(
            {
                "station_id": station_id,
                "station_name": station.get("station_name", ""),
                "lsid": sensor.get("lsid", sensor.get("sensor_id", "")),
                "sensor_type": sensor.get("sensor_type", ""),
                "data_structure_type": sensor.get("data_structure_type", ""),
                "product_name": sensor.get("product_name", ""),
                "sensor_name": sensor.get("sensor_name", sensor.get("name", "")),
                "station_lat": station.get("latitude", ""),
                "station_lon": station.get("longitude", ""),
            }
        )
    return pd.DataFrame(rows)


def catalog_entry(catalog: dict[str, Any], sensor_type: Any) -> dict[str, Any] | None:
    target = str(sensor_type)
    for item in catalog.get("sensor_types", []) if isinstance(catalog, dict) else []:
        if isinstance(item, dict) and str(item.get("sensor_type")) == target:
            return item
    return None


def field_catalog_info(
    catalog: dict[str, Any],
    sensor_type: Any,
    data_structure_type: Any,
    field: str,
) -> dict[str, Any]:
    entry = catalog_entry(catalog, sensor_type)
    if not entry:
        return {}

    # The catalog has appeared in both compact ``data_structure`` form and
    # nested ``data_structures`` form. Recursively search dictionaries named
    # data_structure and prefer matching data_structure_type where available.
    matches: list[dict[str, Any]] = []
    stack = [entry]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "data_structure" in item and isinstance(item["data_structure"], dict):
                ds = item["data_structure"]
                if field in ds and isinstance(ds[field], dict):
                    score = 1 if str(item.get("data_structure_type", "")) == str(data_structure_type) else 0
                    matches.append({"score": score, **ds[field]})
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    if matches:
        return sorted(matches, key=lambda x: x.get("score", 0), reverse=True)[0]
    return {}


def is_soil_moisture_field(
    field: str,
    value: Any,
    sensor: dict[str, Any],
    catalog: dict[str, Any],
    field_regex: re.Pattern[str],
) -> bool:
    if field in {"ts", "tx_id", "arch_int", "tz_offset"}:
        return False
    if value is None:
        return False
    try:
        float(value)
    except Exception:
        return False
    if field_regex.search(field):
        return True
    return False


def infer_depth_cm(row: dict[str, Any]) -> float:
    """Infer Drill & Drop profile depth where WeatherLink gives numbered fields.

    Davis Drill & Drop 60 cm and 90 cm profiles are represented as
    ``moist_soil_last_1``, ``moist_soil_last_2``, ... where the suffix generally
    corresponds to 10 cm increments.
    """
    match = re.search(r"_(\d+)$", str(row.get("raw_field") or ""))
    if not match:
        return np.nan
    sensor_type = str(row.get("sensor_type") or "")
    if sensor_type in {"116", "208"}:
        return float(int(match.group(1)) * 10)
    return np.nan


def normalize_unit(unit: Any) -> str:
    return str(unit or "").strip().lower()


def load_location_map(path: Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    aliases = {
        "sensor_lsid": "lsid",
        "weatherlink_lsid": "lsid",
        "longitude": "lon",
        "latitude": "lat",
        "x": "lon",
        "y": "lat",
        "point_number": "point",
        "point_id": "point",
        "alpha": "alpha_1_per_cm",
        "vg_alpha": "alpha_1_per_cm",
        "vg_n": "n",
        "theta_saturated": "theta_s",
        "theta_residual": "theta_r",
    }
    raw = raw.rename(columns={c: aliases.get(c, c) for c in raw.columns})
    for col in ["lsid", "station_id", "field", "point"]:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.strip()
    for col in ["lon", "lat", "depth_cm", "theta_r", "theta_s", "alpha_1_per_cm", "n"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw


def match_location(location_map: pd.DataFrame, row: dict[str, Any]) -> dict[str, Any]:
    if location_map.empty:
        return {}
    sub = location_map.copy()
    if "lsid" in sub.columns and str(row.get("lsid")):
        exact = sub[sub["lsid"].astype(str) == str(row.get("lsid"))]
        if not exact.empty:
            sub = exact
    if "station_id" in sub.columns and str(row.get("station_id")):
        exact = sub[sub["station_id"].astype(str) == str(row.get("station_id"))]
        if not exact.empty:
            sub = exact
    if "field" in sub.columns and str(row.get("raw_field")):
        exact = sub[sub["field"].astype(str) == str(row.get("raw_field"))]
        if not exact.empty:
            sub = exact
    if sub.empty:
        return {}
    return sub.iloc[0].dropna().to_dict()


def station_coords(station: dict[str, Any]) -> tuple[float, float] | tuple[None, None]:
    lon = first_value(station, ["longitude", "lon", "lng"])
    lat = first_value(station, ["latitude", "lat"])
    try:
        return float(lon), float(lat)
    except Exception:
        return None, None


def as_fraction(value: float) -> float:
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def van_genuchten_percent(cb: float, theta_r: float, theta_s: float, alpha_1_per_cm: float, n: float) -> float:
    """Convert matric potential in centibars/kPa to VWC percent.

    This is only as good as the supplied hydraulic parameters. ``theta_r`` and
    ``theta_s`` may be supplied as fractions or percents.
    """
    cb = max(float(cb), 0.0)
    theta_r = as_fraction(theta_r)
    theta_s = as_fraction(theta_s)
    alpha = float(alpha_1_per_cm)
    n = float(n)
    if theta_s <= theta_r or alpha <= 0 or n <= 1:
        return np.nan
    h_cm = cb * 10.197162129779  # 1 kPa/centibar ≈ 10.197 cm water head.
    m = 1.0 - 1.0 / n
    se = (1.0 + (alpha * h_cm) ** n) ** (-m)
    theta = theta_r + (theta_s - theta_r) * se
    return float(theta * 100.0)


def convert_value(
    row: dict[str, Any],
    loc: dict[str, Any],
    value_mode: str,
) -> tuple[float, str, str]:
    raw = float(row["raw_value"])
    units = normalize_unit(row.get("raw_units"))
    field = str(row.get("raw_field") or "")

    if value_mode == "centibar_as_percent":
        return raw, "centibar_as_percent", "not_recommended"

    if value_mode == "percent" or (
        value_mode == "auto" and ("%" in units or "percent" in units or field.endswith("_pct"))
    ):
        return raw, "percent", ""

    looks_centibar = units in CENTIBAR_UNITS or "centibar" in units or "kpa" in units
    if value_mode == "centibar_vg" or (value_mode == "auto" and looks_centibar):
        required = ["theta_r", "theta_s", "alpha_1_per_cm", "n"]
        if all(k in loc and pd.notna(loc[k]) for k in required):
            return (
                van_genuchten_percent(
                    raw,
                    loc["theta_r"],
                    loc["theta_s"],
                    loc["alpha_1_per_cm"],
                    loc["n"],
                ),
                "centibar_vg",
                "",
            )
        return np.nan, "unconverted_centibar", "missing_van_genuchten_parameters"

    return np.nan, "unconverted", f"unsupported_units:{units or 'unknown'}"


def apply_relative_wetness(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["obs_sm_pct"] = np.nan
    for point, idx in out.groupby("point", sort=False).groups.items():
        sub = out.loc[idx]
        raw = pd.to_numeric(sub["raw_value"], errors="coerce")
        if raw.notna().sum() < 2 or raw.max() == raw.min():
            continue
        # For centibars/tension, lower is wetter. Scale within sensor only.
        out.loc[idx, "obs_sm_pct"] = 100.0 * (raw.max() - raw) / (raw.max() - raw.min())
        out.loc[idx, "conversion_mode"] = "relative_wetness_0_100"
    return out


def flatten_historic_payload(
    payload: dict[str, Any],
    station: dict[str, Any],
    tz: ZoneInfo,
    catalog: dict[str, Any],
    field_regex: re.Pattern[str],
    sensor_metadata_by_lsid: dict[str, dict[str, Any]] | None = None,
    allowed_lsids: set[str] | None = None,
) -> list[dict[str, Any]]:
    station_id = payload.get("station_id", station.get("station_id"))
    station_name = station.get("station_name", "")
    station_lon, station_lat = station_coords(station)
    sensor_metadata_by_lsid = sensor_metadata_by_lsid or {}
    rows: list[dict[str, Any]] = []
    for sensor in payload.get("sensors", []):
        if not isinstance(sensor, dict):
            continue
        lsid = sensor.get("lsid", sensor.get("sensor_id"))
        if allowed_lsids is not None and str(lsid) not in allowed_lsids:
            continue
        sensor_meta = sensor_metadata_by_lsid.get(str(lsid), {})
        sensor_type = sensor.get("sensor_type")
        data_structure_type = sensor.get("data_structure_type")
        lon = first_value(sensor_meta, ["longitude", "lon", "lng"])
        lat = first_value(sensor_meta, ["latitude", "lat"])
        try:
            lon, lat = float(lon), float(lat)
        except Exception:
            lon, lat = station_lon, station_lat
        for record in sensor.get("data", []) or []:
            if not isinstance(record, dict) or "ts" not in record:
                continue
            local_date, local_time = local_record_date(int(record["ts"]), tz)
            for field, value in record.items():
                if not is_soil_moisture_field(field, value, sensor, catalog, field_regex):
                    continue
                info = field_catalog_info(catalog, sensor_type, data_structure_type, field)
                rows.append(
                    {
                        "station_id": str(station_id),
                        "station_name": station_name,
                        "lsid": str(lsid),
                        "sensor_type": sensor_type,
                        "data_structure_type": data_structure_type,
                        "ts": int(record["ts"]),
                        "datetime_utc": datetime.fromtimestamp(int(record["ts"]), tz=timezone.utc).isoformat(),
                        "local_date": local_date,
                        "local_time": local_time,
                        "tx_id": record.get("tx_id"),
                        "node_name": sensor_meta.get("parent_device_name", ""),
                        "product_name": sensor_meta.get("product_name", ""),
                        "raw_field": field,
                        "raw_value": float(value),
                        "raw_units": info.get("units", ""),
                        "station_lon": station_lon,
                        "station_lat": station_lat,
                        "sensor_lon": lon,
                        "sensor_lat": lat,
                    }
                )
    return rows


def attach_locations_and_convert(
    rows: pd.DataFrame,
    location_map: pd.DataFrame,
    value_mode: str,
) -> pd.DataFrame:
    converted = []
    for row in rows.to_dict(orient="records"):
        loc = match_location(location_map, row)
        lon = loc.get("lon", row.get("sensor_lon", row.get("station_lon")))
        lat = loc.get("lat", row.get("sensor_lat", row.get("station_lat")))
        depth_cm = loc.get("depth_cm", infer_depth_cm(row))
        point = loc.get("point")
        if not point or str(point).lower() == "nan":
            depth_part = f"_{int(depth_cm)}cm" if pd.notna(depth_cm) else f"_{row.get('raw_field')}"
            point = f"wl_{row.get('lsid')}{depth_part}"
        obs, mode, reason = convert_value(row, loc, value_mode)
        row.update(
            {
                "point": str(point),
                "lon": lon,
                "lat": lat,
                "depth_cm": depth_cm,
                "obs_sm_pct": obs,
                "conversion_mode": mode,
                "conversion_note": reason,
            }
        )
        converted.append(row)
    out = pd.DataFrame(converted)
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["obs_sm_pct"] = pd.to_numeric(out["obs_sm_pct"], errors="coerce")
    return out


def aggregate_daily(rows: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "none":
        out = rows.copy()
        out["Date"] = out["local_date"]
        out["Time"] = out["local_time"]
        out["Point_number"] = out["point"]
        out["Soil_moisture"] = out["obs_sm_pct"]
        out["x_3577"] = out["lon"]
        out["y_3577"] = out["lat"]
        return out

    if method not in {"mean", "median", "last", "min", "max"}:
        raise ValueError(f"unsupported aggregation method: {method}")

    def agg_value(series: pd.Series) -> float:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            return np.nan
        if method == "mean":
            return float(series.mean())
        if method == "median":
            return float(series.median())
        if method == "last":
            return float(series.iloc[-1])
        if method == "min":
            return float(series.min())
        return float(series.max())

    rows = rows.sort_values(["point", "local_date", "ts"]).copy()
    grouped = rows.groupby(["point", "local_date"], as_index=False, sort=True)
    daily = grouped.agg(
        Soil_moisture=("obs_sm_pct", agg_value),
        raw_value_daily=("raw_value", agg_value),
        lon=("lon", "mean"),
        lat=("lat", "mean"),
        n_records=("raw_value", "size"),
        station_id=("station_id", "first"),
        station_name=("station_name", "first"),
        lsid=("lsid", "first"),
        sensor_type=("sensor_type", "first"),
        raw_field=("raw_field", "first"),
        raw_units=("raw_units", "first"),
        conversion_mode=("conversion_mode", "first"),
        conversion_note=("conversion_note", lambda x: ";".join(sorted(set(str(v) for v in x if str(v))))),
        depth_cm=("depth_cm", "first"),
        node_name=("node_name", "first"),
        product_name=("product_name", "first"),
        first_datetime_utc=("datetime_utc", "first"),
        last_datetime_utc=("datetime_utc", "last"),
    )
    daily = daily.rename(
        columns={
            "point": "Point_number",
            "local_date": "Date",
            "lon": "x_3577",
            "lat": "y_3577",
        }
    )
    daily["Time"] = f"daily_{method}"
    return daily


def write_outputs(
    rows: pd.DataFrame,
    daily: pd.DataFrame,
    metadata: Metadata,
    output_dir: Path,
    basename: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_csv": output_dir / f"{basename}_raw_records.csv",
        "generic_csv": output_dir / f"{basename}_generic_model6_validation.csv",
        "stations_json": output_dir / "weatherlink_stations.json",
        "sensors_csv": output_dir / "weatherlink_sensors.csv",
        "metadata_json": output_dir / f"{basename}_download_metadata.json",
    }
    rows.to_csv(paths["raw_csv"], index=False)
    daily.to_csv(paths["generic_csv"], index=False)
    paths["stations_json"].write_text(json.dumps(metadata.stations, indent=2, default=str))
    metadata_summary(metadata).to_csv(paths["sensors_csv"], index=False)
    finite = int(pd.to_numeric(daily["Soil_moisture"], errors="coerce").notna().sum())
    paths["metadata_json"].write_text(
        json.dumps(
            {
                "raw_records": int(len(rows)),
                "daily_rows": int(len(daily)),
                "finite_soil_moisture_rows": finite,
                "unique_points": int(daily["Point_number"].nunique()) if "Point_number" in daily else 0,
                "date_min": str(daily["Date"].min()) if "Date" in daily and not daily.empty else None,
                "date_max": str(daily["Date"].max()) if "Date" in daily and not daily.empty else None,
            },
            indent=2,
        )
    )
    return paths


def print_list_only(metadata: Metadata) -> None:
    stations = pd.DataFrame(metadata.stations)
    sensors = metadata_summary(metadata)
    print("\nAccessible stations")
    print("-------------------")
    if stations.empty:
        print("No stations returned.")
    else:
        cols = [c for c in ["station_id", "station_name", "latitude", "longitude", "time_zone", "recording_interval"] if c in stations.columns]
        print(stations[cols].to_string(index=False) if cols else stations.head(20).to_string(index=False))

    print("\nSensors")
    print("-------")
    if sensors.empty:
        print("No sensors returned.")
    else:
        print(sensors.head(200).to_string(index=False))
    print("\nTip: if the soil probes do not have point-level coordinates in WeatherLink,")
    print("create a sensor location CSV with columns: lsid,point,lon,lat,depth_cm")
    print("and, for centibar conversion: theta_r,theta_s,alpha_1_per_cm,n")


def build_dataset(args: argparse.Namespace) -> dict[str, Path]:
    api_key, api_secret = credentials_from_env(
        api_key=args.api_key,
        api_secret=args.api_secret,
        env_file=args.env_file,
    )
    client = WeatherLinkClient(api_key, api_secret, demo=args.demo)
    metadata = fetch_metadata(client, args.station_id)

    if args.list_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metadata_summary(metadata).to_csv(args.output_dir / "weatherlink_sensors.csv", index=False)
        (args.output_dir / "weatherlink_stations.json").write_text(
            json.dumps(metadata.stations, indent=2, default=str)
        )
        (args.output_dir / "weatherlink_sensors_raw.json").write_text(
            json.dumps(metadata.sensors, indent=2, default=str)
        )
        (args.output_dir / "weatherlink_nodes.json").write_text(
            json.dumps(metadata.nodes, indent=2, default=str)
        )
        print_list_only(metadata)
        return {
            "sensors_csv": args.output_dir / "weatherlink_sensors.csv",
            "stations_json": args.output_dir / "weatherlink_stations.json",
            "sensors_json": args.output_dir / "weatherlink_sensors_raw.json",
            "nodes_json": args.output_dir / "weatherlink_nodes.json",
        }

    if not metadata.stations:
        raise SystemExit("No WeatherLink stations were returned for these credentials.")

    requested = set(str(x) for x in args.station_id) if args.station_id else None
    stations = [
        s for s in metadata.stations
        if requested is None or str(s.get("station_id")) in requested or str(s.get("station_id_uuid")) in requested
    ]
    if not stations:
        raise SystemExit(f"No matching stations found for --station-id {args.station_id}")

    field_regex = re.compile(args.field_regex)
    location_map = load_location_map(args.sensor_locations_csv)
    allowed_lsids = set(str(x) for x in args.lsid) if getattr(args, "lsid", None) else None
    sensor_metadata_by_lsid = {
        str(s.get("lsid", s.get("sensor_id"))): s
        for s in metadata.sensors
        if isinstance(s, dict)
    }
    all_rows: list[dict[str, Any]] = []
    for station in stations:
        station_id = str(station.get("station_id") or station.get("station_id_uuid"))
        tz = station_timezone(station, args.timezone)
        start_ts, end_ts = unix_window_for_local_days(args.start_date, args.end_date, tz)
        print(f"Downloading station {station_id} ({station.get('station_name', '')}) in timezone {tz.key}", flush=True)
        for i, (start, end) in enumerate(chunk_windows(start_ts, end_ts), 1):
            print(f"  historic chunk {i}: {start} -> {end}", flush=True)
            payload = client.historic(station_id, start, end)
            all_rows.extend(
                flatten_historic_payload(
                    payload,
                    station,
                    tz,
                    metadata.catalog,
                    field_regex,
                    sensor_metadata_by_lsid=sensor_metadata_by_lsid,
                    allowed_lsids=allowed_lsids,
                )
            )

    raw = pd.DataFrame(all_rows)
    if raw.empty:
        raise SystemExit(
            "No soil-moisture-like fields were found in the downloaded WeatherLink historic data. "
            "Try --list-only to inspect sensors, or adjust --field-regex."
        )

    raw = raw.drop_duplicates(subset=["station_id", "lsid", "ts", "raw_field"]).reset_index(drop=True)
    raw = raw[(raw["local_date"] >= args.start_date.isoformat()) & (raw["local_date"] <= args.end_date.isoformat())]
    rows = attach_locations_and_convert(raw, location_map, args.value_mode)
    if args.value_mode == "relative_wetness":
        rows = apply_relative_wetness(rows)

    daily = aggregate_daily(rows, args.daily_agg)
    if not args.keep_unconverted:
        daily = daily[pd.to_numeric(daily["Soil_moisture"], errors="coerce").notna()].copy()

    paths = write_outputs(rows, daily, metadata, args.output_dir, args.basename)

    finite = int(pd.to_numeric(daily["Soil_moisture"], errors="coerce").notna().sum())
    print("\nWeatherLink normalization summary")
    print("---------------------------------")
    print(f"Raw soil-moisture-like records: {len(rows)}")
    print(f"Generic daily rows: {len(daily)}")
    print(f"Finite Soil_moisture rows usable for model validation: {finite}")
    print(f"Unique points: {daily['Point_number'].nunique() if not daily.empty else 0}")
    print(f"Wrote generic validation CSV: {paths['generic_csv']}")
    if finite == 0:
        print("\nNo usable percent soil-moisture observations were produced.")
        print("For Davis centibar sensors, supply --sensor-locations-csv with")
        print("theta_r, theta_s, alpha_1_per_cm and n, then use --value-mode centibar_vg.")
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download WeatherLink v2 soil-moisture sensor data and convert it to the generic validation CSV format."
    )
    parser.add_argument("--start-date", type=parse_day, required=False)
    parser.add_argument("--end-date", type=parse_day, required=False)
    parser.add_argument("--station-id", action="append", help="WeatherLink station ID or UUID; repeat for multiple. Defaults to all accessible stations.")
    parser.add_argument("--lsid", action="append", help="Optional sensor lsid filter; repeat for multiple sensors.")
    parser.add_argument("--timezone", help="IANA timezone override, e.g. Australia/Sydney. Defaults to station metadata or UTC.")
    parser.add_argument("--api-key", help="WeatherLink API key. Prefer env vars or .secrets/weatherlink.env so it is not saved in shell history.")
    parser.add_argument("--api-secret", help="WeatherLink API secret. Prefer env vars or .secrets/weatherlink.env.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--demo", action="store_true", help="Use WeatherLink demo mode.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--basename", default="weatherlink_soil_moisture")
    parser.add_argument("--sensor-locations-csv", type=Path, help="Optional mapping: lsid,point,lon,lat,depth_cm plus optional van Genuchten parameters.")
    parser.add_argument("--field-regex", default=DEFAULT_FIELD_REGEX)
    parser.add_argument(
        "--value-mode",
        choices=["auto", "percent", "centibar_vg", "relative_wetness", "centibar_as_percent"],
        default="auto",
        help="How to turn raw WeatherLink values into Soil_moisture percent. auto uses percent fields or centibar_vg when calibration params exist.",
    )
    parser.add_argument(
        "--daily-agg",
        choices=["median", "mean", "last", "min", "max", "none"],
        default="median",
        help="Daily aggregation for the model6 daily product.",
    )
    parser.add_argument("--keep-unconverted", action="store_true", help="Keep rows with blank Soil_moisture for auditing.")
    parser.add_argument("--list-only", action="store_true", help="Only list stations/sensors and write metadata; do not download historic records.")
    args = parser.parse_args(argv)
    if not args.list_only:
        if args.start_date is None or args.end_date is None:
            parser.error("--start-date and --end-date are required unless --list-only is used")
        if args.end_date < args.start_date:
            parser.error("--end-date must be on or after --start-date")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_dataset(args)
    except WeatherLinkError as exc:
        print(f"WeatherLink error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
