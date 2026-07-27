"""Small WeatherLink v2 API helper used by the validation workflow.

The public WeatherLink v2 docs currently describe authentication as:

- pass the API key as the ``api-key`` query parameter; and
- pass the API secret as the ``X-Api-Secret`` HTTP header.

This module deliberately avoids third-party HTTP dependencies so it can run in
the existing project environment.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


API_BASE = "https://api.weatherlink.com/v2"


class WeatherLinkError(RuntimeError):
    """Raised when WeatherLink returns an error response or invalid JSON."""


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file without echoing secrets.

    The parser is intentionally tiny: blank lines and comments are ignored;
    quotes around values are stripped.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def credentials_from_env(
    *,
    api_key: str | None = None,
    api_secret: str | None = None,
    env_file: Path | None = None,
) -> tuple[str, str]:
    """Return WeatherLink credentials from CLI args, env vars or env file."""
    file_values = load_env_file(env_file) if env_file else {}
    key = (
        api_key
        or os.environ.get("WEATHERLINK_API_KEY")
        or file_values.get("WEATHERLINK_API_KEY")
        or file_values.get("API_KEY")
    )
    secret = (
        api_secret
        or os.environ.get("WEATHERLINK_API_SECRET")
        or file_values.get("WEATHERLINK_API_SECRET")
        or file_values.get("API_SECRET")
    )
    if not key or not secret:
        raise WeatherLinkError(
            "WeatherLink v2 needs both an API key and API secret. Provide them "
            "as WEATHERLINK_API_KEY and WEATHERLINK_API_SECRET environment "
            "variables, or in a local .secrets/weatherlink.env file."
        )
    return key, secret


class WeatherLinkClient:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str = API_BASE,
        demo: bool = False,
        timeout_s: int = 60,
        retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.demo = demo
        self.timeout_s = timeout_s
        self.retries = retries

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        params["api-key"] = self.api_key
        if self.demo:
            params["demo"] = "true"
        url = f"{self.base_url}/{path.lstrip('/')}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "X-Api-Secret": self.api_secret,
                "User-Agent": "DownscalingMoistureModel WeatherLink validation",
            },
            method="GET",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = response.read().decode("utf-8")
                try:
                    return json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise WeatherLinkError(f"WeatherLink returned invalid JSON for {path}") from exc
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise WeatherLinkError(
                    f"WeatherLink API error {exc.code} for {path}: {body[:800]}"
                ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 10))
                    continue
                break
        raise WeatherLinkError(f"WeatherLink request failed for {path}: {last_error}")

    def stations(self, station_ids: list[str] | None = None) -> dict[str, Any]:
        if station_ids:
            return self.get(f"/stations/{','.join(map(str, station_ids))}")
        return self.get("/stations")

    def sensors(self, sensor_ids: list[str] | None = None) -> dict[str, Any]:
        if sensor_ids:
            return self.get(f"/sensors/{','.join(map(str, sensor_ids))}")
        return self.get("/sensors")

    def nodes(self, node_ids: list[str] | None = None) -> dict[str, Any]:
        if node_ids:
            return self.get(f"/nodes/{','.join(map(str, node_ids))}")
        return self.get("/nodes")

    def sensor_catalog(self) -> dict[str, Any]:
        return self.get("/sensor-catalog")

    def historic(self, station_id: str, start_timestamp: int, end_timestamp: int) -> dict[str, Any]:
        return self.get(
            f"/historic/{station_id}",
            {
                "start-timestamp": int(start_timestamp),
                "end-timestamp": int(end_timestamp),
            },
        )
