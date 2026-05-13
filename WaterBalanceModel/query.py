"""Query specification for water balance model runs.

The ``WaterBalanceQuery`` class defines a request to run the water balance
model over a region and time range. It follows the PaddockTS Query pattern:
immutable, hashable, with auto-generated stub for caching.
"""

from __future__ import annotations

import math
from datetime import date
from hashlib import sha256
from os import makedirs
from typing import Optional

from attrs import Factory as F
from attrs import field, frozen
from typing_extensions import Self

from WaterBalanceModel.config import config

_encode = lambda x: sha256(x.encode()).hexdigest()[:16]
_build_stub = F(
    lambda s: _encode(''.join([str(s.bbox), str(s.start), str(s.end), str(s.resolution)])),
    takes_self=True,
)


@frozen
class WaterBalanceQuery:
    """Query for water balance model execution.

    Extends the PaddockTS Query pattern with water-balance-specific attributes.
    The object is immutable and hashable; re-running with the same inputs
    yields the same ``stub`` and reuses cached files on disk.

    Attributes:
        bbox: Bounding box [west, south, east, north] in EPSG:4326.
        start: Inclusive start date.
        end: Inclusive end date.
        resolution: Spatial resolution in meters (default 10m from Sentinel-2).
        stub: Identifier for caching (auto-generated from inputs if not provided).

    Derived attributes (computed from inputs):
        tmp_dir: Directory for intermediate cached data.
        out_dir: Directory for final outputs.
        centre_lon: Longitude of bbox centre.
        centre_lat: Latitude of bbox centre.
        soil_moisture_path: Output path for soil moisture Zarr.
        et_path: Output path for ET Zarr.
        climate_path: Cached climate data path.
        terrain_path: Cached terrain data path.
    """

    bbox: list[float]
    start: date
    end: date
    resolution: float = 10.0
    stub: str = field(default=_build_stub)

    # Derived paths (non-init fields)
    tmp_dir: str = field(init=False)
    out_dir: str = field(init=False)
    centre_lon: float = field(init=False)
    centre_lat: float = field(init=False)
    soil_moisture_path: str = field(init=False)
    et_path: str = field(init=False)
    climate_path: str = field(init=False)
    terrain_path: str = field(init=False)
    ndvi_path: str = field(init=False)
    smips_path: str = field(init=False)
    soil_params_path: str = field(init=False)

    @tmp_dir.default
    def _tmp_dir(s: Self) -> str:
        return f'{config.tmp_dir}/{s.stub}'

    @out_dir.default
    def _out_dir(s: Self) -> str:
        return f'{config.out_dir}/{s.stub}'

    @centre_lon.default
    def _centre_lon(s: Self) -> float:
        return (s.bbox[0] + s.bbox[2]) / 2

    @centre_lat.default
    def _centre_lat(s: Self) -> float:
        return (s.bbox[1] + s.bbox[3]) / 2

    @soil_moisture_path.default
    def _soil_moisture_path(s: Self) -> str:
        return f'{s.out_dir}/{s.stub}_soil_moisture.zarr'

    @et_path.default
    def _et_path(s: Self) -> str:
        return f'{s.tmp_dir}/{s.stub}_evapotranspiration.zarr'

    @climate_path.default
    def _climate_path(s: Self) -> str:
        return f'{s.tmp_dir}/{s.stub}_climate.csv'

    @terrain_path.default
    def _terrain_path(s: Self) -> str:
        return f'{s.tmp_dir}/{s.stub}_terrain.nc'

    @ndvi_path.default
    def _ndvi_path(s: Self) -> str:
        return f'{s.tmp_dir}/{s.stub}_ndvi.zarr'

    @smips_path.default
    def _smips_path(s: Self) -> str:
        return f'{s.tmp_dir}/{s.stub}_smips.nc'

    @soil_params_path.default
    def _soil_params_path(s: Self) -> str:
        return f'{s.tmp_dir}/{s.stub}_soil_params.nc'

    def __attrs_post_init__(s: Self) -> None:
        makedirs(s.tmp_dir, exist_ok=True)
        makedirs(s.out_dir, exist_ok=True)

    def __str__(s: Self) -> str:
        return s.stub

    @classmethod
    def from_lat_lon(
        cls,
        lat: float,
        lon: float,
        buffer_km: float,
        start: date,
        end: date,
        stub: Optional[str] = None,
        resolution: float = 10.0,
    ) -> WaterBalanceQuery:
        """Build a Query from a centre point and buffer in kilometres.

        Convenience constructor for users who think in "X km around a point"
        rather than bounding-box corners.

        Args:
            lat: Centre latitude (EPSG:4326).
            lon: Centre longitude (EPSG:4326).
            buffer_km: Half-width of square buffer in kilometres.
            start: Inclusive start date.
            end: Inclusive end date.
            stub: Optional custom identifier for caching.
            resolution: Spatial resolution in meters (default 10).

        Returns:
            WaterBalanceQuery with computed bounding box.
        """
        lat_buffer = buffer_km / 111.0
        lon_buffer = buffer_km / (111.0 * math.cos(math.radians(lat)))
        bbox = [
            lon - lon_buffer,  # west
            lat - lat_buffer,  # south
            lon + lon_buffer,  # east
            lat + lat_buffer,  # north
        ]
        if stub is not None:
            return cls(bbox=bbox, start=start, end=end, stub=stub, resolution=resolution)
        return cls(bbox=bbox, start=start, end=end, resolution=resolution)
