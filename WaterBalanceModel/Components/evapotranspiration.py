"""Evapotranspiration computation for the water balance model.

This module implements ET = ET0 × Kc × Ks where:
- ET0: Reference evapotranspiration (FAO-56 Penman-Monteith)
- Kc: Crop coefficient (derived from NDVI)
- Ks: Soil water stress coefficient

References:
    Allen, R.G. et al. (1998). FAO Irrigation and Drainage Paper No. 56:
    Crop Evapotranspiration. FAO, Rome.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from attrs import frozen

if TYPE_CHECKING:
    from numpy.typing import NDArray


@frozen
class ETParameters:
    """Parameters for evapotranspiration computation.

    Attributes:
        kc_min: Minimum crop coefficient (bare soil).
        kc_max: Maximum crop coefficient (full vegetation).
        ndvi_min: NDVI threshold for bare soil.
        ndvi_max: NDVI threshold for full vegetation.
        p_factor: Depletion factor for Ks calculation.
    """

    kc_min: float = 0.15
    kc_max: float = 1.20
    ndvi_min: float = 0.10
    ndvi_max: float = 0.80
    p_factor: float = 0.50


def compute_et0_fao56(
    tmin: NDArray[np.float32],
    tmax: NDArray[np.float32],
    radiation: NDArray[np.float32],
    vp: NDArray[np.float32],
    wind: NDArray[np.float32],
    elevation: NDArray[np.float32],
    doy: int,
    lat: float,
) -> NDArray[np.float32]:
    """Compute reference ET using FAO-56 Penman-Monteith equation.

    Implements the standard FAO-56 equation for a hypothetical grass
    reference surface with height 0.12m, surface resistance 70 s/m,
    and albedo 0.23.

    Args:
        tmin: Minimum daily temperature (C).
        tmax: Maximum daily temperature (C).
        radiation: Solar radiation (MJ/m2/day).
        vp: Actual vapor pressure (kPa).
        wind: Wind speed at 2m height (m/s).
        elevation: Elevation (m) - can be scalar or array.
        doy: Day of year (1-366).
        lat: Latitude (degrees).

    Returns:
        Reference ET (mm/day).
    """
    # Mean temperature
    tmean = (tmin + tmax) / 2

    # Atmospheric pressure (kPa) - Eq. 7
    P = 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26

    # Psychrometric constant (kPa/C) - Eq. 8
    gamma = 0.000665 * P

    # Slope of saturation vapor pressure curve (kPa/C) - Eq. 13
    delta = 4098 * (0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2

    # Saturation vapor pressure (kPa) - Eq. 11
    e_tmin = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    e_tmax = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    es = (e_tmin + e_tmax) / 2

    # Vapor pressure deficit
    vpd = es - vp
    vpd = np.maximum(vpd, 0)  # Cannot be negative

    # Net radiation (simplified) - assume Rn ≈ 0.77 * Rs for grass
    # More accurate would use albedo and longwave, but this is reasonable
    Rn = 0.77 * radiation

    # Soil heat flux (G) - negligible for daily timestep
    G = 0

    # FAO-56 Penman-Monteith equation - Eq. 6
    numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (tmean + 273)) * wind * vpd
    denominator = delta + gamma * (1 + 0.34 * wind)

    ET0 = numerator / denominator

    # Ensure non-negative
    ET0 = np.maximum(ET0, 0)

    return ET0.astype(np.float32)


def compute_kc_from_ndvi(
    ndvi: xr.DataArray,
    params: ETParameters = ETParameters(),
) -> xr.DataArray:
    """Compute crop coefficient from NDVI using linear scaling.

    Kc = Kc_min + (Kc_max - Kc_min) * (NDVI - NDVI_min) / (NDVI_max - NDVI_min)

    Args:
        ndvi: NDVI values (can be DataArray with any dims).
        params: ET parameters specifying Kc and NDVI bounds.

    Returns:
        Crop coefficient Kc with same shape as input.
    """
    # Linear scaling
    kc = params.kc_min + (params.kc_max - params.kc_min) * (
        (ndvi - params.ndvi_min) / (params.ndvi_max - params.ndvi_min)
    )

    # Clip to valid range
    kc = kc.clip(min=params.kc_min, max=params.kc_max)

    kc.attrs.update({
        'long_name': 'Crop coefficient',
        'units': '',
    })

    return kc


def compute_ks(
    theta: xr.DataArray,
    theta_fc: xr.DataArray,
    theta_wp: xr.DataArray,
    p: float = 0.5,
) -> xr.DataArray:
    """Compute soil water stress coefficient.

    Linear stress function following FAO-56:
    - Ks = 1 when theta > theta_threshold (no stress)
    - Ks = 0 when theta <= theta_wp (full stress)
    - Linear interpolation in between

    Where theta_threshold = (1 - p) * (theta_fc - theta_wp) + theta_wp

    Args:
        theta: Current soil moisture content (same units as theta_fc/wp).
        theta_fc: Field capacity.
        theta_wp: Wilting point.
        p: Depletion factor (default 0.5). Higher p means stress starts
            at lower soil moisture.

    Returns:
        Stress coefficient Ks in [0, 1].
    """
    # Readily available water
    raw = theta_fc - theta_wp

    # Threshold where stress begins
    theta_threshold = theta_wp + (1 - p) * raw

    # Compute Ks
    ks = (theta - theta_wp) / (theta_threshold - theta_wp + 1e-10)

    # Clip to [0, 1]
    ks = ks.clip(min=0, max=1)

    ks.attrs.update({
        'long_name': 'Soil water stress coefficient',
        'units': '',
    })

    return ks


def compute_et_actual(
    et0: xr.DataArray,
    kc: xr.DataArray,
    ks: xr.DataArray,
) -> xr.DataArray:
    """Compute actual evapotranspiration.

    ET = ET0 × Kc × Ks

    Args:
        et0: Reference evapotranspiration (mm/day).
        kc: Crop coefficient.
        ks: Soil water stress coefficient.

    Returns:
        Actual evapotranspiration (mm/day).
    """
    et = et0 * kc * ks

    et.attrs.update({
        'long_name': 'Actual evapotranspiration',
        'units': 'mm/day',
    })

    return et


@frozen
class ETComputer:
    """Evapotranspiration computation engine.

    Encapsulates parameters and provides methods for computing all
    ET components in the water balance model.

    Attributes:
        params: ET parameters.
    """

    params: ETParameters = ETParameters()

    def compute_et0(
        self,
        climate: dict,
        terrain: xr.Dataset,
        doy: int,
    ) -> xr.DataArray:
        """Compute spatially distributed ET0 from climate data.

        Args:
            climate: Dict with keys 'tmin', 'tmax', 'radiation', 'vp', 'wind'.
            terrain: Terrain dataset with 'dem' variable.
            doy: Day of year.

        Returns:
            ET0 at terrain resolution (mm/day).
        """
        # Get scalar climate values (point data)
        tmin = climate['tmin']
        tmax = climate['tmax']
        radiation = climate['radiation']
        vp = climate['vp']
        wind = climate['wind']

        # Use terrain DEM for elevation
        elevation = terrain['dem'].values
        lat = float(terrain.y.mean())

        # Compute ET0
        et0 = compute_et0_fao56(
            tmin=np.full_like(elevation, tmin),
            tmax=np.full_like(elevation, tmax),
            radiation=np.full_like(elevation, radiation),
            vp=np.full_like(elevation, vp),
            wind=np.full_like(elevation, wind),
            elevation=elevation,
            doy=doy,
            lat=lat,
        )

        # Apply HLI adjustment for aspect effects
        if 'hli' in terrain:
            # Scale ET0 by heat load index (higher HLI = more sun = more ET)
            hli = terrain['hli'].values
            et0 = et0 * (0.7 + 0.6 * hli)

        return xr.DataArray(
            et0,
            dims=['y', 'x'],
            coords={'y': terrain.y, 'x': terrain.x},
            attrs={'units': 'mm/day', 'long_name': 'Reference evapotranspiration'},
        )

    def compute_kc(self, ndvi: xr.DataArray) -> xr.DataArray:
        """Compute crop coefficient from NDVI."""
        return compute_kc_from_ndvi(ndvi, self.params)

    def compute_ks(
        self,
        theta: xr.DataArray,
        theta_fc: xr.DataArray,
        theta_wp: xr.DataArray,
    ) -> xr.DataArray:
        """Compute stress coefficient."""
        return compute_ks(theta, theta_fc, theta_wp, self.params.p_factor)

    def compute_et(
        self,
        et0: xr.DataArray,
        kc: xr.DataArray,
        ks: xr.DataArray,
    ) -> xr.DataArray:
        """Compute actual ET."""
        return compute_et_actual(et0, kc, ks)
