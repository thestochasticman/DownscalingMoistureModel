"""Exact regridding of the lab's native rasters onto the Sentinel-2 grid.

The downscaling model is the one consumer that needs every source on one
grid: pysentinel2's fixed EPSG:6933 / 10 m lattice. Every other source
(SMIPS ~1 km, SILO 5 km, OzWALD 500 m, SLGA 90 m, COP-DEM 30 m) is
coarser than 10 m and delivered in EPSG:4326, and the stores never
resample. So the data-preserving way onto 10 m is *block replication*:
each fine pixel takes the value of the native pixel its centre falls
in. No interpolation, no value invented, and -- because EPSG:4326 and
EPSG:6933 are both cylindrical (x depends only on longitude, y only on
latitude) -- the mapping is separable and exact: one integer index per
target row and one per target column, computed with pyproj once.

That separability also makes the mapping invertible. :func:`aggregate`
groups fine pixels back by their native pixel with a bincount, so a
10 m prediction field can be averaged back to the SMIPS pixel it came
from (the mass-balance check) and :func:`coverage` says which native
pixels are only partly inside the window.

:func:`smooth_upsample` is the one lossy option, opt-in, for a continuous
model *input* like elevation where a bilinear surface is wanted. It is
never used for a target or for anything aggregated back.

Every regridded dataset carries ``attrs['regrid']`` and :func:`regrid`
refuses to regrid one again: no double resampling, ever.

Inputs follow the lab stores' georeferencing contract: ``attrs['crs']``
and ``attrs['transform']`` (six affine numbers, pixel-edge origin,
north-up) on the dataset, coordinates at pixel centres.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from pyproj import Transformer

from pysentinel2 import grid as s2

Window = tuple[int, int, int, int]
TARGET_CRS = s2.CRS
_to_4326 = Transformer.from_crs(s2.CRS, 'EPSG:4326', always_xy=True)
_CYLINDRICAL = {'EPSG:4326', 'EPSG:6933'}


def s2_window(bbox: list[float]) -> Window:
    """Target pixel window on the Sentinel-2 grid: ``bbox`` snapped outward
    to whole 10 m pixels. Every source regridded to the same window over
    the same bbox has identical shape and coordinates."""
    return s2.tight_window_for_bbox(bbox)


def s2_coords(window: Window):
    """``(y, x)`` pixel-centre coordinates of a target window, EPSG:6933."""
    return s2.coords_for_window(window)


def _check_native(ds: xr.Dataset) -> tuple[tuple, str]:
    if 'regrid' in ds.attrs:
        raise ValueError('dataset is already regridded (attrs["regrid"] present); '
                         'regridding twice is refused -- start from the native read')
    if 'transform' not in ds.attrs or 'crs' not in ds.attrs:
        raise ValueError("dataset lacks the stores' georeferencing attrs 'crs'/'transform'")
    t = tuple(float(v) for v in ds.attrs['transform'])
    if len(t) != 6 or t[1] != 0 or t[3] != 0 or t[0] <= 0 or t[4] >= 0:
        raise ValueError(f'transform {t} is not a north-up axis-aligned affine')
    return t, str(ds.attrs['crs']).upper()


def index_maps(transform: tuple, crs: str, window: Window,
               native_shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, cols)``: for each target row / column of ``window``, the
    native pixel row / column whose cell contains that target pixel's
    centre. Exact for cylindrical CRSs. ``-1`` where the centre falls
    outside the native raster (``native_shape`` = (rows, cols) of the
    array the indices will address).
    """
    crs = crs.upper()
    if crs not in _CYLINDRICAL:
        raise ValueError(f'{crs}: only cylindrical CRSs (EPSG:4326, EPSG:6933) map separably; '
                         f'add a 2-D nearest fallback before using this source')
    a, _, c, _, e, f = transform
    y, x = s2_coords(window)
    if crs == 'EPSG:6933':
        xs, ys = x, y
    else:
        xs, _ = _to_4326.transform(x, np.full_like(x, y[0]))     # lon depends on x only
        _, ys = _to_4326.transform(np.full_like(y, x[0]), y)     # lat depends on y only
    cols = np.floor((np.asarray(xs) - c) / a).astype('int64')
    rows = np.floor((np.asarray(ys) - f) / e).astype('int64')
    if native_shape is not None:
        nr, nc = native_shape
        rows = np.where((rows >= 0) & (rows < nr), rows, -1)
        cols = np.where((cols >= 0) & (cols < nc), cols, -1)
    else:
        rows = np.where(rows >= 0, rows, -1)
        cols = np.where(cols >= 0, cols, -1)
    return rows.astype('int32'), cols.astype('int32')


def replicate(native: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Block-replicate a native array ``(..., ny, nx)`` onto the target
    window: ``out[..., i, j] = native[..., rows[i], cols[j]]``. Target
    pixels mapped to ``-1`` become NaN. Zero loss: every value is a
    native value."""
    native = np.asarray(native)
    r = np.where(rows < 0, 0, rows)
    c = np.where(cols < 0, 0, cols)
    out = native[..., r[:, None], c[None, :]]
    bad = (rows < 0)[:, None] | (cols < 0)[None, :]
    if bad.any():
        out = out.astype('float32', copy=True) if out.dtype.kind in 'iu' else out.copy()
        out[..., bad] = np.nan
    return out


def _flat_ids(rows, cols, native_shape):
    nr, nc = native_shape
    ids = rows[:, None] * nc + cols[None, :]
    valid = (rows >= 0)[:, None] & (cols >= 0)[None, :]
    return ids, valid


def coverage(rows: np.ndarray, cols: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
    """How many target pixels fall in each native pixel, ``(nr, nc)``.
    A native pixel fully inside the window has the full count
    (about ``(native_res / 10)^2``); partial ones have fewer; zero means
    outside the window."""
    ids, valid = _flat_ids(rows, cols, native_shape)
    nr, nc = native_shape
    return np.bincount(ids[valid].ravel(), minlength=nr * nc).reshape(nr, nc)


def aggregate(fine: np.ndarray, rows: np.ndarray, cols: np.ndarray,
              native_shape: tuple[int, int], reducer: str = 'mean') -> np.ndarray:
    """The inverse of :func:`replicate`: reduce a target array
    ``(..., ny, nx)`` back to native ``(..., nr, nc)`` by native pixel.
    ``reducer`` is ``'mean'``, ``'sum'`` or ``'max'``; NaNs are ignored.
    Native pixels with no finite target pixel are NaN."""
    fine = np.asarray(fine, dtype='float64')
    ids, valid = _flat_ids(rows, cols, native_shape)
    nr, nc = native_shape
    lead = fine.shape[:-2]
    flat = fine.reshape(-1, *fine.shape[-2:])
    out = np.full((flat.shape[0], nr * nc), np.nan)
    for k in range(flat.shape[0]):
        ok = valid & np.isfinite(flat[k])
        if not ok.any():
            continue
        i, v = ids[ok].ravel(), flat[k][ok].ravel()
        n = np.bincount(i, minlength=nr * nc)
        if reducer == 'max':
            acc = np.full(nr * nc, -np.inf)
            np.maximum.at(acc, i, v)
            res = np.where(n > 0, acc, np.nan)
        else:
            s = np.bincount(i, weights=v, minlength=nr * nc)
            res = np.where(n > 0, s / np.where(n > 0, n, 1) if reducer == 'mean' else s, np.nan)
        out[k] = res
    return out.reshape(*lead, nr, nc)


def smooth_upsample(native: np.ndarray, transform: tuple, crs: str, window: Window,
                    method: str = 'bilinear', nodata=np.nan) -> np.ndarray:
    """Lossy, opt-in: ``rasterio.warp.reproject`` a native ``(..., ny, nx)``
    array onto the target window with a continuous method (``bilinear``
    or ``cubic``). For model inputs such as elevation only."""
    import rasterio.warp
    from affine import Affine
    from rasterio.enums import Resampling
    native = np.asarray(native, dtype='float32')
    lead = native.shape[:-2]
    src = native.reshape(-1, *native.shape[-2:])
    row0, row1, col0, col1 = window
    dst_transform = Affine(s2.RES, 0, s2.X0 + col0 * s2.RES, 0, -s2.RES, s2.Y_TOP - row0 * s2.RES)
    dst = np.full((src.shape[0], row1 - row0, col1 - col0), np.nan, 'float32')
    rasterio.warp.reproject(
        source=src, destination=dst,
        src_transform=Affine(*transform), src_crs=crs, src_nodata=nodata,
        dst_transform=dst_transform, dst_crs=TARGET_CRS, dst_nodata=np.nan,
        resampling=getattr(Resampling, method),
    )
    return dst.reshape(*lead, row1 - row0, col1 - col0)


def regrid(ds: xr.Dataset, bbox: list[float], method: str = 'replicate',
           methods: dict[str, str] | None = None, source_index: bool = False,
           y_dim: str = 'lat', x_dim: str = 'lon') -> xr.Dataset:
    """A native store dataset onto the Sentinel-2 window of ``bbox``.

    Args:
        ds: Native dataset with the stores' ``crs``/``transform`` attrs and
            dims ``(..., lat, lon)`` (or ``(..., y, x)`` for an EPSG:6933
            source, which is only cropped).
        bbox: ``[west, south, east, north]`` in EPSG:4326.
        method: ``'replicate'`` (exact, default) or ``'bilinear'`` /
            ``'cubic'`` (lossy, opt-in) for every variable.
        methods: Per-variable override of ``method``.
        source_index: Also return ``source_row`` / ``source_col`` (int32,
            ``-1`` outside) so values can be aggregated back by native
            pixel.

    Returns:
        xr.Dataset on dims ``(..., y, x)`` with EPSG:6933 coordinates and
        ``attrs['regrid']`` recording the native transform and the method
        per variable. Raises if ``ds`` was regridded already.
    """
    transform, crs = _check_native(ds)
    window = s2_window(bbox)
    y, x = s2_coords(window)
    if crs == 'EPSG:6933':
        y_dim, x_dim = ('y', 'x') if 'y' in ds.dims else (y_dim, x_dim)
    ny, nx = ds.sizes[y_dim], ds.sizes[x_dim]
    rows, cols = index_maps(transform, crs, window, (ny, nx))
    methods = methods or {}
    out, used = {}, {}
    for name, da in ds.data_vars.items():
        if y_dim not in da.dims or x_dim not in da.dims:
            continue
        da = da.transpose(..., y_dim, x_dim)
        m = methods.get(name, method)
        if m == 'replicate':
            arr = replicate(da.values, rows, cols)
        else:
            arr = smooth_upsample(da.values, transform, crs, window, method=m,
                                  nodata=da.attrs.get('nodata', np.nan))
        dims = tuple(d for d in da.dims if d not in (y_dim, x_dim)) + ('y', 'x')
        out[name] = xr.DataArray(arr, dims=dims, attrs=dict(da.attrs))
        used[name] = m
    if source_index:
        out['source_row'] = xr.DataArray(np.broadcast_to(rows[:, None], (len(y), len(x))).copy(),
                                         dims=('y', 'x'))
        out['source_col'] = xr.DataArray(np.broadcast_to(cols[None, :], (len(y), len(x))).copy(),
                                         dims=('y', 'x'))
    coords = {d: ds.coords[d] for d in ds.coords if d not in (y_dim, x_dim) and d in ds.dims}
    coords.update({'y': y, 'x': x})
    res = xr.Dataset(out, coords=coords)
    res.attrs = {k: v for k, v in ds.attrs.items() if k not in ('transform', 'crs')}
    res.attrs.update({
        'crs': TARGET_CRS,
        'transform': [s2.RES, 0.0, s2.X0 + window[2] * s2.RES, 0.0, -s2.RES, s2.Y_TOP - window[0] * s2.RES],
        'regrid': {'native_crs': crs, 'native_transform': list(transform),
                   'native_shape': [ny, nx], 'methods': used, 'window': list(window)},
    })
    return res


# -- offline tests ------------------------------------------------------------

_BBOX = [147.30, -35.52, 147.62, -35.10]      # Kyeamba Creek


def _synthetic(xres, yres, x0, y_top, bbox, value_fn, name='v', time=None) -> xr.Dataset:
    """A native EPSG:4326 dataset covering bbox with a pixel-indexed value."""
    west, south, east, north = bbox
    col0 = int((west - x0) // xres) - 1
    col1 = int((east - x0) // xres) + 2
    row0 = int((y_top - north) // yres) - 1
    row1 = int((y_top - south) // yres) + 2
    lon = x0 + (np.arange(col0, col1) + 0.5) * xres
    lat = y_top - (np.arange(row0, row1) + 0.5) * yres
    rr, cc = np.meshgrid(np.arange(row0, row1), np.arange(col0, col1), indexing='ij')
    data = np.broadcast_to(np.asarray(value_fn(rr, cc), dtype='float32'), rr.shape).copy()
    dims, coords = ('lat', 'lon'), {'lat': lat, 'lon': lon}
    if time is not None:
        data = np.stack([data + k for k in range(len(time))])
        dims, coords = ('time', 'lat', 'lon'), {'time': time, 'lat': lat, 'lon': lon}
    return xr.Dataset({name: (dims, data)}, coords=coords,
                      attrs={'crs': 'EPSG:4326', 'nodata': None,
                             'transform': [xres, 0.0, x0 + col0 * xres, 0.0, -yres, y_top - row0 * yres]})


# The lab's native lattices, as the stores declare them.
_SMIPS = dict(xres=0.009997566018978103, yres=0.009997121616580312, x0=112.904998779, y_top=-9.005000114)
_SILO = dict(xres=0.05, yres=0.05, x0=111.975, y_top=-9.975)
_SLGA = dict(xres=1 / 1200, yres=1 / 1200, x0=112.0, y_top=-9.0)
_COPDEM = dict(xres=1 / 3600, yres=1 / 3600, x0=-180.0, y_top=90.0)


def test_index_maps_match_brute_force():
    """The separable maps agree with transforming every pixel centre."""
    ds = _synthetic(**_SMIPS, bbox=_BBOX, value_fn=lambda r, c: r * 10000 + c)
    window = s2_window(_BBOX)
    t = ds.attrs['transform']
    rows, cols = index_maps(t, 'EPSG:4326', window, (ds.sizes['lat'], ds.sizes['lon']))
    y, x = s2_coords(window)
    xx, yy = np.meshgrid(x, y)
    lon, lat = _to_4326.transform(xx, yy)
    brute_c = np.floor((lon - t[2]) / t[0]).astype(int)
    brute_r = np.floor((lat - t[5]) / t[4]).astype(int)
    return (np.array_equal(brute_r, np.broadcast_to(rows[:, None], brute_r.shape))
            and np.array_equal(brute_c, np.broadcast_to(cols[None, :], brute_c.shape))
            and (rows >= 0).all() and (cols >= 0).all())


def test_replicate_then_aggregate_is_identity():
    """Block replication followed by per-native-pixel mean recovers the
    native values exactly on fully covered pixels; coverage flags partial ones."""
    ds = _synthetic(**_SMIPS, bbox=_BBOX, value_fn=lambda r, c: (r * 7 + c * 3) % 101 + 0.25)
    window = s2_window(_BBOX)
    shape = (ds.sizes['lat'], ds.sizes['lon'])
    rows, cols = index_maps(ds.attrs['transform'], 'EPSG:4326', window, shape)
    fine = replicate(ds['v'].values, rows, cols)
    back = aggregate(fine, rows, cols, shape, 'mean')
    cov = coverage(rows, cols, shape)
    full = cov == cov.max()
    return (fine.shape == (window[1] - window[0], window[3] - window[2])
            and full.sum() > 0 and (cov == 0).sum() > 0
            and np.array_equal(back[full], ds['v'].values[full])
            and np.isnan(back[cov == 0]).all()
            and np.allclose(aggregate(fine, rows, cols, shape, 'sum')[full],
                            ds['v'].values[full] * cov[full]))


def test_every_source_lands_on_one_grid():
    """SMIPS, SILO, SLGA and COP-DEM over one bbox: identical shapes and
    coords, each pixel a native value, provenance recorded."""
    import pandas as pd
    srcs = {
        'smips': _synthetic(**_SMIPS, bbox=_BBOX, value_fn=lambda r, c: r + c, time=pd.date_range('2020-01-01', periods=3)),
        'silo': _synthetic(**_SILO, bbox=_BBOX, value_fn=lambda r, c: r * 2 + c),
        'slga': _synthetic(**_SLGA, bbox=_BBOX, value_fn=lambda r, c: r + c * 2),
        'copdem': _synthetic(**_COPDEM, bbox=_BBOX, value_fn=lambda r, c: 300 + r % 5),
    }
    outs = {k: regrid(v, _BBOX, source_index=True) for k, v in srcs.items()}
    shapes = {tuple(o['v'].shape[-2:]) for o in outs.values()}
    xs = {tuple(np.round(o.x.values[:3], 3)) for o in outs.values()}
    ok_values = all(np.isin(np.unique(o['v'].values[~np.isnan(o['v'].values)]),
                            srcs[k]['v'].values.ravel()).all() for k, o in outs.items())
    return (len(shapes) == 1 and len(xs) == 1
            and outs['smips']['v'].dims == ('time', 'y', 'x') and outs['smips'].sizes['time'] == 3
            and ok_values and outs['smips'].attrs['crs'] == 'EPSG:6933'
            and outs['smips'].attrs['regrid']['methods'] == {'v': 'replicate'}
            and 'source_row' in outs['smips'] and int(outs['smips']['source_row'].min()) >= 0)


def test_regrid_twice_raises_and_bilinear_is_opt_in():
    ds = _synthetic(**_COPDEM, bbox=_BBOX, value_fn=lambda r, c: 300.0 + 0.5 * c)
    out = regrid(ds, _BBOX, methods={'v': 'bilinear'})
    try:
        regrid(out, _BBOX)
        return False
    except ValueError:
        pass
    rep = regrid(ds, _BBOX)
    return (out.attrs['regrid']['methods'] == {'v': 'bilinear'} and out['v'].shape == rep['v'].shape
            and np.isfinite(out['v'].values).mean() > 0.99
            and abs(float(np.nanmean(out['v'].values)) - float(np.nanmean(rep['v'].values))) < 1.0)


def test_window_edge_pixels_are_nan_not_wrapped():
    """A native raster that does not cover the whole window yields NaN,
    never a wrapped-around index."""
    ds = _synthetic(**_SILO, bbox=[147.40, -35.40, 147.50, -35.30], value_fn=lambda r, c: 5.0)
    out = regrid(ds, _BBOX)
    v = out['v'].values
    return np.isnan(v).any() and np.nanmax(v) == 5.0 and np.nanmin(v) == 5.0


def test():
    return all([
        test_index_maps_match_brute_force(),
        test_replicate_then_aggregate_is_identity(),
        test_every_source_lands_on_one_grid(),
        test_regrid_twice_raises_and_bilinear_is_opt_in(),
        test_window_edge_pixels_are_nan_not_wrapped(),
    ])


if __name__ == '__main__':
    print(test())
