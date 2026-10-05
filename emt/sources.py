"""Every covariate of the downscaling model, read from the lab's stores
and stacked onto the Sentinel-2 grid by :mod:`emt.regrid`.

The stores (pysmips, pysilo, pyozwald, pyslga, pycopdem, pysentinel2)
each keep their source's native lattice and never resample; this module
is the one place that puts them on one grid. Per source:

================  ==========  ======================  ===========================
source            native      into the 10 m stack     why
================  ==========  ======================  ===========================
SMIPS             ~1 km       replicate + source idx  target: recoverable exactly
SILO              5 km        replicate               forcing; no sub-cell structure
OzWALD daily      500 m*      replicate               forcing
OzWALD 8-day      500 m       replicate, 8-day time   model handles cadence
SLGA              90 m        replicate               textures: interpolation mixes
COP-DEM elevation 30 m        bilinear (opt-in)       continuous input
COP-DEM slope...  30 m        native w/ buffer, rep.  derivatives from native DEM
Sentinel-2        10 m        as is                   already on the grid
================  ==========  ======================  ===========================

(*the daily meteorology is a ~5 km product delivered on the 500 m lattice.)

Everything is a plain function of ``bbox`` and dates; the stores take
care of caching, concurrency across Gadi nodes, and the ``gaps()``
audit. Load in time chunks: a 20-year daily stack of a 50 km AOI at
10 m is 7300 x 5000 x 5000 values per variable and does not fit in RAM.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import xarray as xr

from emt import regrid

SMIPS_PRODUCTS = ('totalbucket',)
SILO_VARS = ('daily_rain', 'max_temp', 'min_temp', 'radiation', 'vp_deficit', 'et_morton_potential')
OZWALD_DAILY_VARS = ('Pg', 'Tmax', 'Tmin', 'Uavg', 'VPeff')
OZWALD_8DAY_VARS = ('Ssoil', 'Qtot', 'LAI', 'GPP')
SOIL_ATTRIBUTES = ('Clay', 'Sand', 'Silt', 'Bulk_Density', 'Available_Water_Capacity')
SOIL_DEPTHS = ('0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm')
TERRAIN_DERIVATIVES = ('slope', 'aspect', 'accumulation', 'twi', 'hli')
TERRAIN_BUFFER_M = 2000


# -- native reads (one store each) --------------------------------------------

def smips(bbox, start: date, end: date, products=SMIPS_PRODUCTS, **kw) -> xr.Dataset:
    from pysmips.store import Store
    return Store(**kw).get_ds(bbox, start, end, products=products)


def silo(bbox, start: date, end: date, variables=SILO_VARS, **kw) -> xr.Dataset:
    from pysilo.store import Store
    return Store(**kw).get_ds(bbox, start, end, variables=variables)


def ozwald(bbox, start: date, end: date, cadence: str = 'daily', variables=None, **kw) -> xr.Dataset:
    from pyozwald.store import Store
    variables = variables or (OZWALD_DAILY_VARS if cadence == 'daily' else OZWALD_8DAY_VARS)
    return Store(**kw).get_ds(bbox, start, end, cadence=cadence, variables=variables)


def soil(bbox, attributes=SOIL_ATTRIBUTES, depths=SOIL_DEPTHS, **kw) -> xr.Dataset:
    from pyslga.store import Store
    return Store(**kw).get_ds(bbox, attributes=attributes, depths=depths)


def terrain(bbox, derivatives=TERRAIN_DERIVATIVES, buffer_m: float = TERRAIN_BUFFER_M, **kw) -> xr.Dataset:
    from pycopdem.store import Store
    return Store(**kw).get_ds(bbox, derivatives=derivatives, buffer_m=buffer_m)


def sentinel2(bbox, start: date, end: date, clean: bool = True, bands=None, indices=(), **kw) -> xr.Dataset:
    from pysentinel2.cube import Cube
    return Cube(**kw).get_ds(bbox, start, end, clean=clean, bands=bands, indices=indices)


def gaps(bbox, start: date, end: date, **kw) -> dict:
    """Every store's completeness audit for the request, no network."""
    from pysmips.store import Store as Smips
    from pysilo.store import Store as Silo
    from pyozwald.store import Store as Oz
    from pyslga.store import Store as Slga
    from pycopdem.store import Store as Dem
    from pysentinel2.cube import Cube
    return {
        'smips': Smips(**kw).gaps(bbox, start, end, products=SMIPS_PRODUCTS),
        'silo': Silo(**kw).gaps(bbox, start, end),
        'ozwald_daily': Oz(**kw).gaps(bbox, start, end, cadence='daily', variables=OZWALD_DAILY_VARS),
        'ozwald_8day': Oz(**kw).gaps(bbox, start, end, cadence='8day', variables=OZWALD_8DAY_VARS),
        'slga': Slga(**kw).gaps(bbox, attributes=SOIL_ATTRIBUTES, depths=SOIL_DEPTHS),
        'copdem': Dem(**kw).gaps(bbox),
        'sentinel2': Cube(**kw).gaps(bbox, start, end),
    }


# -- stacking ------------------------------------------------------------------

def align(native: dict[str, xr.Dataset], bbox, *, bilinear: dict[str, tuple[str, ...]] | None = None,
          source_index: tuple[str, ...] = ('smips',)) -> xr.Dataset:
    """Regrid a dict of native datasets onto the Sentinel-2 window of
    ``bbox`` and merge them, prefixing variables with the source name.

    Pure: no network, no stores. ``bilinear`` names, per source, the
    variables to upsample bilinearly instead of replicating (elevation).
    ``source_index`` names the sources whose native row/col maps are kept
    as ``{source}_source_row`` / ``{source}_source_col``.
    """
    bilinear = bilinear or {}
    parts = []
    for name, ds in native.items():
        if str(ds.attrs.get('crs', '')).upper() == 'EPSG:6933':
            out = regrid.regrid(ds, bbox)                      # crop only
        else:
            methods = {v: 'bilinear' for v in bilinear.get(name, ())}
            out = regrid.regrid(ds, bbox, methods=methods, source_index=name in source_index)
        renamed = {v: f'{name}_{v}' for v in out.data_vars}
        parts.append(out.rename(renamed))
    merged = xr.merge(parts, compat='override', combine_attrs='drop')
    merged.attrs = {'crs': regrid.TARGET_CRS, 'transform': parts[0].attrs['transform'],
                    'regrid': {k: v.attrs['regrid'] for k, v in zip(native, parts)}}
    return merged


def statics(bbox, **kw) -> xr.Dataset:
    """Time-invariant layers on the grid: soil (replicated) and terrain
    (elevation bilinear; derivatives computed on native 30 m with a
    buffer, then replicated)."""
    return align({'soil': soil(bbox, **kw), 'terrain': terrain(bbox, **kw)}, bbox,
                 bilinear={'terrain': ('elevation',)}, source_index=())


def frames(bbox, start: date, end: date, chunk_days: int = 64, with_sentinel2: bool = False,
           with_ozwald: bool = True, **kw):
    """Yield one aligned dataset per time chunk of ``[start, end]``: SMIPS
    (with its native index maps), SILO, OzWALD daily and, if asked,
    Sentinel-2. Statics come from :func:`statics` separately so they are
    regridded once, not per chunk."""
    cur = start
    while cur <= end:
        stop = min(end, cur + timedelta(days=chunk_days - 1))
        native = {'smips': smips(bbox, cur, stop, **kw), 'silo': silo(bbox, cur, stop, **kw)}
        if with_ozwald:
            native['ozwald'] = ozwald(bbox, cur, stop, 'daily', **kw)
        if with_sentinel2:
            native['sentinel2'] = sentinel2(bbox, cur, stop, **kw)
        yield align(native, bbox)
        cur = stop + timedelta(days=1)


def mass_balance(stack: xr.Dataset, fine: np.ndarray, native_shape=None, source: str = 'smips'):
    """Mean of a 10 m field per native pixel of ``source``, using the index
    maps kept by :func:`align` -- the check that a downscaled field
    conserves the coarse value it came from."""
    rows = stack[f'{source}_source_row'].values[:, 0]
    cols = stack[f'{source}_source_col'].values[0, :]
    native_shape = native_shape or tuple(stack.attrs['regrid'][source]['native_shape'])
    return regrid.aggregate(fine, rows, cols, native_shape, 'mean')


# -- CLI: build one AOI stack as a zarr --------------------------------------------

def build(bbox, start: date, end: date, out: str, chunk_days: int = 64, with_sentinel2: bool = False,
          verbose: bool = True) -> str:
    """Fill every store for ``bbox`` x ``[start, end]``, audit, and write the
    aligned 10 m stack to ``out`` (zarr), statics first then one time
    chunk at a time."""
    st = statics(bbox)
    st.to_zarr(out, mode='w')
    if verbose:
        print(f'statics: {list(st.data_vars)} -> {out}', flush=True)
    first = True
    for frame in frames(bbox, start, end, chunk_days=chunk_days, with_sentinel2=with_sentinel2):
        dyn = frame.drop_vars([v for v in frame.data_vars if 'time' not in frame[v].dims])
        dyn.to_zarr(out, mode='a', append_dim=None if first else 'time')
        first = False
        if verbose:
            t = frame.time.values
            print(f'  {str(t[0])[:10]}..{str(t[-1])[:10]}: {len(t)} days, {list(dyn.data_vars)}', flush=True)
    reports = gaps(bbox, start, end)
    for name, r in reports.items():
        if verbose:
            print(f'{name}: {r.summary().replace(chr(10), " | ")}', flush=True)
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='Build one AOI stack on the Sentinel-2 grid from the stores.')
    p.add_argument('--bbox', nargs=4, type=float, required=True, metavar=('W', 'S', 'E', 'N'))
    p.add_argument('--start', type=date.fromisoformat, required=True)
    p.add_argument('--end', type=date.fromisoformat, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--chunk-days', type=int, default=64)
    p.add_argument('--sentinel2', action='store_true')
    a = p.parse_args(argv)
    build(a.bbox, a.start, a.end, a.out, chunk_days=a.chunk_days, with_sentinel2=a.sentinel2)


# -- offline tests ------------------------------------------------------------

def test_align_stacks_synthetic_sources():
    import pandas as pd
    from emt.regrid import _synthetic, _SMIPS, _SILO, _SLGA, _COPDEM, _BBOX
    native = {
        'smips': _synthetic(**_SMIPS, bbox=_BBOX, value_fn=lambda r, c: r + c, name='totalbucket',
                            time=pd.date_range('2020-01-01', periods=2)),
        'silo': _synthetic(**_SILO, bbox=_BBOX, value_fn=lambda r, c: 1.0, name='daily_rain',
                           time=pd.date_range('2020-01-01', periods=2)),
        'soil': _synthetic(**_SLGA, bbox=_BBOX, value_fn=lambda r, c: 30.0, name='Clay_5-15cm'),
        'terrain': _synthetic(**_COPDEM, bbox=_BBOX, value_fn=lambda r, c: 300.0 + c, name='elevation'),
    }
    st = align(native, _BBOX, bilinear={'terrain': ('elevation',)})
    fine = st['smips_totalbucket'].values[0]
    back = mass_balance(st, fine)
    shape = tuple(st.attrs['regrid']['smips']['native_shape'])
    rows = st['smips_source_row'].values[:, 0]; cols = st['smips_source_col'].values[0, :]
    full = regrid.coverage(rows, cols, shape) == regrid.coverage(rows, cols, shape).max()
    return (set(st.data_vars) >= {'smips_totalbucket', 'silo_daily_rain', 'soil_Clay_5-15cm',
                                  'terrain_elevation', 'smips_source_row', 'smips_source_col'}
            and st['smips_totalbucket'].dims == ('time', 'y', 'x')
            and st['soil_Clay_5-15cm'].dims == ('y', 'x')
            and st['terrain_elevation'].shape == st['soil_Clay_5-15cm'].shape
            and st.attrs['regrid']['terrain']['methods'] == {'elevation': 'bilinear'}
            and np.array_equal(back[full], native['smips']['totalbucket'].values[0][full]))


def test():
    return all([test_align_stacks_synthetic_sources()])


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        print(test())
