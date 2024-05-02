import importlib
from contextlib import nullcontext

import numpy as np
import packaging.version
import pandas as pd
import pytest

pd_types = (pd.Index,)

try:
    import dask
    import dask.array as da

    dask_array_type = da.Array
except ImportError:
    dask_array_type = ()  # type: ignore[assignment, misc]


try:
    import xarray as xr

    xr_types = (xr.DataArray, xr.Dataset)
except ImportError:
    xr_types = ()  # type: ignore[assignment]


def _importorskip(modname, minversion=None):
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            if LooseVersion(mod.__version__) < LooseVersion(minversion):
                raise ImportError("Minimum version not satisfied")
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


def LooseVersion(vstring):
    # Our development version is something like '0.10.9+aac7bfc'
    # This function just ignored the git commit id.
    vstring = vstring.split("+")[0]
    return packaging.version.Version(vstring)


has_cftime, requires_cftime = _importorskip("cftime")
has_cubed, requires_cubed = _importorskip("cubed")
has_dask, requires_dask = _importorskip("dask")
has_numba, requires_numba = _importorskip("numba")
has_numbagg, requires_numbagg = _importorskip("numbagg")
has_scipy, requires_scipy = _importorskip("scipy")
has_xarray, requires_xarray = _importorskip("xarray")


class CountingScheduler:
    """Simple dask scheduler counting the number of computes.

    Reference: https://stackoverflow.com/questions/53289286/"""

    def __init__(self, max_computes=0):
        self.total_computes = 0
        self.max_computes = max_computes

    def __call__(self, dsk, keys, **kwargs):
        self.total_computes += 1
        if self.total_computes > self.max_computes:
            raise RuntimeError(
                "Too many computes. Total: %d > max: %d." % (self.total_computes, self.max_computes)
            )
        return dask.get(dsk, keys, **kwargs)


def raise_if_dask_computes(max_computes=0):
    # return a dummy context manager so that this can be used for non-dask objects
    if not has_dask:
        return nullcontext()
    scheduler = CountingScheduler(max_computes)
    return dask.config.set(scheduler=scheduler)


def assert_equal(a, b, tolerance=None):
    __tracebackhide__ = True

    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)

    if isinstance(a, pd_types) or isinstance(b, pd_types):
        pd.testing.assert_index_equal(a, b)
        return
    if has_xarray and isinstance(a, xr_types) or isinstance(b, xr_types):
        xr.testing.assert_identical(a, b)
        return

    if tolerance is None and (
        np.issubdtype(a.dtype, np.float64) | np.issubdtype(b.dtype, np.float64)
    ):
        tolerance = {"atol": 1e-18, "rtol": 1e-15}
    else:
        tolerance = {}

    if has_dask and isinstance(a, dask_array_type) or isinstance(b, dask_array_type):
        # sometimes it's nice to see values and shapes
        # rather than being dropped into some file in dask
        np.testing.assert_allclose(a, b, **tolerance)
        # does some validation of the dask graph
        da.utils.assert_eq(a, b, equal_nan=True)
    else:
        if a.dtype != b.dtype:
            raise AssertionError(f"a and b have different dtypes: (a: {a.dtype}, b: {b.dtype})")

        np.testing.assert_allclose(a, b, equal_nan=True, **tolerance)


def assert_equal_tuple(a, b):
    """assert_equal for .blocks indexing tuples"""
    assert len(a) == len(b)

    for a_, b_ in zip(a, b):
        assert type(a_) == type(b_)
        if isinstance(a_, np.ndarray):
            np.testing.assert_array_equal(a_, b_)
        else:
            assert a_ == b_


SCIPY_STATS_FUNCS = ("mode", "nanmode")
BLOCKWISE_FUNCS = ("median", "nanmedian", "quantile", "nanquantile") + SCIPY_STATS_FUNCS
ALL_FUNCS = (
    "sum",
    "nansum",
    "argmax",
    "nanfirst",
    "nanargmax",
    "prod",
    "nanprod",
    "mean",
    "nanmean",
    "var",
    "nanvar",
    "std",
    "nanstd",
    "max",
    "nanmax",
    "min",
    "nanmin",
    "argmin",
    "nanargmin",
    "any",
    "all",
    "nanlast",
    "median",
    "nanmedian",
    "quantile",
    "nanquantile",
) + tuple(pytest.param(func, marks=requires_scipy) for func in SCIPY_STATS_FUNCS)
