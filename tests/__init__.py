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
    import sparse

    sparse_array_type = sparse.COO
except ImportError:
    sparse_array_type = ()


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
has_sparse, requires_sparse = _importorskip("sparse")
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
            raise RuntimeError(f"Too many computes. Total: {self.total_computes} > max: {self.max_computes}.")
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

    if tolerance is None:
        if np.issubdtype(a.dtype, np.float64) | np.issubdtype(b.dtype, np.float64):
            tolerance = {"atol": 1e-18, "rtol": 1e-15}
        else:
            tolerance = {}

    # Always run the numpy comparison first, so that we get nice error messages with dask.
    # sometimes it's nice to see values and shapes
    # rather than being dropped into some file in dask
    if a.dtype != b.dtype:
        raise AssertionError(f"a and b have different dtypes: (a: {a.dtype}, b: {b.dtype})")

    if has_dask:
        a_eager = a.compute() if isinstance(a, dask_array_type) else a
        b_eager = b.compute() if isinstance(b, dask_array_type) else b
    else:
        a_eager, b_eager = a, b

    if has_sparse:
        one_is_sparse = isinstance(a_eager, sparse_array_type) or isinstance(b_eager, sparse_array_type)
        a_eager = a_eager.todense() if isinstance(a_eager, sparse_array_type) else a_eager
        b_eager = b_eager.todense() if isinstance(b_eager, sparse_array_type) else b_eager
    else:
        one_is_sparse = False

    if a.dtype.kind in "SUMmO":
        np.testing.assert_equal(a_eager, b_eager)
    else:
        np.testing.assert_allclose(a_eager, b_eager, equal_nan=True, **tolerance)

    if has_dask and isinstance(a, dask_array_type) or isinstance(b, dask_array_type):
        # does some validation of the dask graph
        dask_assert_eq(a, b, equal_nan=True, check_type=not one_is_sparse)


def assert_equal_tuple(a, b):
    """assert_equal for .blocks indexing tuples"""
    assert len(a) == len(b)

    for a_, b_ in zip(a, b):
        assert type(a_) is type(b_)
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
) + tuple(SCIPY_STATS_FUNCS)


def dask_assert_eq(
    a,
    b,
    check_shape=True,
    check_graph=True,
    check_meta=True,
    check_chunks=True,
    check_ndim=True,
    check_type=True,
    check_dtype=True,
    equal_nan=True,
    scheduler="sync",
    **kwargs,
):
    """dask.array.utils.assert_eq modified to skip value checks. Their code is buggy for some dtypes.
    We just check values through numpy and care about validating the graph in this function."""
    from dask.array.utils import _get_dt_meta_computed

    a_original = a
    b_original = b

    if isinstance(a, list | int | float):
        a = np.array(a)
    if isinstance(b, list | int | float):
        b = np.array(b)

    a, adt, a_meta, a_computed = _get_dt_meta_computed(
        a,
        check_shape=check_shape,
        check_graph=check_graph,
        check_chunks=check_chunks,
        check_ndim=check_ndim,
        scheduler=scheduler,
    )
    b, bdt, b_meta, b_computed = _get_dt_meta_computed(
        b,
        check_shape=check_shape,
        check_graph=check_graph,
        check_chunks=check_chunks,
        check_ndim=check_ndim,
        scheduler=scheduler,
    )

    if check_type:
        _a = a if a.shape else a.item()
        _b = b if b.shape else b.item()
        assert type(_a) is type(_b), f"a and b have different types (a: {type(_a)}, b: {type(_b)})"
    if check_meta:
        if hasattr(a, "_meta") and hasattr(b, "_meta"):
            dask_assert_eq(a._meta, b._meta)
        if hasattr(a_original, "_meta"):
            msg = (
                f"compute()-ing 'a' changes its number of dimensions "
                f"(before: {a_original._meta.ndim}, after: {a.ndim})"
            )
            assert a_original._meta.ndim == a.ndim, msg
            if a_meta is not None:
                msg = (
                    f"compute()-ing 'a' changes its type "
                    f"(before: {type(a_original._meta)}, after: {type(a_meta)})"
                )
                assert type(a_original._meta) is type(a_meta), msg
                if not (np.isscalar(a_meta) or np.isscalar(a_computed)):
                    msg = (
                        f"compute()-ing 'a' results in a different type than implied by its metadata "
                        f"(meta: {type(a_meta)}, computed: {type(a_computed)})"
                    )
                    assert type(a_meta) is type(a_computed), msg
        if hasattr(b_original, "_meta"):
            msg = (
                f"compute()-ing 'b' changes its number of dimensions "
                f"(before: {b_original._meta.ndim}, after: {b.ndim})"
            )
            assert b_original._meta.ndim == b.ndim, msg
            if b_meta is not None:
                msg = (
                    f"compute()-ing 'b' changes its type "
                    f"(before: {type(b_original._meta)}, after: {type(b_meta)})"
                )
                assert type(b_original._meta) is type(b_meta), msg
                if not (np.isscalar(b_meta) or np.isscalar(b_computed)):
                    msg = (
                        f"compute()-ing 'b' results in a different type than implied by its metadata "
                        f"(meta: {type(b_meta)}, computed: {type(b_computed)})"
                    )
                    assert type(b_meta) is type(b_computed), msg
