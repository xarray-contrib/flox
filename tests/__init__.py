import importlib
from contextlib import contextmanager
from distutils import version

import numpy as np
import pandas as pd
import pytest

pd_types = (pd.Index,)

try:
    import dask
    import dask.array as da

    dask_array_type = da.Array
except ImportError:
    dask_array_type = ()


try:
    import xarray as xr

    xr_types = (xr.DataArray, xr.Dataset)
except ImportError:
    xr_types = ()


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
    return version.LooseVersion(vstring)


has_dask, requires_dask = _importorskip("dask")
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


@contextmanager
def dummy_context():
    yield None


def raise_if_dask_computes(max_computes=0):
    # return a dummy context manager so that this can be used for non-dask objects
    if not has_dask:
        return dummy_context()
    scheduler = CountingScheduler(max_computes)
    return dask.config.set(scheduler=scheduler)


def assert_equal(a, b):
    __tracebackhide__ = True

    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    if isinstance(a, pd_types) or isinstance(b, pd_types):
        pd.testing.assert_index_equal(a, b)
    elif has_xarray and isinstance(a, xr_types) or isinstance(b, xr_types):
        xr.testing.assert_identical(a, b)
    elif has_dask and isinstance(a, dask_array_type) or isinstance(b, dask_array_type):
        # sometimes it's nice to see values and shapes
        # rather than being dropped into some file in dask
        np.testing.assert_allclose(a, b)
        # does some validation of the dask graph
        da.utils.assert_eq(a, b, equal_nan=True)
    else:
        np.testing.assert_allclose(a, b, equal_nan=True)


@pytest.fixture(scope="module", params=["flox", "numpy", "numba"])
def engine(request):
    if request.param == "numba":
        try:
            import numba
        except ImportError:
            pytest.xfail()
    return request.param
