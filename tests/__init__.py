import dask
import dask.array as da
import numpy as np
import xarray as xr


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
    scheduler = CountingScheduler(max_computes)
    return dask.config.set(scheduler=scheduler)


def assert_equal(a, b):
    __tracebackhide__ = True

    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    if isinstance(a, (xr.DataArray, xr.Dataset)) or isinstance(b, (xr.DataArray, xr.Dataset)):
        xr.testing.assert_identical(a, b)
    elif isinstance(a, da.Array) or isinstance(b, da.Array):
        # does some validation of the dask graph
        da.utils.assert_eq(a, b, equal_nan=True)
    else:
        np.testing.assert_allclose(a, b, equal_nan=True)
