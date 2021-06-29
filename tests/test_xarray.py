import dask
import numpy as np
import pytest
import xarray as xr

from dask_groupby.xarray import xarray_groupby_reduce, xarray_reduce

from . import assert_equal, raise_if_dask_computes

dask.config.set(scheduler="sync")


def test_xarray_groupby_reduce():
    arr = np.ones((4, 12))

    labels = np.array(["a", "a", "c", "c", "c", "b", "b", "c", "c", "b", "b", "f"])
    labels = np.array(labels)
    labels2 = np.array([1, 2, 2, 1])

    da = xr.DataArray(
        arr, dims=("x", "y"), coords={"labels2": ("x", labels2), "labels": ("y", labels)}
    ).expand_dims(z=4)

    grouped = da.groupby("labels")
    expected = grouped.mean()
    actual = xarray_groupby_reduce(grouped, "mean")
    assert_equal(expected, actual)

    actual = xarray_groupby_reduce(da.transpose("y", ...).groupby("labels"), "mean")
    assert_equal(expected, actual)

    # TODO: fails because of stacking
    # grouped = da.groupby("labels2")
    # expected = grouped.mean()
    # actual = xarray_groupby_reduce(grouped, "mean")
    # assert_equal(expected, actual)


def test_xarray_reduce_multiple_groupers():
    arr = np.ones((4, 12))

    labels = np.array(["a", "a", "c", "c", "c", "b", "b", "c", "c", "b", "b", "f"])
    labels = np.array(labels)
    labels2 = np.array([1, 2, 2, 1])

    da = xr.DataArray(
        arr, dims=("x", "y"), coords={"labels2": ("x", labels2), "labels": ("y", labels)}
    ).expand_dims(z=4)

    expected = xr.DataArray(
        [[4, 4], [10, 10], [8, 8], [2, 2]],
        dims=("labels", "labels2"),
        coords={"labels": ["a", "c", "b", "f"], "labels2": [1, 2]},
    ).expand_dims(z=4)

    actual = xarray_reduce(da, da.labels, da.labels2, func="count")
    xr.testing.assert_identical(expected, actual)

    actual = xarray_reduce(da, "labels", "labels2", func="count", fill_value=0)
    xr.testing.assert_identical(expected, actual)

    with raise_if_dask_computes():
        actual = xarray_reduce(da.chunk({"x": 2, "z": 1}), da.labels, da.labels2, func="count")
    xr.testing.assert_identical(expected, actual)

    with pytest.raises(ValueError):
        actual = xarray_reduce(da.chunk({"x": 2, "z": 1}), "labels", "labels2", func="count")
    # xr.testing.assert_identical(expected, actual)
