import dask
import numpy as np
import pytest
import xarray as xr

from dask_groupby.xarray import (
    _get_optimal_chunks_for_groups,
    resample_reduce,
    xarray_groupby_reduce,
    xarray_reduce,
)

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


def test_xarray_reduce_single_grouper():

    ds = xr.tutorial.open_dataset("rasm", chunks={"time": 4})
    actual = xarray_reduce(ds.Tair, ds.time.dt.month, func="mean")
    expected = ds.Tair.groupby("time.month").mean().transpose("x", "y", "month")
    xr.testing.assert_allclose(actual, expected.transpose(*actual.dims))


@pytest.mark.parametrize("isdask", [True, False])
@pytest.mark.parametrize("chunklen", [27, 4 * 31 + 1, 4 * 31 + 20])
def test_xarray_resample(chunklen, isdask):
    ds = xr.tutorial.open_dataset("air_temperature", chunks={"time": chunklen}).air
    if not isdask:
        ds = ds.compute()
    resampler = ds.resample(time="M")
    actual = resample_reduce(resampler, "mean")
    expected = resampler.mean()
    xr.testing.assert_allclose(actual, expected.transpose(*actual.dims))


@pytest.mark.parametrize(
    "inchunks, expected",
    [
        [(1,) * 10, (3, 2, 2, 3)],
        [(2,) * 5, (3, 2, 2, 3)],
        [(4, 4, 2), (3, 4, 3)],
        [(5, 5), (5, 5)],
        [(6, 4), (5, 5)],
        [(7, 3), (7, 3)],
        [(8, 2), (7, 3)],
        [(9, 1), (10,)],
    ],
)
def test_optimal_rechunking(inchunks, expected):
    labels = np.array([1, 1, 1, 2, 2, 3, 3, 5, 5, 5])
    assert _get_optimal_chunks_for_groups(inchunks, labels) == expected
