import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.array import from_array
from numpy_groupies.aggregate_numpy import aggregate

from dask_groupby.core import groupby_reduce, reindex_, xarray_groupby_reduce, xarray_reduce

from . import raise_if_dask_computes

labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0])
nan_labels = labels.astype(float)  # copy
nan_labels[:5] = np.nan
labels2d = np.array([labels[:5], np.flip(labels[:5])])

dask.config.set(scheduler="sync")


def assert_equal(a, b):
    __tracebackhide__ = True

    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    if isinstance(a, da.Array) or isinstance(b, da.Array):
        # does some validation of the dask graph
        try:
            da.utils.assert_eq(a, b)
        except AssertionError:
            # dask doesn't consider nans in the same place to be equal
            # from xarray.core.duck_array_ops.array_equiv
            flag_array = (a == b) | (np.isnan(a) & np.isnan(b))
            assert bool(flag_array.all())
    elif isinstance(a, xr.DataArray) | isinstance(b, xr.DataArray):
        xr.testing.assert_identical(a, b)
    else:
        np.testing.assert_equal(a, b)


# TODO: Add max,argmax here
@pytest.mark.parametrize("chunk, split_out", [(False, 1), (True, 1), (True, 2), (True, 3)])
@pytest.mark.parametrize("expected_groups", [None, [0, 1, 2], np.array([0, 1, 2])])
@pytest.mark.parametrize(
    "func, array, by, expected",
    [
        ("sum", np.ones((12,)), labels, [3, 4, 5]),  # form 1
        ("sum", np.ones((12,)), nan_labels, [1, 4, 2]),  # form 1
        ("sum", np.ones((2, 12)), labels, [[3, 4, 5], [3, 4, 5]]),  # form 3
        ("sum", np.ones((2, 12)), nan_labels, [[1, 4, 2], [1, 4, 2]]),  # form 3
        ("sum", np.ones((2, 12)), np.array([labels, labels]), [6, 8, 10]),  # form 1 after reshape
        (
            "sum",
            np.ones((2, 12)),
            np.array([nan_labels, nan_labels]),
            [2, 8, 4],
        ),  # form 1 after reshape
        # (np.ones((12,)), np.array([labels, labels])),  # form 4
        ("nanmean", np.ones((12,)), labels, [1, 1, 1]),  # form 1
        ("nanmean", np.ones((12,)), nan_labels, [1, 1, 1]),  # form 1
        ("nanmean", np.ones((2, 12)), labels, [[1, 1, 1], [1, 1, 1]]),  # form 3
        ("nanmean", np.ones((2, 12)), nan_labels, [[1, 1, 1], [1, 1, 1]]),  # form 3
        (
            "nanmean",
            np.ones((2, 12)),
            np.array([labels, labels]),
            [1, 1, 1],
        ),  # form 1 after reshape
        (
            "nanmean",
            np.ones((2, 12)),
            np.array([nan_labels, nan_labels]),
            [1, 1, 1],
        ),  # form 1 after reshape
        # (np.ones((12,)), np.array([labels, labels])),  # form 4
    ],
)
def test_groupby_reduce(array, by, expected, func, expected_groups, chunk, split_out):
    if chunk:
        if expected_groups is None:
            pytest.skip()
        array = da.from_array(array, chunks=(3,) if array.ndim == 1 else (1, 3))
        by = da.from_array(by, chunks=(3,) if by.ndim == 1 else (1, 3))

    result, _ = groupby_reduce(
        array,
        by,
        func=func,
        expected_groups=expected_groups,
        fill_value=123,
        split_out=split_out,
    )
    assert_equal(expected, result)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_groupby_reduce_preserves_dtype(dtype):
    array = np.ones((2, 12), dtype=dtype)
    by = np.array([labels] * 2)
    result, _ = groupby_reduce(from_array(array, chunks=(-1, 4)), by, func="sum")
    assert result.dtype == array.dtype


def test_numpy_reduce_nd_md():
    array = np.ones((2, 12))
    by = np.array([labels] * 2)

    expected = aggregate(by.ravel(), array.ravel(), func="sum")
    result, groups = groupby_reduce(array, by, func="sum", fill_value=123)
    actual = reindex_(result, groups, np.unique(by), axis=0, fill_value=0)
    np.testing.assert_equal(expected, actual)

    array = np.ones((4, 2, 12))
    by = np.array([labels] * 2)

    expected = aggregate(by.ravel(), array.reshape(4, 24), func="sum", axis=-1, fill_value=0)
    result, groups = groupby_reduce(array, by, func="sum")
    actual = reindex_(result, groups, np.unique(by), axis=-1, fill_value=0)
    assert_equal(expected, actual)

    array = np.ones((4, 2, 12))
    by = np.broadcast_to(np.array([labels] * 2), array.shape)
    expected = aggregate(by.ravel(), array.ravel(), func="sum", axis=-1)
    result, groups = groupby_reduce(array, by, func="sum")
    actual = reindex_(result, groups, np.unique(by), axis=-1, fill_value=0)
    assert_equal(expected, actual)


@pytest.mark.parametrize(
    "func",
    (
        "sum",
        "count",
        "prod",
        "mean",
        "std",
        "var",
        "max",
        "min",
        "argmax",
        "argmin",
        # "first",
        # "last",
    ),
)
@pytest.mark.parametrize("add_nan", [False, True])
@pytest.mark.parametrize("dtype", (float,))
@pytest.mark.parametrize(
    "array, group_chunks",
    [
        (da.ones((12,), (3,)), 3),  # form 1
        (da.ones((12,), (3,)), (4,)),  # form 1, chunks not aligned
        (da.ones((12,), ((3, 5, 4),)), (2,)),  # form 1
        (da.ones((10, 12), (3, 3)), -1),  # form 3
        (da.ones((10, 12), (3, 3)), 3),  # form 3
    ],
)
def test_groupby_agg_dask(func, array, group_chunks, add_nan, dtype):
    """ Tests groupby_reduce with dask arrays against groupby_reduce with numpy arrays"""

    array = array.astype(dtype)

    if func in ["first", "last"]:
        pytest.skip()

    labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0])
    if add_nan:
        labels = labels.astype(float)
        labels[:3] = np.nan  # entire block is NaN when group_chunks=3
        labels[-2:] = np.nan

    kwargs = dict(func=func, expected_groups=[0, 1, 2], fill_value=123)

    by = from_array(labels, group_chunks)
    expected, _ = groupby_reduce(array.compute(), by.compute(), **kwargs)
    with raise_if_dask_computes():
        actual, _ = groupby_reduce(array, by, **kwargs)
    assert_equal(expected, actual)

    kwargs["expected_groups"] = [0, 2, 1]
    with raise_if_dask_computes():
        actual, groups = groupby_reduce(array, by, **kwargs)
    assert_equal(groups, [0, 2, 1])
    assert_equal(expected, actual[..., [0, 2, 1]])


def test_numpy_reduce_axis_subset():
    # TODO: add NaNs
    by = labels2d
    array = np.ones_like(by)
    result, _ = groupby_reduce(array, by, "count", axis=1)
    assert_equal(result, [[2, 3], [2, 3]])

    by = np.broadcast_to(labels2d, (3, *labels2d.shape))
    array = np.ones_like(by)
    result, _ = groupby_reduce(array, by, "count", axis=1)
    subarr = np.array([[1, 1], [1, 1], [0, 2], [1, 1], [1, 1]])
    expected = np.tile(subarr, (3, 1, 1))
    assert_equal(result, expected)

    result, _ = groupby_reduce(array, by, "count", axis=2)
    subarr = np.array([[2, 3], [2, 3]])
    expected = np.tile(subarr, (3, 1, 1))
    assert_equal(result, expected)

    result, _ = groupby_reduce(array, by, "count", axis=(1, 2))
    expected = np.array([[4, 6], [4, 6], [4, 6]])
    assert_equal(result, expected)

    result, _ = groupby_reduce(array, by, "count", axis=(2, 1))
    assert_equal(result, expected)

    result, _ = groupby_reduce(array, by[0, ...], "count", axis=(1, 2))
    expected = np.array([[4, 6], [4, 6], [4, 6]])
    assert_equal(result, expected)


def test_dask_reduce_axis_subset():

    by = labels2d
    array = np.ones_like(by)
    with raise_if_dask_computes():
        result, _ = groupby_reduce(
            da.from_array(array, chunks=(2, 3)),
            da.from_array(by, chunks=(2, 2)),
            "count",
            axis=1,
            expected_groups=[0, 2],
        )
    assert_equal(result, [[2, 3], [2, 3]])

    by = np.broadcast_to(labels2d, (3, *labels2d.shape))
    array = np.ones_like(by)
    subarr = np.array([[1, 1], [1, 1], [123, 2], [1, 1], [1, 1]])
    expected = np.tile(subarr, (3, 1, 1))
    with raise_if_dask_computes():
        result, _ = groupby_reduce(
            da.from_array(array, chunks=(1, 2, 3)),
            da.from_array(by, chunks=(2, 2, 2)),
            "count",
            axis=1,
            expected_groups=[0, 2],
            fill_value=123,
        )
    assert_equal(result, expected)

    subarr = np.array([[2, 3], [2, 3]])
    expected = np.tile(subarr, (3, 1, 1))
    with raise_if_dask_computes():
        result, _ = groupby_reduce(
            da.from_array(array, chunks=(1, 2, 3)),
            da.from_array(by, chunks=(2, 2, 2)),
            "count",
            axis=2,
            expected_groups=[0, 2],
        )
    assert_equal(result, expected)

    with pytest.raises(NotImplementedError):
        groupby_reduce(
            da.from_array(array, chunks=(1, 3, 2)),
            da.from_array(by, chunks=(2, 2, 2)),
            "count",
            axis=2,
        )


@pytest.mark.parametrize(
    "axis", [None, (0, 1, 2), (0, 1), (0, 2), (1, 2), 0, 1, 2, (0,), (1,), (2,)]
)
def test_groupby_reduce_axis_subset_against_numpy(axis):
    # tests against the numpy output to make sure dask compute matches
    by = np.broadcast_to(labels2d, (3, *labels2d.shape))
    array = np.ones_like(by)
    kwargs = dict(func="count", axis=axis, expected_groups=[0, 2], fill_value=123)
    with raise_if_dask_computes():
        actual, _ = groupby_reduce(
            da.from_array(array, chunks=(-1, 2, 3)),
            da.from_array(by, chunks=(-1, 2, 2)),
            **kwargs,
        )
    expected, _ = groupby_reduce(array, by, **kwargs)
    assert_equal(actual, expected)


@pytest.mark.parametrize("chunks", [None, (2, 2, 3)])
@pytest.mark.parametrize(
    "axis, groups, expected_shape",
    [
        (2, [0, 1, 2], (3, 5, 3)),
        (None, [0, 1, 2], (3,)),  # global reduction; 0 shaped group axis
        (None, [0], (1,)),  # global reduction; 0 shaped group axis; 1 group
    ],
)
def test_groupby_reduce_nans(chunks, axis, groups, expected_shape):
    def _maybe_chunk(arr):
        if chunks:
            return da.from_array(arr, chunks=chunks)
        else:
            return arr

    # test when entire by  are NaNs
    by = np.full((3, 5, 2), fill_value=np.nan)
    array = np.ones_like(by)

    # along an axis; requires expected_group
    # TODO: this should check for fill_value
    result, _ = groupby_reduce(
        _maybe_chunk(array),
        _maybe_chunk(by),
        "count",
        expected_groups=groups,
        axis=axis,
        fill_value=0,
    )
    assert_equal(result, np.zeros(expected_shape))

    # now when subsets are NaN
    # labels = np.array([0, 0, 1, 1, 1], dtype=float)
    # labels2d = np.array([labels[:5], np.flip(labels[:5])])
    # labels2d[0, :5] = np.nan
    # labels2d[1, 5:] = np.nan
    # by = np.broadcast_to(labels2d, (3, *labels2d.shape))


def test_groupby_all_nan_blocks():
    labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0])
    nan_labels = labels.astype(float)  # copy
    nan_labels[:5] = np.nan

    array, by, expected = (
        np.ones((2, 12)),
        np.array([nan_labels, nan_labels[::-1]]),
        [2, 8, 4],
    )

    actual, _ = groupby_reduce(
        da.from_array(array, chunks=(1, 3)),
        da.from_array(by, chunks=(1, 3)),
        func="sum",
        expected_groups=None,
    )
    assert_equal(actual, expected)


def test_reindex():
    array = np.array([1, 2])
    groups = np.array(["a", "b"])
    expected_groups = ["a", "b", "c"]
    fill_value = 0
    result = reindex_(array, groups, expected_groups, fill_value, axis=-1)
    assert_equal(result, np.array([1, 2, 0]))


@pytest.mark.xfail
def test_bad_npg_behaviour():
    labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0], dtype=int)
    # fmt: off
    array = np.array([[1] * 12, [1] * 12])
    # fmt: on
    assert_equal(aggregate(labels, array, axis=-1, func="argmax"), np.array([[0, 5, 2], [0, 5, 2]]))

    assert (
        aggregate(
            np.array([0, 1, 2, 0, 1, 2]), np.array([-np.inf, 0, 0, -np.inf, 0, 0]), func="max"
        )[0]
        == -np.inf
    )


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
