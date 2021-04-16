import dask.array as da
import numpy as np
import pytest
from dask.array import from_array
from numpy_groupies.aggregate_numpy import aggregate

from dask_groupby.core import chunk_reduce, groupby_reduce, reindex_

from . import raise_if_dask_computes

labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0])
nan_labels = labels.astype(float)  # copy
nan_labels[:5] = np.nan
labels2d = np.array([labels[:5], np.flip(labels[:5])])


def assert_equal(a, b):
    __tracebackhide__ = True

    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    if isinstance(a, da.Array) or isinstance(b, da.Array):
        # does some validation of the dask graph
        func = da.utils.assert_eq
    else:
        func = np.testing.assert_equal
    func(a, b)


@pytest.mark.parametrize("dask", [False, True])
@pytest.mark.parametrize("expected_groups", [None, [0, 1, 2], np.array([0, 1, 2])])
@pytest.mark.parametrize(
    "func, array, to_group, expected",
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
def test_groupby_reduce(array, to_group, expected, func, expected_groups, dask):
    if dask:
        if expected_groups is None:
            pytest.skip()
        array = da.from_array(array, chunks=(3,) if array.ndim == 1 else (1, 3))
        to_group = da.from_array(to_group, chunks=(3,) if to_group.ndim == 1 else (1, 3))

    result = groupby_reduce(array, to_group, func=func, expected_groups=expected_groups)
    assert_equal(expected, result[func])


def test_numpy_reduce_nd_md():
    array = np.ones((2, 12))
    to_group = np.array([labels] * 2)

    expected = aggregate(to_group.ravel(), array.ravel(), func="sum")
    result = groupby_reduce(array, to_group, func="sum")
    actual = reindex_(result["sum"], result["groups"], np.unique(to_group), axis=0)
    np.testing.assert_equal(expected, actual)

    array = np.ones((4, 2, 12))
    to_group = np.array([labels] * 2)

    expected = aggregate(to_group.ravel(), array.reshape(4, 24), func="sum", axis=-1)
    result = groupby_reduce(array, to_group, func=("sum",))
    actual = reindex_(result["sum"], result["groups"], np.unique(to_group), axis=-1)
    assert_equal(expected, actual)

    array = np.ones((4, 2, 12))
    to_group = np.broadcast_to(np.array([labels] * 2), array.shape)
    expected = aggregate(to_group.ravel(), array.ravel(), func="sum", axis=-1)
    result = groupby_reduce(array, to_group, func=("sum",))
    actual = reindex_(result["sum"], result["groups"], np.unique(to_group), axis=-1)
    assert_equal(expected, actual)


@pytest.mark.parametrize("add_nan", [False, True])
@pytest.mark.parametrize(
    "array, group_chunks",
    [
        (da.ones((12,), (3,)), 3),  # form 1
        (da.ones((12,), (3,)), (4,)),  # form 1
        (da.ones((12,), ((3, 5, 4),)), (2,)),  # form 1
        (da.ones((10, 12), (3, 3)), -1),  # form 3
        (da.ones((10, 12), (3, 3)), 3),  # form 3
    ],
)
def test_groupby_agg_dask(array, group_chunks, add_nan):
    labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0], dtype=float)
    if add_nan:
        labels[:3] = np.nan  # entire block is NaN when group_chunks=3
        labels[-2:] = np.nan

    kwargs = dict(func="sum", expected_groups=[0, 1, 2])

    to_group = from_array(labels, group_chunks)
    expected = groupby_reduce(array.compute(), to_group.compute(), **kwargs)["sum"]
    with raise_if_dask_computes():
        actual = groupby_reduce(array, to_group, **kwargs)["sum"]
    assert_equal(expected, actual)

    with raise_if_dask_computes():
        actual = groupby_reduce(array, to_group, func=("sum",), expected_groups=[0, 2, 1])["sum"]
    assert_equal(expected, actual[..., [0, 2, 1]])


def test_numpy_reduce_axis_subset():

    # TODO: add NaNs
    to_group = labels2d
    array = np.ones_like(to_group)
    result = groupby_reduce(array, to_group, ("count",), axis=1)
    assert_equal(result["count"], [[2, 3], [2, 3]])

    to_group = np.broadcast_to(labels2d, (3, *labels2d.shape))
    array = np.ones_like(to_group)
    result = groupby_reduce(array, to_group, ("count",), axis=1)
    subarr = np.array([[1, 1], [1, 1], [0, 2], [1, 1], [1, 1]])
    expected = np.tile(subarr, (3, 1, 1))
    assert_equal(result["count"], expected)

    result = groupby_reduce(array, to_group, ("count",), axis=2)
    subarr = np.array([[2, 3], [2, 3]])
    expected = np.tile(subarr, (3, 1, 1))
    assert_equal(result["count"], expected)

    result = groupby_reduce(array, to_group, ("count",), axis=(1, 2))
    expected = np.array([[4, 6], [4, 6], [4, 6]])
    assert_equal(result["count"], expected)

    result = groupby_reduce(array, to_group, ("count",), axis=(2, 1))
    assert_equal(result["count"], expected)

    result = groupby_reduce(array, to_group[0, ...], ("count",), axis=(1, 2))
    expected = np.array([[4, 6], [4, 6], [4, 6]])
    assert_equal(result["count"], expected)


def test_dask_reduce_axis_subset():

    to_group = labels2d
    array = np.ones_like(to_group)
    with raise_if_dask_computes():
        result = groupby_reduce(
            da.from_array(array, chunks=(2, 3)),
            da.from_array(to_group, chunks=(2, 2)),
            ("count",),
            axis=1,
            expected_groups=[0, 2],
        )
    assert_equal(result["count"], [[2, 3], [2, 3]])

    to_group = np.broadcast_to(labels2d, (3, *labels2d.shape))
    array = np.ones_like(to_group)
    subarr = np.array([[1, 1], [1, 1], [0, 2], [1, 1], [1, 1]])
    expected = np.tile(subarr, (3, 1, 1))
    with raise_if_dask_computes():
        result = groupby_reduce(
            da.from_array(array, chunks=(1, 2, 3)),
            da.from_array(to_group, chunks=(2, 2, 2)),
            ("count",),
            axis=1,
            expected_groups=[0, 2],
        )
    assert_equal(result["count"], expected)

    subarr = np.array([[2, 3], [2, 3]])
    expected = np.tile(subarr, (3, 1, 1))
    with raise_if_dask_computes():
        result = groupby_reduce(
            da.from_array(array, chunks=(1, 2, 3)),
            da.from_array(to_group, chunks=(2, 2, 2)),
            ("count",),
            axis=2,
            expected_groups=[0, 2],
        )
    assert_equal(result["count"], expected)

    with pytest.raises(NotImplementedError):
        groupby_reduce(
            da.from_array(array, chunks=(1, 2, 3)),
            da.from_array(to_group, chunks=(2, 2, 2)),
            ("count",),
            axis=2,
        )


@pytest.mark.parametrize(
    "axis", [None, (0, 1, 2), (0, 1), (0, 2), (1, 2), 0, 1, 2, (0,), (1,), (2,)]
)
def test_groupby_reduce_axis_subset_against_numpy(axis):
    # tests against the numpy output to make sure dask compute matches
    to_group = np.broadcast_to(labels2d, (3, *labels2d.shape))
    array = np.ones_like(to_group)
    kwargs = dict(func="count", axis=axis, expected_groups=[0, 2])
    with raise_if_dask_computes():
        actual = groupby_reduce(
            da.from_array(array, chunks=(-1, 2, 3)),
            da.from_array(to_group, chunks=(-1, 2, 2)),
            **kwargs,
        )["count"]
    expected = groupby_reduce(array, to_group, **kwargs)["count"]
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

    # test when entire to_group  are NaNs
    to_group = np.full((3, 5, 2), fill_value=np.nan)
    array = np.ones_like(to_group)

    # along an axis; requires expected_group
    # TODO: this should check for fill_value
    result = groupby_reduce(
        _maybe_chunk(array),
        _maybe_chunk(to_group),
        "count",
        expected_groups=groups,
        axis=axis,
    )
    assert_equal(result["count"], np.zeros(expected_shape))

    # now when subsets are NaN
    labels = np.array([0, 0, 1, 1, 1], dtype=float)
    labels2d = np.array([labels[:5], np.flip(labels[:5])])
    labels2d[:] = np.nan
    to_group = np.broadcast_to(labels2d, (3, *labels2d.shape))


def test_reindex():
    array = np.array([1, 2])
    groups = np.array(["a", "b"])
    expected_groups = ["a", "b", "c"]
    fill_value = 0
    result = reindex_(array, groups, expected_groups, fill_value, axis=-1)
    assert_equal(result, np.array([1, 2, 0]))
