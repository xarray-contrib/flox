import dask.array as da
import numpy as np
import pytest
from dask.array import from_array
from numpy_groupies.aggregate_numpy import aggregate

from dask_groupby.core import chunk_reduce, groupby_reduce, reindex_

from . import raise_if_dask_computes

labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0])
labels2d = np.array([labels[:5], np.flip(labels[:5])])


def assert_equal(a, b):
    if isinstance(a, da.Array) or isinstance(b, da.Array):
        # does some validation of the dask graph
        func = da.utils.assert_eq
    else:
        func = np.testing.assert_equal
    func(a, b)


@pytest.mark.parametrize(
    "array, to_group, size, axis",
    [
        (np.ones((12,)), labels, None, -1),  # form 1
        (np.ones((2, 12)), labels, None, -1)  # form 3
        # (np.ones((12,)), np.array([labels, labels]), (3, 3)),  # form 4
    ],
)
def test_chunk_reduce(array, to_group, size, axis):
    expected = aggregate(to_group, array, func="sum", size=size, axis=axis)

    result = chunk_reduce(array, to_group, func=("sum",))
    actual = reindex_(result["sum"], result["groups"], np.unique(to_group), axis=axis)
    assert_equal(expected, actual)


def test_chunk_reduce_nd_md():
    array = np.ones((2, 12))
    to_group = np.array([labels] * 2)

    expected = aggregate(to_group.ravel(), array.ravel(), func="sum")
    result = chunk_reduce(array, to_group, func=("sum",))
    actual = reindex_(result["sum"], result["groups"], np.unique(to_group), axis=0)
    np.testing.assert_equal(expected, actual)

    array = np.ones((4, 2, 12))
    to_group = np.array([labels] * 2)

    expected = aggregate(to_group.ravel(), array.reshape(4, 24), func="sum", axis=-1)
    result = chunk_reduce(array, to_group, func=("sum",))
    actual = reindex_(result["sum"], result["groups"], np.unique(to_group), axis=-1)
    assert_equal(expected, actual)


@pytest.mark.parametrize(
    "array, to_group",
    [
        (da.ones((12,), (3,)), from_array(labels, 3)),  # form 1
        (da.ones((12,), (3,)), from_array(labels, (4,))),  # form 1
        (da.ones((12,), ((3, 5, 4),)), from_array(labels, (2,))),  # form 1
        (da.ones((10, 12), (3, 3)), from_array(labels, -1)),  # form 3
    ],
)
def test_groupby_agg(array, to_group):
    expected = aggregate(to_group.compute(), array.compute(), func="sum", axis=-1)
    with raise_if_dask_computes():
        actual = groupby_reduce(array, to_group, func=("sum",), expected_groups=[0, 1, 2])["sum"]
    assert_equal(expected, actual)

    with raise_if_dask_computes():
        actual = groupby_reduce(array, to_group, func=("sum",), expected_groups=[0, 2, 1])["sum"]
    assert_equal(expected, actual[..., [0, 2, 1]])


def test_chunk_reduce_axis_subset():

    to_group = labels2d
    array = np.ones_like(to_group)
    result = chunk_reduce(array, to_group, ("count",), axis=1)
    assert_equal(result["count"], [[2, 3], [2, 3]])

    to_group = np.broadcast_to(labels2d, (3, *labels2d.shape))
    array = np.ones_like(to_group)
    result = chunk_reduce(array, to_group, ("count",), axis=1)
    subarr = np.array([[1, 1], [1, 1], [0, 2], [1, 1], [1, 1]])
    expected = np.tile(subarr, (3, 1, 1))
    assert_equal(result["count"], expected)

    result = chunk_reduce(array, to_group, ("count",), axis=2)
    subarr = np.array([[2, 3], [2, 3]])
    expected = np.tile(subarr, (3, 1, 1))
    assert_equal(result["count"], expected)


def test_groupby_reduce_axis_subset():

    to_group = labels2d
    array = np.ones_like(to_group)
    with raise_if_dask_computes():
        result = groupby_reduce(
            da.from_array(array, chunks=(2, 3)),
            da.from_array(to_group, chunks=(2, 2)),
            ("count",),
            axis=1,
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
        )
    assert_equal(result["count"], expected)
