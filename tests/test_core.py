import dask.array as da
import numpy as np
import pytest
from dask.array import from_array
from numpy.testing import assert_array_equal
from numpy_groupies.aggregate_numpy import aggregate

from dask_groupby.core import chunk_reduce, groupby_reduce, reindex_

labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0])


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

    np.testing.assert_array_equal(expected, actual)


def test_chunk_reduce_nd_md():
    array = np.ones((2, 12))
    to_group = np.array([labels] * 2)

    expected = aggregate(to_group.ravel(), array.ravel(), func="sum")
    result = chunk_reduce(array, to_group, func=("sum",))
    actual = reindex_(result["sum"], result["groups"], np.unique(to_group), axis=0)
    np.testing.assert_array_equal(expected, actual)

    array = np.ones((4, 2, 12))
    to_group = np.array([labels] * 2)

    expected = aggregate(to_group.ravel(), array.reshape(4, 24), func="sum", axis=-1)
    result = chunk_reduce(array, to_group, func=("sum",))
    actual = reindex_(result["sum"], result["groups"], np.unique(to_group), axis=-1)
    assert_array_equal(expected, actual)


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
    actual = groupby_reduce(array, to_group, func=("sum",), expected_groups=[0, 1, 2])["sum"]
    assert_array_equal(expected, actual)

    actual = groupby_reduce(array, to_group, func=("sum",), expected_groups=[0, 2, 1])["sum"]
    assert_array_equal(expected, actual[..., [0, 2, 1]])


def test_axis_subset():

    newlabels = np.array([labels[:5], np.flip(labels[:5])])
    to_group = newlabels
    array = np.ones_like(to_group)
    result = chunk_reduce(array, to_group, ("count",), axis=1)
    assert_array_equal(result["count"], [[2, 3], [2, 3]])

    to_group = np.broadcast_to(newlabels, (3, *newlabels.shape))
    array = np.ones_like(to_group)
    result = chunk_reduce(array, to_group, ("count",), axis=1)
    sub = np.array([[1, 1], [1, 1], [0, 2], [1, 1], [1, 1]])
    expected = np.tile(sub, (3, 1, 1))
    assert_array_equal(result["count"], expected)

    result = chunk_reduce(array, to_group, ("count",), axis=2)
    sub = np.array([[2, 3], [2, 3]])
    expected = np.tile(sub, (3, 1, 1))
    assert_array_equal(result["count"], expected)
