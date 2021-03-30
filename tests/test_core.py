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


@pytest.mark.parametrize("reduce_", [chunk_reduce, groupby_reduce])
@pytest.mark.parametrize("expected_groups", [None, [0, 1, 2], np.array([0, 1, 2])])
@pytest.mark.parametrize(
    "array, to_group, expected",
    [
        (np.ones((12,)), labels, [3, 4, 5]),  # form 1
        (np.ones((12,)), nan_labels, [1, 4, 2]),  # form 1
        (np.ones((2, 12)), labels, [[3, 4, 5], [3, 4, 5]]),  # form 3
        (np.ones((2, 12)), nan_labels, [[1, 4, 2], [1, 4, 2]]),  # form 3
        # (np.ones((12,)), np.array([labels, labels])),  # form 4
    ],
)
def test_chunk_reduce(array, to_group, expected, reduce_, expected_groups):
    result = reduce_(array, to_group, func=("sum",), expected_groups=expected_groups)
    assert_equal(expected, result["sum"])


@pytest.mark.parametrize("reduce_", [chunk_reduce, groupby_reduce])
def test_chunk_reduce_nd_md(reduce_):
    array = np.ones((2, 12))
    to_group = np.array([labels] * 2)

    expected = aggregate(to_group.ravel(), array.ravel(), func="sum")
    result = reduce_(array, to_group, func="sum")
    actual = reindex_(result["sum"], result["groups"], np.unique(to_group), axis=0)
    np.testing.assert_equal(expected, actual)

    array = np.ones((4, 2, 12))
    to_group = np.array([labels] * 2)

    expected = aggregate(to_group.ravel(), array.reshape(4, 24), func="sum", axis=-1)
    result = reduce_(array, to_group, func=("sum",))
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
    expected = chunk_reduce(array.compute(), to_group.compute(), **kwargs)["sum"]
    with raise_if_dask_computes():
        actual = groupby_reduce(array, to_group, **kwargs)["sum"]
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


def test_groupby_reduce_nans():
    # test when entire to_group  are NaNs
    to_group = np.full((3, 5, 2), fill_value=np.nan)
    array = np.ones_like(to_group)

    result = chunk_reduce(
        da.from_array(array, chunks=(2, 2, 3)),
        da.from_array(to_group, chunks=(2, 2, 2)),
        ("count",),
        expected_groups=[0, 1, 2],
        axis=2,
    )
    assert_equal(result["count"], np.zeros(array.shape[:-1] + (3,)))

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
