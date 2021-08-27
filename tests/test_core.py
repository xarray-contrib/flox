import dask
import dask.array as da
import numpy as np
import pytest
from dask.array import from_array
from numpy_groupies.aggregate_numpy import aggregate

from dask_groupby.core import groupby_reduce, reindex_

from . import assert_equal, raise_if_dask_computes

labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0])
nan_labels = labels.astype(float)  # copy
nan_labels[:5] = np.nan
labels2d = np.array([labels[:5], np.flip(labels[:5])])

dask.config.set(scheduler="sync")


def test_alignment_error():
    da = np.ones((12,))
    labels = np.ones((5,))

    with pytest.raises(ValueError):
        groupby_reduce(da, labels, func="mean")


@pytest.mark.parametrize("dtype", (float, int))
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
        ("sum", np.ones((2, 12)), np.array([nan_labels, nan_labels]), [2, 8, 4]),
        # (np.ones((12,)), np.array([labels, labels])),  # form 4
        ("count", np.ones((12,)), labels, [3, 4, 5]),  # form 1
        ("count", np.ones((12,)), nan_labels, [1, 4, 2]),  # form 1
        ("count", np.ones((2, 12)), labels, [[3, 4, 5], [3, 4, 5]]),  # form 3
        ("count", np.ones((2, 12)), nan_labels, [[1, 4, 2], [1, 4, 2]]),  # form 3
        ("count", np.ones((2, 12)), np.array([labels, labels]), [6, 8, 10]),  # form 1 after reshape
        ("count", np.ones((2, 12)), np.array([nan_labels, nan_labels]), [2, 8, 4]),
        ("nanmean", np.ones((12,)), labels, [1, 1, 1]),  # form 1
        ("nanmean", np.ones((12,)), nan_labels, [1, 1, 1]),  # form 1
        ("nanmean", np.ones((2, 12)), labels, [[1, 1, 1], [1, 1, 1]]),  # form 3
        ("nanmean", np.ones((2, 12)), nan_labels, [[1, 1, 1], [1, 1, 1]]),  # form 3
        ("nanmean", np.ones((2, 12)), np.array([labels, labels]), [1, 1, 1]),
        ("nanmean", np.ones((2, 12)), np.array([nan_labels, nan_labels]), [1, 1, 1]),
        # (np.ones((12,)), np.array([labels, labels])),  # form 4
    ],
)
def test_groupby_reduce(array, by, expected, func, expected_groups, chunk, split_out, dtype):
    array = array.astype(dtype)
    if chunk:
        if expected_groups is None:
            pytest.skip()
        array = da.from_array(array, chunks=(3,) if array.ndim == 1 else (1, 3))
        by = da.from_array(by, chunks=(3,) if by.ndim == 1 else (1, 3))

    if "mean" in func:
        expected = np.array(expected, dtype=float)
    elif func == "sum":
        expected = np.array(expected, dtype=dtype)
    elif func == "count":
        expected = np.array(expected, dtype=int)

    result, _ = groupby_reduce(
        array,
        by,
        func=func,
        expected_groups=expected_groups,
        fill_value=123,
        split_out=split_out,
    )
    assert_equal(expected, result)


@pytest.mark.parametrize("size", ((12,), (12, 5)))
@pytest.mark.parametrize(
    "func",
    (
        "sum",
        "nansum",
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
        "argmax",
        pytest.param("nanargmax", marks=(pytest.mark.xfail,)),
        "argmin",
        pytest.param("nanargmin", marks=(pytest.mark.xfail,)),
    ),
)
def test_groupby_reduce_all(size, func):

    array = np.random.randn(*size)
    by = np.ones(size[-1])

    if "nan" in func and "nanarg" not in func:
        array[[1, 4, 5], ...] = np.nan
    elif "nanarg" in func and len(size) > 1:
        array[[1, 4, 5], 1] = np.nan

    expected = getattr(np, func)(array, axis=-1)
    expected = np.expand_dims(expected, -1)

    actual, _ = groupby_reduce(array, by, func=func)
    if "arg" in func:
        assert actual.dtype.kind == "i"
    assert_equal(actual, expected)

    actual, _ = groupby_reduce(da.from_array(array, chunks=3), by, func=func)
    if "arg" in func:
        assert actual.dtype.kind == "i"
    assert_equal(actual, expected)


@pytest.mark.parametrize("size", ((12,), (12, 5)))
@pytest.mark.parametrize("func", ("argmax", "nanargmax", "argmin", "nanargmin"))
def test_arg_reduction_dtype_is_int(size, func):
    """avoid bugs being hidden by the xfail in the above test."""

    array = np.random.randn(*size)
    by = np.ones(size[-1])

    if "nanarg" in func and len(size) > 1:
        array[[1, 4, 5], 1] = np.nan

    expected = getattr(np, func)(array, axis=-1)
    expected = np.expand_dims(expected, -1)

    actual, _ = groupby_reduce(array, by, func=func)
    assert actual.dtype.kind == "i"

    actual, _ = groupby_reduce(da.from_array(array, chunks=3), by, func=func)
    assert actual.dtype.kind == "i"


def test_groupby_reduce_count():
    array = np.array([0, 0, np.nan, np.nan, np.nan, 1, 1])
    labels = np.array(["a", "b", "b", "b", "c", "c", "c"])
    result, _ = groupby_reduce(array, labels, func="count")
    assert_equal(result, [1, 1, 2])


@pytest.mark.parametrize("func", ("sum", "prod"))
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_groupby_reduce_preserves_dtype(dtype, func):
    array = np.ones((2, 12), dtype=dtype)
    by = np.array([labels] * 2)
    result, _ = groupby_reduce(from_array(array, chunks=(-1, 4)), by, func=func)
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
    """Tests groupby_reduce with dask arrays against groupby_reduce with numpy arrays"""

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
    assert_equal(result, np.zeros(expected_shape, dtype=np.int64))

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
        np.ones((2, 12), dtype=np.int64),
        np.array([nan_labels, nan_labels[::-1]]),
        np.array([2, 8, 4], dtype=np.int64),
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


@pytest.mark.xfail
@pytest.mark.parametrize("func", ("nanargmax", "nanargmin"))
def test_npg_nanarg_bug(func):
    array = np.array([1, 1, 2, 1, 1, np.nan, 6, 1])
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1]) - 1

    actual = aggregate(labels, array, func=func).astype(int)
    expected = getattr(np, func)(array)

    assert_equal(actual, expected)
