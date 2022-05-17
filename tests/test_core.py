from functools import reduce

import numpy as np
import pandas as pd
import pytest
from numpy_groupies.aggregate_numpy import aggregate

from flox.core import (
    _convert_expected_groups_to_index,
    _get_optimal_chunks_for_groups,
    factorize_,
    find_group_cohorts,
    groupby_reduce,
    rechunk_for_cohorts,
    reindex_,
)

from . import assert_equal, engine, has_dask, raise_if_dask_computes, requires_dask

labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0])
nan_labels = labels.astype(float)  # copy
nan_labels[:5] = np.nan
labels2d = np.array([labels[:5], np.flip(labels[:5])])

# isort:off
if has_dask:
    import dask
    import dask.array as da
    from dask.array import from_array

    dask.config.set(scheduler="sync")
else:

    def dask_array_ones(*args):
        return None


# isort:on

ALL_FUNCS = (
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
    pytest.param("nanargmax", marks=(pytest.mark.skip,)),
    "argmin",
    pytest.param("nanargmin", marks=(pytest.mark.skip,)),
    "any",
    "all",
    pytest.param("median", marks=(pytest.mark.skip,)),
    pytest.param("nanmedian", marks=(pytest.mark.skip,)),
)


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
def test_groupby_reduce(
    array, by, expected, func, expected_groups, chunk, split_out, dtype, engine
):
    array = array.astype(dtype)
    if chunk:
        if not has_dask or expected_groups is None:
            pytest.skip()
        array = da.from_array(array, chunks=(3,) if array.ndim == 1 else (1, 3))
        by = da.from_array(by, chunks=(3,) if by.ndim == 1 else (1, 3))

    if "mean" in func:
        expected = np.array(expected, dtype=float)
    elif func == "sum":
        expected = np.array(expected, dtype=dtype)
    elif func == "count":
        expected = np.array(expected, dtype=int)

    result, groups, = groupby_reduce(
        array,
        by,
        func=func,
        expected_groups=expected_groups,
        fill_value=123,
        split_out=split_out,
        engine=engine,
    )
    assert_equal(groups, [0, 1, 2])
    assert_equal(expected, result)


def gen_array_by(size, func):
    by = np.ones(size[-1])
    rng = np.random.default_rng(12345)
    array = rng.random(size)
    if "nan" in func and "nanarg" not in func:
        array[[1, 4, 5], ...] = np.nan
    elif "nanarg" in func and len(size) > 1:
        array[[1, 4, 5], 1] = np.nan
    if func in ["any", "all"]:
        array = array > 0.5
    return array, by


@pytest.mark.parametrize("chunks", [None, -1, 3, 4])
@pytest.mark.parametrize("nby", [1, 2, 3])
@pytest.mark.parametrize("size", ((12,), (12, 9)))
@pytest.mark.parametrize("add_nan_by", [True, False])
@pytest.mark.parametrize("func", ALL_FUNCS)
def test_groupby_reduce_all(nby, size, chunks, func, add_nan_by, engine):
    if chunks is not None and not has_dask:
        pytest.skip()
    if "arg" in func and engine == "flox":
        pytest.skip()

    array, by = gen_array_by(size, func)
    if chunks:
        array = dask.array.from_array(array, chunks=chunks)
    by = (by,) * nby
    by = [b + idx for idx, b in enumerate(by)]
    if add_nan_by:
        for idx in range(nby):
            by[idx][2 * idx : 2 * idx + 3] = np.nan
    by = tuple(by)
    nanmask = reduce(np.logical_or, (np.isnan(b) for b in by))

    finalize_kwargs = [{}]
    if "var" in func or "std" in func:
        finalize_kwargs = finalize_kwargs + [{"ddof": 1}, {"ddof": 0}]
        fill_value = np.nan
    else:
        fill_value = None

    for kwargs in finalize_kwargs:
        flox_kwargs = dict(func=func, engine=engine, finalize_kwargs=kwargs, fill_value=fill_value)
        with np.errstate(invalid="ignore", divide="ignore"):
            if "arg" in func and add_nan_by:
                array[..., nanmask] = np.nan
                expected = getattr(np, "nan" + func)(array, axis=-1, **kwargs)
            else:
                expected = getattr(np, func)(array[..., ~nanmask], axis=-1, **kwargs)
        for _ in range(nby):
            expected = np.expand_dims(expected, -1)

        actual, *groups = groupby_reduce(array, *by, **flox_kwargs)
        assert actual.ndim == (array.ndim + nby - 1)
        assert expected.ndim == (array.ndim + nby - 1)
        expected_groups = tuple(np.array([idx + 1.0]) for idx in range(nby))
        for actual_group, expect in zip(groups, expected_groups):
            assert_equal(actual_group, expect)
        if "arg" in func:
            assert actual.dtype.kind == "i"
        assert_equal(actual, expected)

        if not has_dask:
            continue
        for method in ["map-reduce", "cohorts", "split-reduce"]:
            if "arg" in func and method != "map-reduce":
                continue
            actual, *groups = groupby_reduce(array, *by, method=method, **flox_kwargs)
            for actual_group, expect in zip(groups, expected_groups):
                assert_equal(actual_group, expect)
            if "arg" in func:
                assert actual.dtype.kind == "i"
            assert_equal(actual, expected)


@requires_dask
@pytest.mark.parametrize("size", ((12,), (12, 5)))
@pytest.mark.parametrize("func", ("argmax", "nanargmax", "argmin", "nanargmin"))
def test_arg_reduction_dtype_is_int(size, func):
    """avoid bugs being hidden by the xfail in the above test."""

    rng = np.random.default_rng(12345)
    array = rng.random(size)
    by = np.ones(size[-1])

    if "nanarg" in func and len(size) > 1:
        array[[1, 4, 5], 1] = np.nan

    expected = getattr(np, func)(array, axis=-1)
    expected = np.expand_dims(expected, -1)

    actual, _ = groupby_reduce(array, by, func=func, engine="numpy")
    assert actual.dtype.kind == "i"

    actual, _ = groupby_reduce(da.from_array(array, chunks=3), by, func=func, engine="numpy")
    assert actual.dtype.kind == "i"


def test_groupby_reduce_count():
    array = np.array([0, 0, np.nan, np.nan, np.nan, 1, 1])
    labels = np.array(["a", "b", "b", "b", "c", "c", "c"])
    result, _ = groupby_reduce(array, labels, func="count")
    assert_equal(result, [1, 1, 2])


def test_func_is_aggregation():
    from flox.aggregations import mean

    array = np.array([0, 0, np.nan, np.nan, np.nan, 1, 1])
    labels = np.array(["a", "b", "b", "b", "c", "c", "c"])
    expected, _ = groupby_reduce(array, labels, func="mean")
    actual, _ = groupby_reduce(array, labels, func=mean)
    assert_equal(actual, expected)


@requires_dask
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
    actual = reindex_(result, groups, pd.Index(np.unique(by)), axis=0, fill_value=0)
    np.testing.assert_equal(expected, actual)

    array = np.ones((4, 2, 12))
    by = np.array([labels] * 2)

    expected = aggregate(by.ravel(), array.reshape(4, 24), func="sum", axis=-1, fill_value=0)
    result, groups = groupby_reduce(array, by, func="sum")
    actual = reindex_(result, groups, pd.Index(np.unique(by)), axis=-1, fill_value=0)
    assert_equal(expected, actual)

    array = np.ones((4, 2, 12))
    by = np.broadcast_to(np.array([labels] * 2), array.shape)
    expected = aggregate(by.ravel(), array.ravel(), func="sum", axis=-1)
    result, groups = groupby_reduce(array, by, func="sum")
    actual = reindex_(result, groups, pd.Index(np.unique(by)), axis=-1, fill_value=0)
    assert_equal(expected, actual)

    array = np.ones((2, 3, 4))
    by = np.ones((2, 3, 4))

    actual, _ = groupby_reduce(array, by, axis=(1, 2), func="sum")
    expected = np.sum(array, axis=(1, 2), keepdims=True).squeeze(2)
    assert_equal(actual, expected)


@requires_dask
@pytest.mark.parametrize("reindex", [None, False, True])
@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize("add_nan", [False, True])
@pytest.mark.parametrize("dtype", (float,))
@pytest.mark.parametrize(
    "shape, array_chunks, group_chunks",
    [
        ((12,), (3,), 3),  # form 1
        ((12,), (3,), (4,)),  # form 1, chunks not aligned
        ((12,), ((3, 5, 4),), (2,)),  # form 1
        ((10, 12), (3, 3), -1),  # form 3
        ((10, 12), (3, 3), 3),  # form 3
    ],
)
def test_groupby_agg_dask(func, shape, array_chunks, group_chunks, add_nan, dtype, engine, reindex):
    """Tests groupby_reduce with dask arrays against groupby_reduce with numpy arrays"""

    rng = np.random.default_rng(12345)
    array = dask.array.from_array(rng.random(shape), chunks=array_chunks).astype(dtype)
    array = dask.array.ones(shape, chunks=array_chunks)

    if func in ["first", "last"]:
        pytest.skip()

    if "arg" in func and (engine == "flox" or reindex):
        pytest.skip()

    labels = np.array([0, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0])
    if add_nan:
        labels = labels.astype(float)
        labels[:3] = np.nan  # entire block is NaN when group_chunks=3
        labels[-2:] = np.nan

    kwargs = dict(
        func=func, expected_groups=[0, 1, 2], fill_value=False if func in ["all", "any"] else 123
    )

    expected, _ = groupby_reduce(array.compute(), labels, engine="numpy", **kwargs)
    actual, _ = groupby_reduce(array.compute(), labels, engine=engine, **kwargs)
    assert_equal(actual, expected)

    with raise_if_dask_computes():
        actual, _ = groupby_reduce(array, labels, engine=engine, **kwargs)
    assert_equal(actual, expected)

    by = from_array(labels, group_chunks)
    with raise_if_dask_computes():
        actual, _ = groupby_reduce(array, by, engine=engine, **kwargs)
    assert_equal(expected, actual)

    kwargs["expected_groups"] = [0, 2, 1]
    with raise_if_dask_computes():
        actual, groups = groupby_reduce(array, by, engine=engine, **kwargs, sort=False)
    assert_equal(groups, [0, 2, 1])
    assert_equal(expected, actual[..., [0, 2, 1]])

    kwargs["expected_groups"] = [0, 2, 1]
    with raise_if_dask_computes():
        actual, groups = groupby_reduce(array, by, engine=engine, **kwargs, sort=True)
    assert_equal(groups, [0, 1, 2])
    assert_equal(expected, actual)


def test_numpy_reduce_axis_subset(engine):
    # TODO: add NaNs
    by = labels2d
    array = np.ones_like(by)
    kwargs = dict(func="count", engine=engine, fill_value=0)
    result, _ = groupby_reduce(array, by, **kwargs, axis=1)
    assert_equal(result, [[2, 3], [2, 3]])

    by = np.broadcast_to(labels2d, (3, *labels2d.shape))
    array = np.ones_like(by)
    result, _ = groupby_reduce(array, by, **kwargs, axis=1)
    subarr = np.array([[1, 1], [1, 1], [0, 2], [1, 1], [1, 1]])
    expected = np.tile(subarr, (3, 1, 1))
    assert_equal(result, expected)

    result, _ = groupby_reduce(array, by, **kwargs, axis=2)
    subarr = np.array([[2, 3], [2, 3]])
    expected = np.tile(subarr, (3, 1, 1))
    assert_equal(result, expected)

    result, _ = groupby_reduce(array, by, **kwargs, axis=(1, 2))
    expected = np.array([[4, 6], [4, 6], [4, 6]])
    assert_equal(result, expected)

    result, _ = groupby_reduce(array, by, **kwargs, axis=(2, 1))
    assert_equal(result, expected)

    result, _ = groupby_reduce(array, by[0, ...], **kwargs, axis=(1, 2))
    expected = np.array([[4, 6], [4, 6], [4, 6]])
    assert_equal(result, expected)


@requires_dask
def test_dask_reduce_axis_subset():

    by = labels2d
    array = np.ones_like(by)
    with raise_if_dask_computes():
        result, _ = groupby_reduce(
            da.from_array(array, chunks=(2, 3)),
            da.from_array(by, chunks=(2, 2)),
            func="count",
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
            func="count",
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
            func="count",
            axis=2,
            expected_groups=[0, 2],
        )
    assert_equal(result, expected)

    with pytest.raises(NotImplementedError):
        groupby_reduce(
            da.from_array(array, chunks=(1, 3, 2)),
            da.from_array(by, chunks=(2, 2, 2)),
            func="count",
            axis=2,
        )


@requires_dask
@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize(
    "axis", [None, (0, 1, 2), (0, 1), (0, 2), (1, 2), 0, 1, 2, (0,), (1,), (2,)]
)
def test_groupby_reduce_axis_subset_against_numpy(func, axis, engine):
    if "arg" in func and engine == "flox":
        pytest.skip()

    if not isinstance(axis, int) and "arg" in func and (axis is None or len(axis) > 1):
        pytest.skip()
    if func in ["all", "any"]:
        fill_value = False
    else:
        fill_value = 123
    # tests against the numpy output to make sure dask compute matches
    by = np.broadcast_to(labels2d, (3, *labels2d.shape))
    rng = np.random.default_rng(12345)
    array = rng.random(by.shape)
    kwargs = dict(
        func=func, axis=axis, expected_groups=[0, 2], fill_value=fill_value, engine=engine
    )
    with raise_if_dask_computes():
        actual, _ = groupby_reduce(
            da.from_array(array, chunks=(-1, 2, 3)),
            da.from_array(by, chunks=(-1, 2, 2)),
            **kwargs,
        )
    expected, _ = groupby_reduce(array, by, **kwargs)
    if engine == "flox":
        kwargs.pop("engine")
        expected_npg, _ = groupby_reduce(array, by, **kwargs, engine="numpy")
        assert_equal(expected_npg, expected)
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
def test_groupby_reduce_nans(chunks, axis, groups, expected_shape, engine):
    def _maybe_chunk(arr):
        if chunks:
            if not has_dask:
                pytest.skip()
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
        func="count",
        expected_groups=groups,
        axis=axis,
        fill_value=0,
        engine=engine,
    )
    assert_equal(result, np.zeros(expected_shape, dtype=np.int64))

    # now when subsets are NaN
    # labels = np.array([0, 0, 1, 1, 1], dtype=float)
    # labels2d = np.array([labels[:5], np.flip(labels[:5])])
    # labels2d[0, :5] = np.nan
    # labels2d[1, 5:] = np.nan
    # by = np.broadcast_to(labels2d, (3, *labels2d.shape))


@requires_dask
def test_groupby_all_nan_blocks(engine):
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
        engine=engine,
    )
    assert_equal(actual, expected)


@pytest.mark.parametrize("axis", (0, 1, 2, -1))
def test_reindex(axis):
    shape = [2, 2, 2]
    fill_value = 0

    array = np.broadcast_to(np.array([1, 2]), shape)
    groups = np.array(["a", "b"])
    expected_groups = pd.Index(["a", "b", "c"])
    actual = reindex_(array, groups, expected_groups, fill_value=fill_value, axis=axis)

    if axis < 0:
        axis = array.ndim + axis
    result_shape = tuple(len(expected_groups) if ax == axis else s for ax, s in enumerate(shape))
    slicer = tuple(slice(None, s) for s in shape)
    expected = np.full(result_shape, fill_value)
    expected[slicer] = array

    assert_equal(actual, expected)


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


@pytest.mark.parametrize("method", ["split-reduce", "cohorts", "map-reduce"])
@pytest.mark.parametrize("chunk_labels", [False, True])
@pytest.mark.parametrize("chunks", ((), (1,), (2,)))
def test_groupby_bins(chunk_labels, chunks, engine, method) -> None:
    array = [1, 1, 1, 1, 1, 1]
    labels = [0.2, 1.5, 1.9, 2, 3, 20]

    if method in ["split-reduce", "cohorts"] and chunk_labels:
        pytest.xfail()

    if chunks:
        if not has_dask:
            pytest.skip()
        array = dask.array.from_array(array, chunks=chunks)
        if chunk_labels:
            labels = dask.array.from_array(labels, chunks=chunks)

    with raise_if_dask_computes():
        actual, groups = groupby_reduce(
            array,
            labels,
            func="count",
            expected_groups=np.array([1, 2, 4, 5]),
            isbin=True,
            fill_value=0,
            engine=engine,
            method=method,
        )
    expected = np.array([3, 1, 0])
    for left, right in zip(groups, pd.IntervalIndex.from_arrays([1, 2, 4], [2, 4, 5]).to_numpy()):
        assert left == right
    assert_equal(actual, expected)


@pytest.mark.parametrize(
    "inchunks, expected",
    [
        [(1,) * 10, (3, 2, 2, 3)],
        [(2,) * 5, (3, 2, 2, 3)],
        [(3, 3, 3, 1), (3, 2, 5)],
        [(3, 1, 1, 2, 1, 1, 1), (3, 2, 2, 3)],
        [(3, 2, 2, 3), (3, 2, 2, 3)],
        [(4, 4, 2), (3, 4, 3)],
        [(5, 5), (5, 5)],
        [(6, 4), (5, 5)],
        [(7, 3), (7, 3)],
        [(8, 2), (7, 3)],
        [(9, 1), (10,)],
        [(10,), (10,)],
    ],
)
def test_rechunk_for_blockwise(inchunks, expected):
    labels = np.array([1, 1, 1, 2, 2, 3, 3, 5, 5, 5])
    assert _get_optimal_chunks_for_groups(inchunks, labels) == expected


@requires_dask
@pytest.mark.parametrize(
    "expected, labels, chunks, merge",
    [
        [[[1, 2, 3, 4]], [1, 2, 3, 1, 2, 3, 4], (3, 4), True],
        [[[1, 2, 3], [4]], [1, 2, 3, 1, 2, 3, 4], (3, 4), False],
        [[[1], [2], [3], [4]], [1, 2, 3, 1, 2, 3, 4], (2, 2, 2, 1), False],
        [[[3], [2], [1], [4]], [1, 2, 3, 1, 2, 3, 4], (2, 2, 2, 1), True],
        [[[1, 2, 3], [4]], [1, 2, 3, 1, 2, 3, 4], (3, 3, 1), True],
        [[[1, 2, 3], [4]], [1, 2, 3, 1, 2, 3, 4], (3, 3, 1), False],
        [
            [[2, 3, 4, 1], [5], [0]],
            np.repeat(np.arange(6), [4, 4, 12, 2, 3, 4]),
            (4, 8, 4, 9, 4),
            True,
        ],
    ],
)
def test_find_group_cohorts(expected, labels, chunks, merge):
    actual = list(find_group_cohorts(labels, (chunks,), merge, method="cohorts"))
    assert actual == expected, (actual, expected)

    actual = find_group_cohorts(labels, (chunks,), merge, method="split-reduce")
    expected = [[label] for label in np.unique(labels)]
    assert actual == expected, (actual, expected)


@requires_dask
@pytest.mark.parametrize(
    "chunk_at,expected",
    [
        [1, ((1, 6, 1, 6, 1, 6, 1, 6, 1, 1),)],
        [0, ((7, 7, 7, 7, 2),)],
        [3, ((3, 4, 3, 4, 3, 4, 3, 4, 2),)],
    ],
)
def test_rechunk_for_cohorts(chunk_at, expected):
    array = dask.array.ones((30,), chunks=7)
    labels = np.arange(0, 30) % 7
    rechunked = rechunk_for_cohorts(array, axis=-1, force_new_chunk_at=chunk_at, labels=labels)
    assert rechunked.chunks == expected


@pytest.mark.parametrize("chunks", [None, 3])
@pytest.mark.parametrize("fill_value", [123, np.nan])
@pytest.mark.parametrize("func", ALL_FUNCS)
def test_fill_value_behaviour(func, chunks, fill_value, engine):
    # fill_value = np.nan tests promotion of int counts to float
    # This is used by xarray
    if func in ["all", "any"] or "arg" in func:
        pytest.skip()
    if chunks is not None and not has_dask:
        pytest.skip()

    if func == "count":

        def npfunc(x):
            x = np.asarray(x)
            return (~np.isnan(x)).sum()

    else:
        npfunc = getattr(np, func)

    by = np.array([1, 2, 3, 1, 2, 3])
    array = np.array([np.nan, 1, 1, np.nan, 1, 1])
    if chunks:
        array = dask.array.from_array(array, chunks)
    actual, _ = groupby_reduce(
        array, by, func=func, engine=engine, fill_value=fill_value, expected_groups=[0, 1, 2, 3]
    )
    expected = np.array([fill_value, fill_value, npfunc([1.0, 1.0]), npfunc([1.0, 1.0])])
    assert_equal(actual, expected)


@requires_dask
@pytest.mark.parametrize("func", ["mean", "sum"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_dtype_preservation(dtype, func, engine):
    if func == "sum" or (func == "mean" and "float" in dtype):
        expected = np.dtype(dtype)
    elif func == "mean" and "int" in dtype:
        expected = np.float64
    array = np.ones((20,), dtype=dtype)
    by = np.ones(array.shape, dtype=int)
    actual, _ = groupby_reduce(array, by, func=func, engine=engine)
    assert actual.dtype == expected

    array = dask.array.from_array(array, chunks=(4,))
    actual, _ = groupby_reduce(array, by, func=func, engine=engine)
    assert actual.dtype == expected


@requires_dask
@pytest.mark.parametrize("method", ["split-reduce", "map-reduce", "cohorts"])
def test_cohorts(method):
    repeats = [4, 4, 12, 2, 3, 4]
    labels = np.repeat(np.arange(6), repeats)
    array = dask.array.from_array(labels, chunks=(4, 8, 4, 9, 4))

    actual, actual_groups = groupby_reduce(array, labels, func="count", method=method)
    assert_equal(actual_groups, np.arange(6))
    assert_equal(actual, repeats)


@requires_dask
@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize("axis", (-1, None))
@pytest.mark.parametrize("method", ["blockwise", "cohorts", "map-reduce", "split-reduce"])
def test_cohorts_nd_by(func, method, axis, engine):
    o = dask.array.ones((3,), chunks=-1)
    o2 = dask.array.ones((2, 3), chunks=-1)

    array = dask.array.block([[o, 2 * o], [3 * o2, 4 * o2]])
    by = array.compute().astype(int)
    by[0, 1] = 30
    by[2, 1] = 40
    by[0, 4] = 31
    array = np.broadcast_to(array, (2, 3) + array.shape)

    if "arg" in func and (axis is None or engine == "flox"):
        pytest.skip()

    if func in ["any", "all"]:
        fill_value = False
    else:
        fill_value = -123

    if axis is not None and method != "map-reduce":
        pytest.xfail()

    kwargs = dict(func=func, engine=engine, method=method, axis=axis, fill_value=fill_value)
    actual, groups = groupby_reduce(array, by, **kwargs)
    expected, sorted_groups = groupby_reduce(array.compute(), by, **kwargs)
    assert_equal(groups, sorted_groups)
    assert_equal(actual, expected)

    actual, groups = groupby_reduce(array, by, sort=False, **kwargs)
    if method == "cohorts":
        assert_equal(groups, [4, 3, 40, 2, 31, 1, 30])
    elif method in ("split-reduce", "map-reduce"):
        assert_equal(groups, [1, 30, 2, 31, 3, 4, 40])
    elif method == "blockwise":
        assert_equal(groups, [1, 30, 2, 31, 3, 40, 4])
    reindexed = reindex_(actual, groups, pd.Index(sorted_groups))
    assert_equal(reindexed, expected)


@pytest.mark.parametrize("func", ["sum", "count"])
@pytest.mark.parametrize("fill_value, expected", ((0, np.integer), (np.nan, np.floating)))
def test_dtype_promotion(func, fill_value, expected, engine):
    array = np.array([1, 1])
    by = [0, 1]

    actual, _ = groupby_reduce(
        array, by, func=func, expected_groups=[1, 2], fill_value=fill_value, engine=engine
    )
    assert np.issubdtype(actual.dtype, expected)


@pytest.mark.parametrize("func", ["mean", "nanmean"])
def test_empty_bins(func, engine):
    array = np.ones((2, 3, 2))
    by = np.broadcast_to([0, 1], array.shape)

    actual, _ = groupby_reduce(
        array,
        by,
        func=func,
        expected_groups=[-1, 0, 1, 2],
        isbin=True,
        engine=engine,
        axis=(0, 1, 2),
    )
    expected = np.array([1.0, 1.0, np.nan])
    assert_equal(actual, expected)


def test_datetime_binning():
    time_bins = pd.date_range(start="2010-08-01", end="2010-08-15", freq="24H")
    by = pd.date_range("2010-08-01", "2010-08-15", freq="15min")

    (actual,) = _convert_expected_groups_to_index((time_bins,), isbin=(True,), sort=False)
    expected = pd.IntervalIndex.from_arrays(time_bins[:-1], time_bins[1:])
    assert_equal(actual, expected)

    ret = factorize_((by.to_numpy(),), axis=0, expected_groups=(actual,))
    group_idx = ret[0]
    expected = pd.cut(by, time_bins).codes.copy()
    expected[0] = 14  # factorize doesn't return -1 for nans
    assert_equal(group_idx, expected)


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_bool_reductions(func, engine):
    if "arg" in func and engine == "flox":
        pytest.skip()
    groups = np.array([1, 1, 1])
    data = np.array([True, True, False])
    expected = np.expand_dims(getattr(np, func)(data), -1)
    actual, _ = groupby_reduce(data, groups, func=func, engine=engine)
    assert_equal(expected, actual)


@requires_dask
def test_map_reduce_blockwise_mixed():
    t = pd.date_range("2000-01-01", "2000-12-31", freq="D").to_series()
    data = t.dt.dayofyear
    actual = groupby_reduce(
        dask.array.from_array(data.values, chunks=365),
        t.dt.month,
        func="mean",
        method="split-reduce",
    )
    expected = groupby_reduce(data, t.dt.month, func="mean")
    assert_equal(expected, actual)
