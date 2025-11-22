# Unlike the other aggregate_* submodules, this one simply defines a wrapper function
# because we run the groupby on the underlying dense data.

from collections.abc import Callable
from functools import partial
from typing import Any, TypeAlias

import numpy as np
import sparse

from flox.factorize import _factorize_multiple, factorize_
from flox.lib import _is_sparse_supported_reduction
from flox.xrdtypes import INF, NINF, _get_fill_value
from flox.xrutils import notnull


def nanadd(a, b):
    """
    Annoyingly, there is no numpy ufunc for nan-skipping elementwise addition
    unlike np.fmin, np.fmax :(

    From https://stackoverflow.com/a/50642947/1707127
    """
    ab = a + b
    return np.where(np.isnan(ab), np.where(np.isnan(a), b, a), ab)


CallableMap: TypeAlias = dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]
BINARY_OPS: CallableMap = {
    "sum": np.add,
    "nansum": nanadd,
    "max": np.maximum,
    "nanmax": np.fmax,
    "min": np.minimum,
    "nanmin": np.fmin,
}
HYPER_OPS: CallableMap = {"sum": np.multiply, "nansum": np.multiply}
IDENTITY: dict[str, Any] = {
    "sum": 0,
    "nansum": 0,
    "prod": 1,
    "nanprod": 1,
    "max": NINF,
    "nanmax": NINF,
    "min": INF,
    "nanmin": INF,
}


def _sparse_agg(
    group_idx: np.ndarray,
    array: sparse.COO,
    func: str,
    engine: str,
    axis: int = -1,
    size: int | None = None,
    fill_value=None,
    dtype=None,
    **kwargs,
):
    """Wrapper function, that unwraps the underlying dense arrays, executes the groupby,
    and constructs the output sparse array."""
    from flox.aggregations import generic_aggregate

    if not isinstance(array, sparse.COO):
        raise ValueError("Sparse aggregations only supported for sparse.COO arrays")

    if not _is_sparse_supported_reduction(func):
        raise ValueError(f"{func} is unsupported for sparse arrays.")

    group_idx_subset = group_idx[array.coords[axis, :]]
    if array.ndim > 1:
        new_by = tuple(array.coords[:axis, :]) + (group_idx_subset,)
    else:
        new_by = (group_idx_subset,)
    codes, groups, shape = _factorize_multiple(
        new_by, expected_groups=(None,) * len(new_by), any_by_dask=False
    )
    # factorize again so we can construct a sparse result
    sparse_codes, sparse_groups, sparse_shape, _, sparse_size, _ = factorize_(codes, axes=(0,))

    dense_result = generic_aggregate(
        sparse_codes,
        array.data,
        func=func,
        engine=engine,
        dtype=dtype,
        size=sparse_size,
        fill_value=fill_value,
    )
    dense_counts = generic_aggregate(
        sparse_codes,
        array.data,
        # This counts is used to handle fill_value, so we need a count
        # of populated data, regardless of NaN value
        func="len",
        engine=engine,
        dtype=int,
        size=sparse_size,
        fill_value=0,
    )
    assert len(sparse_groups) == 1
    result_coords = np.stack(tuple(g[i] for g, i in zip(groups, np.unravel_index(*sparse_groups, shape))))

    full_shape = array.shape[:-1] + (size,)
    count = sparse.COO(coords=result_coords, data=dense_counts, shape=full_shape, fill_value=0)

    assert axis in (-1, array.ndim - 1)
    grouped_count = generic_aggregate(
        group_idx, group_idx, engine=engine, func="len", dtype=np.int64, size=size, fill_value=0
    )
    total_count = sparse.COO.from_numpy(
        np.expand_dims(grouped_count, tuple(range(array.ndim - 1))), fill_value=0
    )

    assert func in BINARY_OPS
    binop = BINARY_OPS[func]
    ident = _get_fill_value(array.dtype, IDENTITY[func])
    diff_count = total_count - count
    if (hyper_op := HYPER_OPS.get(func, None)) is not None:
        fill = hyper_op(diff_count, array.fill_value) if (diff_count > 0).any() else ident
    else:
        if "max" in func or "min" in func:
            # Note that fill_value for total_count, and count is 0.
            # So the fill_value for the `fill` result is the False branch i.e. `ident`
            fill = np.where(diff_count > 0, array.fill_value, ident)
        else:
            raise NotImplementedError

    result = sparse.COO(coords=result_coords, data=dense_result, shape=full_shape, fill_value=ident)
    with_fill = binop(result, fill)
    return with_fill


def nanlen(
    group_idx: np.ndarray,
    array: sparse.COO,
    engine: str,
    axis: int = -1,
    size: int | None = None,
    fill_value=None,
    dtype=None,
    **kwargs,
):
    new_array = sparse.COO(
        coords=array.coords,
        data=notnull(array.data),
        shape=array.shape,
        fill_value=notnull(array.fill_value),
    )
    return _sparse_agg(
        group_idx, new_array, func="sum", engine=engine, axis=axis, size=size, fill_value=0, dtype=dtype
    )


def mean(
    group_idx: np.ndarray,
    array: sparse.COO,
    engine: str,
    axis: int = -1,
    size: int | None = None,
    fill_value=None,
    dtype=None,
    **kwargs,
):
    sums = sum(
        group_idx, array, func="sum", engine=engine, axis=axis, size=size, fill_value=fill_value, dtype=dtype
    )
    counts = nanlen(
        group_idx, array, func="sum", engine=engine, axis=axis, size=size, fill_value=0, dtype=dtype
    )
    return sums / counts


def nanmean(
    group_idx: np.ndarray,
    array: sparse.COO,
    engine: str,
    axis: int = -1,
    size: int | None = None,
    fill_value=None,
    dtype=None,
    **kwargs,
):
    sums = sum(
        group_idx,
        array,
        func="nansum",
        engine=engine,
        axis=axis,
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )
    counts = nanlen(
        group_idx, array, func="sum", engine=engine, axis=axis, size=size, fill_value=0, dtype=dtype
    )
    return sums / counts


sum = partial(_sparse_agg, func="sum")
nansum = partial(_sparse_agg, func="nansum")
max = partial(_sparse_agg, func="max")
nanmax = partial(_sparse_agg, func="nanmax")
min = partial(_sparse_agg, func="min")
nanmin = partial(_sparse_agg, func="nanmin")
