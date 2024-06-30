from functools import partial

import numpy as np
import numpy_groupies as npg


def _get_aggregate(engine):
    return npg.aggregate_numpy if engine == "numpy" else npg.aggregate_numba


def _casting_wrapper(func, grp, dtype):
    """Used for generic aggregates. The group is dtype=object, need to cast back to fix weird bugs"""
    return func(grp.astype(dtype))


def sum_of_squares(
    group_idx,
    array,
    engine,
    *,
    axis=-1,
    size=None,
    fill_value=None,
    dtype=None,
):
    return _get_aggregate(engine).aggregate(
        group_idx,
        array,
        axis=axis,
        func="sumofsquares",
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def nansum_of_squares(
    group_idx,
    array,
    engine,
    *,
    axis=-1,
    size=None,
    fill_value=None,
    dtype=None,
):
    return _get_aggregate(engine).aggregate(
        group_idx,
        array,
        axis=axis,
        func="nansumofsquares",
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def nansum(group_idx, array, engine, *, axis=-1, size=None, fill_value=None, dtype=None):
    # npg takes out NaNs before calling np.bincount
    # This means that all NaN groups are equivalent to absent groups
    # This behaviour does not work for xarray

    return _get_aggregate(engine).aggregate(
        group_idx,
        np.where(np.isnan(array), 0, array),
        axis=axis,
        func="sum",
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def nanprod(group_idx, array, engine, *, axis=-1, size=None, fill_value=None, dtype=None):
    # npg takes out NaNs before calling np.bincount
    # This means that all NaN groups are equivalent to absent groups
    # This behaviour does not work for xarray

    return _get_aggregate(engine).aggregate(
        group_idx,
        np.where(np.isnan(array), 1, array),
        axis=axis,
        func="prod",
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def _len(group_idx, array, engine, *, func, axis=-1, size=None, fill_value=None, dtype=None):
    if array.dtype.kind in "US":
        array = np.broadcast_to(np.array([1]), array.shape)
    result = _get_aggregate(engine).aggregate(
        group_idx,
        array,
        axis=axis,
        func=func,
        size=size,
        fill_value=0,
        dtype=np.int64,
    )
    if fill_value is not None:
        result = result.astype(np.array([fill_value]).dtype)
        result[result == 0] = fill_value
    return result


len = partial(_len, func="len")
nanlen = partial(_len, func="nanlen")


def _var_std_wrapper(group_idx, array, engine, *, axis=-1, **kwargs):
    # Attempt to increase numerical stability by subtracting the first element.
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    # Cast any unsigned types first
    dtype = np.result_type(array, np.int8(-1) * array[0])
    array = array.astype(dtype, copy=False)
    first = _get_aggregate(engine).aggregate(group_idx, array, func="nanfirst", axis=axis)
    array = array - first[..., group_idx]
    return _get_aggregate(engine).aggregate(group_idx, array, axis=axis, **kwargs)


var = partial(_var_std_wrapper, func="var")
nanvar = partial(_var_std_wrapper, func="nanvar")
std = partial(_var_std_wrapper, func="std")
nanstd = partial(_var_std_wrapper, func="nanstd")


def median(group_idx, array, engine, *, axis=-1, size=None, fill_value=None, dtype=None):
    return npg.aggregate_numpy.aggregate(
        group_idx,
        array,
        func=partial(_casting_wrapper, np.median, dtype=np.result_type(array.dtype)),
        axis=axis,
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def nanmedian(group_idx, array, engine, *, axis=-1, size=None, fill_value=None, dtype=None):
    return npg.aggregate_numpy.aggregate(
        group_idx,
        array,
        func=partial(_casting_wrapper, np.nanmedian, dtype=np.result_type(array.dtype)),
        axis=axis,
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def quantile(group_idx, array, engine, *, q, axis=-1, size=None, fill_value=None, dtype=None):
    return npg.aggregate_numpy.aggregate(
        group_idx,
        array,
        func=partial(
            _casting_wrapper,
            partial(np.quantile, q=q),
            dtype=np.result_type(dtype, array.dtype),
        ),
        axis=axis,
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def nanquantile(group_idx, array, engine, *, q, axis=-1, size=None, fill_value=None, dtype=None):
    return npg.aggregate_numpy.aggregate(
        group_idx,
        array,
        func=partial(
            _casting_wrapper,
            partial(np.nanquantile, q=q),
            dtype=np.result_type(dtype, array.dtype),
        ),
        axis=axis,
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def mode_(array, nan_policy, dtype):
    from scipy.stats import mode

    # npg splits `array` into object arrays for each group
    # scipy.stats.mode does not like that
    # here we cast back
    return mode(array.astype(dtype, copy=False), nan_policy=nan_policy, axis=-1, keepdims=True).mode


def mode(group_idx, array, engine, *, axis=-1, size=None, fill_value=None, dtype=None):
    return npg.aggregate_numpy.aggregate(
        group_idx,
        array,
        func=partial(mode_, nan_policy="propagate", dtype=array.dtype),
        axis=axis,
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def nanmode(group_idx, array, engine, *, axis=-1, size=None, fill_value=None, dtype=None):
    return npg.aggregate_numpy.aggregate(
        group_idx,
        array,
        func=partial(mode_, nan_policy="omit", dtype=array.dtype),
        axis=axis,
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )
