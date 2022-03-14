from functools import partial

import numpy as np
import numpy_groupies as npg


def _get_aggregate(engine):
    return npg.aggregate_numpy if engine == "numpy" else npg.aggregate_numba


def sum_of_squares(
    group_idx, array, engine, *, axis=-1, func="sum", size=None, fill_value=None, dtype=None
):

    return _get_aggregate(engine).aggregate(
        group_idx,
        array**2,
        axis=axis,
        func=func,
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


def nansum_of_squares(group_idx, array, engine, *, axis=-1, size=None, fill_value=None, dtype=None):
    return sum_of_squares(
        group_idx,
        array,
        engine=engine,
        func="nansum",
        size=size,
        fill_value=fill_value,
        axis=axis,
        dtype=dtype,
    )


def _len(group_idx, array, engine, *, func, axis=-1, size=None, fill_value=None, dtype=None):
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
