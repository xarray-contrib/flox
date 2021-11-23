import numpy as np
import numpy_groupies as npg


def sum_of_squares(
    group_idx, array, *, axis=-1, func="sum", size=None, fill_value=None, dtype=None
):

    return npg.aggregate_numpy.aggregate(
        group_idx,
        array ** 2,
        axis=axis,
        func=func,
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def nansum(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    # npg takes out NaNs before calling np.bincount
    # This means that all NaN groups are equivalent to absent groups
    # This behaviour does not work for xarray

    return npg.aggregate_numpy.aggregate(
        group_idx,
        np.where(np.isnan(array), 0, array),
        axis=axis,
        func="sum",
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def nanprod(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    # npg takes out NaNs before calling np.bincount
    # This means that all NaN groups are equivalent to absent groups
    # This behaviour does not work for xarray

    return npg.aggregate_numpy.aggregate(
        group_idx,
        np.where(np.isnan(array), 1, array),
        axis=axis,
        func="prod",
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def nansum_of_squares(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    return sum_of_squares(
        group_idx, array, func="nansum", size=size, fill_value=fill_value, axis=axis, dtype=dtype
    )
