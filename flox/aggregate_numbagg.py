import numpy as np
from numbagg.grouped import group_nanmean, group_nansum


def nansum_of_squares(
    group_idx, array, *, axis=-1, func="sum", size=None, fill_value=None, dtype=None
):
    return group_nansum(
        array**2,
        group_idx,
        axis=axis,
        num_labels=size,
        # fill_value=fill_value,
        # dtype=dtype,
    )


def nansum(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    return group_nansum(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        # fill_value=fill_value,
        # dtype=dtype,
    )


def nanmean(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    return group_nanmean(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        # fill_value=fill_value,
        # dtype=dtype,
    )


def nanlen(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    return group_nansum(
        (~np.isnan(array)).astype(int),
        group_idx,
        axis=axis,
        num_labels=size,
        # fill_value=fill_value,
        # dtype=dtype,
    )


# sum = nansum
# mean = nanmean
# sum_of_squares = nansum_of_squares
