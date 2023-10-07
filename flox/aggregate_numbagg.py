from functools import partial

import numbagg
import numbagg.grouped
import numpy as np


def _numbagg_wrapper(
    group_idx,
    array,
    *,
    axis=-1,
    func="sum",
    size=None,
    fill_value=None,
    dtype=None,
    numbagg_func=None,
):
    return numbagg_func(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        # The following are unsupported
        # fill_value=fill_value,
        # dtype=dtype,
    )


def nansum(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    if np.issubdtype(array.dtype, np.bool_):
        array = array.astype(np.in64)
    return numbagg.grouped.group_nansum(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        # fill_value=fill_value,
        # dtype=dtype,
    )


def nanmean(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    if np.issubdtype(array.dtype, np.int_):
        array = array.astype(np.float64)
    return numbagg.grouped.group_nanmean(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        # fill_value=fill_value,
        # dtype=dtype,
    )


def nanvar(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None, ddof=0):
    assert ddof != 0
    if np.issubdtype(array.dtype, np.int_):
        array = array.astype(np.float64)
    return numbagg.grouped.group_nanvar(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        # ddof=0,
        # fill_value=fill_value,
        # dtype=dtype,
    )


def nanstd(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None, ddof=0):
    assert ddof != 0
    if np.issubdtype(array.dtype, np.int_):
        array = array.astype(np.float64)
    return numbagg.grouped.group_nanstd(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        # ddof=0,
        # fill_value=fill_value,
        # dtype=dtype,
    )


nansum_of_squares = partial(_numbagg_wrapper, numbagg_func=numbagg.grouped.group_nansum_of_squares)
nanlen = partial(_numbagg_wrapper, numbagg_func=numbagg.grouped.group_nancount)
nanprod = partial(_numbagg_wrapper, numbagg_func=numbagg.grouped.group_nanprod)
nanfirst = partial(_numbagg_wrapper, numbagg_func=numbagg.grouped.group_nanfirst)
nanlast = partial(_numbagg_wrapper, numbagg_func=numbagg.grouped.group_nanlast)
# nanargmax = partial(_numbagg_wrapper, numbagg_func=numbagg.grouped.group_nanargmax)
# nanargmin = partial(_numbagg_wrapper, numbagg_func=numbagg.grouped.group_nanargmin)
nanmax = partial(_numbagg_wrapper, numbagg_func=numbagg.grouped.group_nanmax)
nanmin = partial(_numbagg_wrapper, numbagg_func=numbagg.grouped.group_nanmin)
any = partial(_numbagg_wrapper, numbagg_func=numbagg.grouped.group_nanany)
all = partial(_numbagg_wrapper, numbagg_func=numbagg.grouped.group_nanall)

# sum = nansum
# mean = nanmean
# sum_of_squares = nansum_of_squares
