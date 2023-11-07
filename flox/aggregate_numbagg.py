from functools import partial

import numbagg
import numbagg.grouped
import numpy as np

DEFAULT_FILL_VALUE = {
    "nansum": 0,
    "nanmean": np.nan,
    "nanvar": np.nan,
    "nanstd": np.nan,
    "nanmin": np.nan,
    "nanmax": np.nan,
    "nanany": False,
    "nanall": False,
    "nansum_of_squares": 0,
    "nanprod": 1,
    "nancount": 0,
    "nanargmax": np.nan,
    "nanargmin": np.nan,
    "nanfirst": np.nan,
    "nanlast": np.nan,
}

CAST_TO = {
    "nansum": {np.bool_: np.int64},
    "nanmean": {np.int_: np.float64},
    "nanvar": {np.int_: np.float64},
    "nanstd": {np.int_: np.float64},
}


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
    cast_to = CAST_TO.get(numbagg_func, None)
    if cast_to:
        for from_, to_ in cast_to.items():
            if isinstance(array, from_):
                array = array.astype(to_)

    func_ = getattr(numbagg.grouped, f"group_{numbagg_func}")
    result = func_(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        # The following are unsupported
        # fill_value=fill_value,
        # dtype=dtype,
    )

    default_fv = DEFAULT_FILL_VALUE[numbagg_func]
    if fill_value is not None and fill_value != default_fv:
        count = numbagg.grouped.group_nancount(array, group_idx, axis=axis, num_labels=size)
        result[count == 0] = fill_value
    return result


def nanvar(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None, ddof=0):
    assert ddof != 0

    return _numbagg_wrapper(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        numbagg_func="nanvar"
        # ddof=0,
        # fill_value=fill_value,
        # dtype=dtype,
    )


def nanstd(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None, ddof=0):
    assert ddof != 0

    return numbagg.grouped.group_nanstd(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        # ddof=0,
        # fill_value=fill_value,
        # dtype=dtype,
    )


nansum = partial(_numbagg_wrapper, numbagg_func="nansum")
nanmean = partial(_numbagg_wrapper, numbagg_func="nanmean")
nanprod = partial(_numbagg_wrapper, numbagg_func="nanprod")
nansum_of_squares = partial(_numbagg_wrapper, numbagg_func="nansum_of_squares")
nanlen = partial(_numbagg_wrapper, numbagg_func="nancount")
nanprod = partial(_numbagg_wrapper, numbagg_func="nanprod")
nanfirst = partial(_numbagg_wrapper, numbagg_func="nanfirst")
nanlast = partial(_numbagg_wrapper, numbagg_func="nanlast")
# nanargmax = partial(_numbagg_wrapper, numbagg_func="nanargmax)
# nanargmin = partial(_numbagg_wrapper, numbagg_func="nanargmin)
nanmax = partial(_numbagg_wrapper, numbagg_func="nanmax")
nanmin = partial(_numbagg_wrapper, numbagg_func="nanmin")
any = partial(_numbagg_wrapper, numbagg_func="nanany")
all = partial(_numbagg_wrapper, numbagg_func="nanall")

# sum = nansum
# mean = nanmean
# sum_of_squares = nansum_of_squares
