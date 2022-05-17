from functools import partial

import numpy as np

from .xrutils import isnull


def _np_grouped_op(group_idx, array, op, axis=-1, size=None, fill_value=None, dtype=None, out=None):
    """
    most of this code is from shoyer's gist
    https://gist.github.com/shoyer/f538ac78ae904c936844
    """
    # assumes input is sorted, which I do in core._prepare_for_flox
    aux = group_idx

    flag = np.concatenate(([True], aux[1:] != aux[:-1]))
    uniques = aux[flag]
    (inv_idx,) = flag.nonzero()

    if size is None:
        size = np.max(uniques) + 1
    if dtype is None:
        dtype = array.dtype

    if out is None:
        out = np.full(array.shape[:-1] + (size,), fill_value=fill_value, dtype=dtype)

    if (len(uniques) == size) and (uniques == np.arange(size)).all():
        # The previous version of this if condition
        #     ((uniques[1:] - uniques[:-1]) == 1).all():
        # does not work when group_idx is [1, 2] for e.g.
        # This happens  during binning
        op.reduceat(array, inv_idx, axis=axis, dtype=dtype, out=out)
    else:
        out[..., uniques] = op.reduceat(array, inv_idx, axis=axis, dtype=dtype)

    return out


def _nan_grouped_op(group_idx, array, func, fillna, *args, **kwargs):
    result = func(group_idx, np.where(isnull(array), fillna, array), *args, **kwargs)
    # np.nanmax([np.nan, np.nan]) = np.nan
    # To recover this behaviour, we need to search for the fillna value
    # (either np.inf or -np.inf), and replace with NaN
    # Our choice of fillna does the right thing for sum, prod
    if fillna in (np.inf, -np.inf):
        allnangroups = result == fillna
        if allnangroups.any():
            result[allnangroups] = kwargs["fill_value"]
    return result


sum = partial(_np_grouped_op, op=np.add)
nansum = partial(_nan_grouped_op, func=sum, fillna=0)
prod = partial(_np_grouped_op, op=np.multiply)
nanprod = partial(_nan_grouped_op, func=prod, fillna=1)
max = partial(_np_grouped_op, op=np.maximum)
nanmax = partial(_nan_grouped_op, func=max, fillna=-np.inf)
min = partial(_np_grouped_op, op=np.minimum)
nanmin = partial(_nan_grouped_op, func=min, fillna=np.inf)
# TODO: all, any


def sum_of_squares(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):

    return sum(
        group_idx,
        array**2,
        axis=axis,
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def nansum_of_squares(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    return sum_of_squares(
        group_idx,
        np.where(isnull(array), 0, array),
        size=size,
        fill_value=fill_value,
        axis=axis,
        dtype=dtype,
    )


def nanlen(group_idx, array, *args, **kwargs):
    return sum(group_idx, (~isnull(array)).astype(int), *args, **kwargs)


def mean(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    if fill_value is None:
        fill_value = 0
    out = np.full(array.shape[:-1] + (size,), fill_value=fill_value, dtype=dtype)
    sum(group_idx, array, axis=axis, size=size, dtype=dtype, out=out)
    out /= nanlen(group_idx, array, size=size, axis=axis, fill_value=0)
    return out


def nanmean(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    if fill_value is None:
        fill_value = 0
    out = np.full(array.shape[:-1] + (size,), fill_value=fill_value, dtype=dtype)
    nansum(group_idx, array, size=size, axis=axis, dtype=dtype, out=out)
    out /= nanlen(group_idx, array, size=size, axis=axis, fill_value=0)
    return out
