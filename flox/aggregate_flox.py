from functools import partial

import numpy as np

from .xrutils import is_scalar, isnull, notnull


def _prepare_for_flox(group_idx, array, lexsort):
    """
    Sort the input array once to save time.
    """
    assert array.shape[-1] == group_idx.shape[0]

    if lexsort:
        # lexsort allows us to sort by label AND array value
        # numpy's quantile uses partition, which could be a big win
        # IF we can figure out how to do that.
        # This trick was snagged from scipy.ndimage.median() :)
        labels_broadcast = np.broadcast_to(group_idx, array.shape)
        idxs = np.lexsort((array, labels_broadcast), axis=-1)
        ordered_array = np.take_along_axis(array, idxs, axis=-1)
        group_idx = np.take_along_axis(group_idx, idxs[(0,) * (idxs.ndim - 1) + (...,)], axis=-1)
    else:
        issorted = (group_idx[:-1] <= group_idx[1:]).all()
        if issorted:
            ordered_array = array
        else:
            perm = group_idx.argsort(kind="stable")
            group_idx = group_idx[..., perm]
            ordered_array = array[..., perm]
    return group_idx, ordered_array


def _lerp(a, b, *, t, dtype, out=None):
    """
    COPIED from numpy.

    Compute the linear interpolation weighted by gamma on each point of
    two same shape array.

    a : array_like
        Left bound.
    b : array_like
        Right bound.
    t : array_like
        The interpolation weight.
    """
    if out is None:
        out = np.empty_like(a, dtype=dtype)
    diff_b_a = np.subtract(b, a)
    # asanyarray is a stop-gap until gh-13105
    np.add(a, diff_b_a * t, out=out)
    np.subtract(b, diff_b_a * (1 - t), out=out, where=t >= 0.5)
    return out


def quantile_(array, inv_idx, *, q, axis, skipna, dtype=None, out=None):
    inv_idx = np.concatenate((inv_idx, [array.shape[-1]]))

    if skipna:
        sizes = np.add.reduceat(notnull(array), inv_idx[:-1], axis=axis)
    else:
        newshape = (1,) * (array.ndim - 1) + (inv_idx.size - 1,)
        sizes = np.reshape(np.diff(inv_idx), newshape)
        # NaNs get sorted to the end, so look at the last element in the group to decide
        # if there are NaNs
        last_group_elem = np.broadcast_to(inv_idx[1:] - 1, newshape)
        nanmask = isnull(np.take_along_axis(array, last_group_elem, axis=axis))

    qin = q
    q = np.atleast_1d(qin)
    q = np.reshape(q, (len(q),) + (1,) * array.ndim)

    # This is numpy's method="linear"
    # TODO: could support all the interpolations here
    virtual_index = q * (sizes - 1) + inv_idx[:-1]

    is_scalar_q = is_scalar(qin)
    if is_scalar_q:
        virtual_index = virtual_index.squeeze(axis=0)
        idxshape = array.shape[:-1] + (sizes.shape[-1],)
        a_ = array
    else:
        idxshape = (q.shape[0],) + array.shape[:-1] + (sizes.shape[-1],)
        a_ = np.broadcast_to(array, (q.shape[0],) + array.shape)

    # Broadcast to (num quantiles, ..., num labels)
    lo_ = np.floor(virtual_index, casting="unsafe", out=np.empty(idxshape, dtype=np.int64))
    hi_ = np.ceil(virtual_index, casting="unsafe", out=np.empty(idxshape, dtype=np.int64))

    # get bounds
    loval = np.take_along_axis(a_, lo_, axis=axis)
    hival = np.take_along_axis(a_, hi_, axis=axis)

    # TODO: could support all the interpolations here
    gamma = np.broadcast_to(virtual_index, idxshape) - lo_
    result = _lerp(loval, hival, t=gamma, out=out, dtype=dtype)
    if not skipna and np.any(nanmask):
        result[..., nanmask] = np.nan
    return result


def _np_grouped_op(
    group_idx, array, op, axis=-1, size=None, fill_value=None, dtype=None, out=None, **kwargs
):
    """
    most of this code is from shoyer's gist
    https://gist.github.com/shoyer/f538ac78ae904c936844
    """
    # assumes input is sorted, which I do in core._prepare_for_flox
    aux = group_idx

    flag = np.concatenate((np.array([True], like=array), aux[1:] != aux[:-1]))
    uniques = aux[flag]
    (inv_idx,) = flag.nonzero()

    if size is None:
        size = np.max(uniques) + 1
    if dtype is None:
        dtype = array.dtype

    if out is None:
        q = kwargs.get("q", None)
        if q is None:
            out = np.full(array.shape[:-1] + (size,), fill_value=fill_value, dtype=dtype)
        else:
            nq = len(np.atleast_1d(q))
            out = np.full((nq,) + array.shape[:-1] + (size,), fill_value=fill_value, dtype=dtype)

    if (len(uniques) == size) and (uniques == np.arange(size, like=array)).all():
        # The previous version of this if condition
        #     ((uniques[1:] - uniques[:-1]) == 1).all():
        # does not work when group_idx is [1, 2] for e.g.
        # This happens during binning
        op(array, inv_idx, axis=axis, dtype=dtype, out=out, **kwargs)
    else:
        out[..., uniques] = op(array, inv_idx, axis=axis, dtype=dtype, **kwargs)

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


sum = partial(_np_grouped_op, op=np.add.reduceat)
nansum = partial(_nan_grouped_op, func=sum, fillna=0)
prod = partial(_np_grouped_op, op=np.multiply.reduceat)
nanprod = partial(_nan_grouped_op, func=prod, fillna=1)
max = partial(_np_grouped_op, op=np.maximum.reduceat)
nanmax = partial(_nan_grouped_op, func=max, fillna=-np.inf)
min = partial(_np_grouped_op, op=np.minimum.reduceat)
nanmin = partial(_nan_grouped_op, func=min, fillna=np.inf)
quantile = partial(_np_grouped_op, op=partial(quantile_, skipna=False))
nanquantile = partial(_np_grouped_op, op=partial(quantile_, skipna=True))
median = partial(_np_grouped_op, op=partial(quantile_, q=0.5, skipna=False))
nanmedian = partial(_np_grouped_op, op=partial(quantile_, q=0.5, skipna=True))
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
    return sum(group_idx, (notnull(array)).astype(int), *args, **kwargs)


def mean(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    if fill_value is None:
        fill_value = 0
    out = sum(group_idx, array, axis=axis, size=size, dtype=dtype, fill_value=fill_value)
    with np.errstate(invalid="ignore", divide="ignore"):
        out /= nanlen(group_idx, array, size=size, axis=axis, fill_value=0)
    return out


def nanmean(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    if fill_value is None:
        fill_value = 0
    out = nansum(group_idx, array, size=size, axis=axis, dtype=dtype, fill_value=fill_value)
    with np.errstate(invalid="ignore", divide="ignore"):
        out /= nanlen(group_idx, array, size=size, axis=axis, fill_value=0)
    return out
