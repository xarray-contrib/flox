from functools import partial
from itertools import zip_longest

import numpy as np
from xarray.core import dtypes, utils


def _get_fill_value(dtype, fill_value):
    """ Returns dtype appropriate infinity. Returns +Inf equivalent for None."""
    if fill_value == dtypes.INF or fill_value is None:
        return dtypes.get_pos_infinity(dtype, max_for_int=True)
    if fill_value == dtypes.NINF:
        return dtypes.get_neg_infinity(dtype, min_for_int=True)
    return fill_value


def _atleast_1d(inp):
    if utils.is_scalar(inp):
        inp = (inp,)
    return inp


class Aggregation:
    def __init__(
        self,
        name,
        chunk,
        combine,
        preprocess=None,
        aggregate=None,
        finalize=None,
        fill_value=None,
        dtype=None,
        reduction_type="reduce",
    ):
        self.name = name
        # preprocess before blockwise
        self.preprocess = preprocess
        # Use "chunk_reduce" or "chunk_argreduce"
        self.reduction_type = reduction_type
        # initialize blockwise reduction
        self.chunk = _atleast_1d(chunk)
        # how to aggregate results after first round of reduction
        self.combine = _atleast_1d(combine)
        # final aggregation
        self.aggregate = aggregate if aggregate else combine
        # finalize results (see mean)
        self.finalize = finalize if finalize else lambda x: x
        # fill_value is used to reindex to expected_groups.
        # They should make sense when aggregated together with results from other blocks
        fill_value = _atleast_1d(fill_value)
        self.fill_value = dict(zip_longest(self.chunk, fill_value, fillvalue=fill_value[0]))
        self.fill_value.update(dict(zip_longest(self.combine, fill_value, fillvalue=fill_value[0])))
        self.fill_value.update(
            dict(zip_longest(self.aggregate, fill_value, fillvalue=fill_value[0]))
        )
        if self.name not in self.fill_value:
            self.fill_value.update({self.name: fill_value[0]})
        # np.dtype(None) = np.dtype("float64")!
        self.dtype = dtype

    def __repr__(self):
        return "\n".join(
            (
                f"{self.name}, fill: {np.unique(self.fill_value.values())}, dtype: {self.dtype}",
                f"chunk: {self.chunk}",
                f"combine: {self.combine}",
                f"aggregate: {self.aggregate}" f"finalize: {self.finalize}",
            )
        )


def sum_of_squares(array, axis=-1):
    return np.sum(array ** 2, axis)


def nansum_of_squares(array, axis=-1):
    return np.nansum(array ** 2, axis)


count = Aggregation("count", chunk="count", combine="sum", fill_value=0, dtype=int)
sum = Aggregation("sum", chunk="sum", combine="sum", fill_value=0)
nansum = Aggregation("nansum", chunk="nansum", combine="sum", fill_value=0)
prod = Aggregation("prod", chunk="prod", combine="prod", fill_value=1)
nanprod = Aggregation("nanprod", chunk="nanprod", combine="prod", fill_value=1)
mean = Aggregation(
    "mean",
    chunk=("sum", "count"),
    combine=("sum", "sum"),
    finalize=lambda sum_, count: sum_ / count,
    fill_value=0,
)
nanmean = Aggregation(
    "nanmean",
    chunk=("nansum", "count"),
    combine=("sum", "sum"),
    finalize=lambda sum_, count: sum_ / count,
    fill_value=0,
)


# TODO: fix this for complex numbers
def _var_finalize(sumsq, sum_, count, ddof=0):
    result = (sumsq - (sum_ ** 2 / count)) / (count - ddof)
    result[(count - ddof) <= 0] = np.nan
    return result


def _std_finalize(sumsq, sum_, count, ddof=0):
    return np.sqrt(_var_finalize(sumsq, sum_, count, ddof))


var = Aggregation(
    "var",
    chunk=(sum_of_squares, "sum", "count"),
    combine=("sum", "sum", "sum"),
    finalize=_var_finalize,
    fill_value=0,
)
nanvar = Aggregation(
    "nanvar",
    chunk=(nansum_of_squares, "count"),
    combine=("sum", "sum"),
    finalize=_var_finalize,
    fill_value=0,
)
std = Aggregation(
    "std",
    chunk=(sum_of_squares, "sum", "count"),
    combine=("sum", "sum", "sum"),
    finalize=_std_finalize,
    fill_value=0,
)
nanstd = Aggregation(
    "nanstd",
    chunk=(nansum_of_squares, "count"),
    combine=("sum", "sum"),
    finalize=_std_finalize,
    fill_value=0,
)


# TODO: How does this work if data has fillvalue?
def _minmax_finalize(data, fv):

    fv = _get_fill_value(data.dtype, fv)
    data[data == fv] = np.nan

    # TODO: workaround npg bug
    # aggregate(np.array([0, 1, 2, 0, 1, 2]), np.array([dtypes.NINF, 0, 0, dtypes.NINF, 0, 0]), func="max",)
    # is array([-1.79769313e+308,  0.00000000e+000,  0.00000000e+000])
    # instead of array[-inf, 0.0, 0.0]
    #
    if np.isneginf(fv):
        fv = np.finfo(data.dtype).min
    elif np.isposinf(fv):
        fv = np.finfo(data.dtype).max
    data[data == fv] = np.nan
    return data


_min_finalize = None  # partial(_minmax_finalize, fv=dtypes.INF)
_max_finalize = None  # partial(_minmax_finalize, fv=dtypes.NINF)

min = Aggregation("min", chunk="min", combine="min", fill_value=dtypes.INF, finalize=_min_finalize)
nanmin = Aggregation(
    "nanmin", chunk="nanmin", combine="min", fill_value=dtypes.INF, finalize=_min_finalize
)
max = Aggregation("max", chunk="max", combine="max", fill_value=dtypes.NINF, finalize=_max_finalize)
nanmax = Aggregation(
    "nanmax", chunk="nanmax", combine="max", fill_value=dtypes.NINF, finalize=_max_finalize
)


def argreduce_preprocess(array, axis):
    """Returns a tuple of array, index along axis.

    Copied from dask.array.chunk.argtopk_preprocess
    """
    import dask.array
    import numpy as np

    # TODO: arg reductions along multiple axes seems weird.
    assert len(axis) == 1
    axis = axis[0]

    idx = dask.array.arange(array.shape[axis], chunks=array.chunks[axis], dtype=np.intp)
    # broadcast (TODO: is this needed?)
    idx = idx[tuple(slice(None) if i == axis else np.newaxis for i in range(array.ndim))]

    def _zip_index(array_, idx_):
        return (array_, idx_)

    return dask.array.map_blocks(
        _zip_index,
        array,
        idx,
        dtype=array.dtype,
        meta=array._meta,
        name="groupby-argreduce-preprocess",
    )


def argreduce_finalize(*args):
    # TODO: clean this up :/
    if len(args) > 1:
        # dask inputs
        return args[1]
    else:
        # numpy inputs
        return args[0]


argmax = Aggregation(
    "argmax",
    preprocess=argreduce_preprocess,
    chunk=("max", "argmax"),  # order is important
    combine=("max", "argmax"),
    reduction_type="argreduce",
    fill_value=(dtypes.NINF, 0),
    finalize=argreduce_finalize,
)

argmin = Aggregation(
    "argmin",
    preprocess=argreduce_preprocess,
    chunk=("min", "argmin"),  # order is important
    combine=("min", "argmin"),
    reduction_type="argreduce",
    fill_value=(dtypes.INF, 0),
    finalize=argreduce_finalize,
)

nanargmax = Aggregation(
    "nanargmax",
    preprocess=argreduce_preprocess,
    chunk=("nanmax", "nanargmax"),  # order is important
    combine=("max", "argmax"),
    reduction_type="argreduce",
    fill_value=(dtypes.NINF, 0),
    finalize=argreduce_finalize,
)

nanargmin = Aggregation(
    "nanargmin",
    preprocess=argreduce_preprocess,
    chunk=("nanmin", "nanargmin"),  # order is important
    combine=("min", "argmin"),
    reduction_type="argreduce",
    fill_value=(dtypes.INF, 0),
    finalize=argreduce_finalize,
)

first = Aggregation("first", chunk="first", combine="first", fill_value=np.nan)
last = Aggregation("last", chunk="last", combine="last", fill_value=np.nan)
# all
# any
# median - should be doable since dask implements t-digest percentile for 1D?
