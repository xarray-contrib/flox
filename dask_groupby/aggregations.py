# from functools import partial
from itertools import zip_longest

import numpy as np
from xarray.core import dtypes, utils


def _get_fill_value(dtype, fill_value):
    """Returns dtype appropriate infinity. Returns +Inf equivalent for None."""
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
        self.aggregate = aggregate if aggregate else self.combine[0]
        # finalize results (see mean)
        self.finalize = finalize if finalize else lambda x: x

        # fill_value is used to reindex to group labels
        # They should make sense when aggregated together with results from other blocks
        fill_value = _atleast_1d(fill_value)
        self.fill_value = dict(zip_longest(self.chunk, fill_value, fillvalue=fill_value[0]))
        self.fill_value.update(dict(zip_longest(self.combine, fill_value, fillvalue=fill_value[0])))
        self.fill_value.update(dict(zip((self.aggregate,), fill_value)))
        if self.name not in self.fill_value:
            self.fill_value.update({self.name: fill_value[0]})

        # np.dtype(None) = np.dtype("float64")!
        self.dtype = dtype

    def __dask_tokenize__(self):
        return (
            Aggregation,
            self.name,
            self.preprocess,
            self.reduction_type,
            self.chunk,
            self.combine,
            self.aggregate,
            self.finalize,
            self.fill_value,
            self.dtype,
        )

    def __repr__(self):
        return "\n".join(
            (
                f"{self.name}, fill: {np.unique(self.fill_value.values())}, dtype: {self.dtype}",
                f"chunk: {self.chunk}",
                f"combine: {self.combine}",
                f"aggregate: {self.aggregate}",
                f"finalize: {self.finalize}",
            )
        )


def sum_of_squares(group_idx, array, func="sum", size=None, fill_value=None):
    import numpy_groupies as npg

    return npg.aggregate_numpy.aggregate(
        group_idx,
        array ** 2,
        axis=-1,
        func=func,
        size=size,
        fill_value=fill_value,
    )


def nansum_of_squares(group_idx, array, size=None, fill_value=None):
    return sum_of_squares(group_idx, array, func="nansum", size=size, fill_value=fill_value)


def _count(group_idx, array, size=None, fill_value=None):
    import numpy_groupies as npg

    return npg.aggregate_numpy.aggregate(
        group_idx,
        (~np.isnan(array)).astype(int),
        axis=-1,
        func="sum",
        size=size,
        fill_value=fill_value,
    )


count = Aggregation("count", chunk=_count, combine="sum", fill_value=0, dtype=int)

# note that the fill values are  the result of np.func([np.nan, np.nan])
sum = Aggregation("sum", chunk="sum", combine="sum", fill_value=0)
nansum = Aggregation("nansum", chunk="nansum", combine="sum", fill_value=0)
prod = Aggregation("prod", chunk="prod", combine="prod", fill_value=1)
nanprod = Aggregation("nanprod", chunk="nanprod", combine="prod", fill_value=1)
mean = Aggregation(
    "mean",
    chunk=("sum", _count),
    combine=("sum", "sum"),
    finalize=lambda sum_, count: sum_ / count,
    fill_value=(dtypes.NA, 0),
)
nanmean = Aggregation(
    "nanmean",
    chunk=("nansum", _count),
    combine=("sum", "sum"),
    finalize=lambda sum_, count: sum_ / count,
    fill_value=(dtypes.NA, 0),
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
    chunk=(sum_of_squares, "sum", _count),
    combine=("sum", "sum", "sum"),
    finalize=_var_finalize,
    fill_value=0,
)
nanvar = Aggregation(
    "nanvar",
    chunk=(nansum_of_squares, "nansum", _count),
    combine=("sum", "sum", "sum"),
    finalize=_var_finalize,
    fill_value=0,
)
std = Aggregation(
    "std",
    chunk=(sum_of_squares, "sum", _count),
    combine=("sum", "sum", "sum"),
    finalize=_std_finalize,
    fill_value=0,
)
nanstd = Aggregation(
    "nanstd",
    chunk=(nansum_of_squares, "nansum", _count),
    combine=("sum", "sum", "sum"),
    finalize=_std_finalize,
    fill_value=0,
)


min = Aggregation("min", chunk="min", combine="min", fill_value=dtypes.INF, finalize=None)
nanmin = Aggregation("nanmin", chunk="nanmin", combine="min", fill_value=dtypes.INF, finalize=None)
max = Aggregation("max", chunk="max", combine="max", fill_value=dtypes.NINF, finalize=None)
nanmax = Aggregation("nanmax", chunk="nanmax", combine="max", fill_value=dtypes.NINF, finalize=None)


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


argmax = Aggregation(
    "argmax",
    preprocess=argreduce_preprocess,
    chunk=("max", "argmax"),  # order is important
    combine=("max", "argmax"),
    reduction_type="argreduce",
    fill_value=(dtypes.NINF, 0),
    finalize=lambda *x: x[1],
    dtype=np.int,
)

argmin = Aggregation(
    "argmin",
    preprocess=argreduce_preprocess,
    chunk=("min", "argmin"),  # order is important
    combine=("min", "argmin"),
    reduction_type="argreduce",
    fill_value=(dtypes.INF, 0),
    finalize=lambda *x: x[1],
    dtype=np.int,
)

nanargmax = Aggregation(
    "nanargmax",
    preprocess=argreduce_preprocess,
    chunk=("nanmax", "nanargmax"),  # order is important
    combine=("max", "argmax"),
    reduction_type="argreduce",
    fill_value=(dtypes.NINF, 0),
    finalize=lambda *x: x[1],
    dtype=np.int,
)

nanargmin = Aggregation(
    "nanargmin",
    preprocess=argreduce_preprocess,
    chunk=("nanmin", "nanargmin"),  # order is important
    combine=("min", "argmin"),
    reduction_type="argreduce",
    fill_value=(dtypes.INF, 0),
    finalize=lambda *x: x[1],
    dtype=np.int,
)

first = Aggregation("first", chunk="first", combine="first", fill_value=0)
last = Aggregation("last", chunk="last", combine="last", fill_value=0)
nanfirst = Aggregation("nanfirst", chunk="nanfirst", combine="nanfirst", fill_value=np.nan)
nanlast = Aggregation("nanlast", chunk="nanlast", combine="nanlast", fill_value=np.nan)
# all
# any
# median - should be doable since dask implements t-digest percentile for 1D?
