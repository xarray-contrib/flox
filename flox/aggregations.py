import numpy as np

from . import xrdtypes as dtypes, xrutils


def _get_fill_value(dtype, fill_value):
    """Returns dtype appropriate infinity. Returns +Inf equivalent for None."""
    if fill_value == dtypes.INF or fill_value is None:
        return dtypes.get_pos_infinity(dtype, max_for_int=True)
    if fill_value == dtypes.NINF:
        return dtypes.get_neg_infinity(dtype, min_for_int=True)
    if fill_value == dtypes.NA:
        if np.issubdtype(dtype, np.floating):
            return np.nan
        # This is madness, but npg checks that fill_value is compatible
        # with array dtype even if the fill_value is never used.
        elif np.issubdtype(dtype, np.integer):
            return dtypes.get_neg_infinity(dtype, min_for_int=True)
        else:
            return None
    return fill_value


def _atleast_1d(inp):
    if xrutils.is_scalar(inp):
        inp = (inp,)
    return inp


class Aggregation:
    def __init__(
        self,
        name,
        *,
        chunk,
        combine,
        numpy=None,
        preprocess=None,
        aggregate=None,
        finalize=None,
        fill_value=None,
        final_fill_value=dtypes.NA,
        dtype=None,
        reduction_type="reduce",
    ):
        self.name = name
        # preprocess before blockwise
        self.preprocess = preprocess
        # Use "chunk_reduce" or "chunk_argreduce"
        self.reduction_type = reduction_type
        self.numpy = numpy if numpy else self.name
        # initialize blockwise reduction
        self.chunk = _atleast_1d(chunk)
        # how to aggregate results after first round of reduction
        self.combine = _atleast_1d(combine)
        # final aggregation
        self.aggregate = aggregate if aggregate else self.combine[0]
        # finalize results (see mean)
        self.finalize = finalize if finalize else lambda x: x

        self.fill_value = {}

        # This is used for the final reindexing
        self.fill_value[name] = final_fill_value

        # Aggregation.fill_value is used to reindex to group labels
        # at the *intermediate* step.
        # They should make sense when aggregated together with results from other blocks
        fill_value = _atleast_1d(fill_value)
        if len(fill_value) == 1 and len(fill_value) < len(self.chunk):
            fill_value = fill_value * len(self.chunk)
        if len(fill_value) != len(self.chunk):
            raise ValueError(f"Bad fill_value specified for Aggregation {name}.")
        self.fill_value["intermediate"] = fill_value

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


def _nansum(group_idx, array, size=None, fill_value=None):
    # npg takes out NaNs before calling np.bincount
    # This means that all NaN groups are equivalent to absent groups
    # This behaviour does not work for xarray
    import numpy_groupies as npg

    return npg.aggregate_numpy.aggregate(
        group_idx,
        np.where(np.isnan(array), 0, array),
        axis=-1,
        func="sum",
        size=size,
        fill_value=fill_value,
    )


def _nanprod(group_idx, array, size=None, fill_value=None):
    # npg takes out NaNs before calling np.bincount
    # This means that all NaN groups are equivalent to absent groups
    # This behaviour does not work for xarray
    import numpy_groupies as npg

    return npg.aggregate_numpy.aggregate(
        group_idx,
        np.where(np.isnan(array), 1, array),
        axis=-1,
        func="prod",
        size=size,
        fill_value=fill_value,
    )


def nansum_of_squares(group_idx, array, size=None, fill_value=None):
    return sum_of_squares(group_idx, array, func="nansum", size=size, fill_value=fill_value)


count = Aggregation(
    "count",
    numpy="nanlen",
    chunk="nanlen",
    combine="sum",
    fill_value=0,
    final_fill_value=0,
    dtype=np.intp,
)

# note that the fill values are the result of np.func([np.nan, np.nan])
# final_fill_value is used for groups that don't exist. This is usually np.nan
sum = Aggregation("sum", chunk="sum", combine="sum", fill_value=0)
nansum = Aggregation("nansum", chunk=_nansum, numpy=_nansum, combine="sum", fill_value=0)
prod = Aggregation("prod", chunk="prod", combine="prod", fill_value=1, final_fill_value=1)
nanprod = Aggregation(
    "nanprod",
    numpy=_nanprod,
    chunk=_nanprod,
    combine="prod",
    fill_value=1,
    final_fill_value=dtypes.NA,
)
mean = Aggregation(
    "mean",
    chunk=("sum", "nanlen"),
    combine=("sum", "sum"),
    finalize=lambda sum_, count: sum_ / count,
    fill_value=(0, 0),
    dtype=np.floating,
)
nanmean = Aggregation(
    "nanmean",
    chunk=("nansum", "nanlen"),
    combine=("sum", "sum"),
    finalize=lambda sum_, count: sum_ / count,
    fill_value=(0, 0),
    dtype=np.floating,
)


# TODO: fix this for complex numbers
def _var_finalize(sumsq, sum_, count, ddof=0):
    result = (sumsq - (sum_ ** 2 / count)) / (count - ddof)
    result[count <= ddof] = np.nan
    return result


def _std_finalize(sumsq, sum_, count, ddof=0):
    return np.sqrt(_var_finalize(sumsq, sum_, count, ddof))


# var, std always promote to float, so we set nan
var = Aggregation(
    "var",
    chunk=(sum_of_squares, "sum", "nanlen"),
    combine=("sum", "sum", "sum"),
    finalize=_var_finalize,
    fill_value=0,
    final_fill_value=np.nan,
    dtype=np.floating,
)
nanvar = Aggregation(
    "nanvar",
    chunk=(nansum_of_squares, "nansum", "nanlen"),
    combine=("sum", "sum", "sum"),
    finalize=_var_finalize,
    fill_value=0,
    final_fill_value=np.nan,
    dtype=np.floating,
)
std = Aggregation(
    "std",
    chunk=(sum_of_squares, "sum", "nanlen"),
    combine=("sum", "sum", "sum"),
    finalize=_std_finalize,
    fill_value=0,
    final_fill_value=np.nan,
    dtype=np.floating,
)
nanstd = Aggregation(
    "nanstd",
    chunk=(nansum_of_squares, "nansum", "nanlen"),
    combine=("sum", "sum", "sum"),
    finalize=_std_finalize,
    fill_value=0,
    final_fill_value=np.nan,
    dtype=np.floating,
)


min = Aggregation("min", chunk="min", combine="min", fill_value=dtypes.INF)
nanmin = Aggregation("nanmin", chunk="nanmin", combine="min", fill_value=dtypes.INF)
max = Aggregation("max", chunk="max", combine="max", fill_value=dtypes.NINF)
nanmax = Aggregation("nanmax", chunk="nanmax", combine="max", fill_value=dtypes.NINF)


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
    final_fill_value=-1,
    finalize=lambda *x: x[1],
    dtype=np.intp,
)

argmin = Aggregation(
    "argmin",
    preprocess=argreduce_preprocess,
    chunk=("min", "argmin"),  # order is important
    combine=("min", "argmin"),
    reduction_type="argreduce",
    fill_value=(dtypes.INF, 0),
    final_fill_value=-1,
    finalize=lambda *x: x[1],
    dtype=np.intp,
)

nanargmax = Aggregation(
    "nanargmax",
    preprocess=argreduce_preprocess,
    chunk=("nanmax", "nanargmax"),  # order is important
    combine=("max", "argmax"),
    reduction_type="argreduce",
    fill_value=(dtypes.NINF, -1),
    final_fill_value=-1,
    finalize=lambda *x: x[1],
    dtype=np.intp,
)

nanargmin = Aggregation(
    "nanargmin",
    preprocess=argreduce_preprocess,
    chunk=("nanmin", "nanargmin"),  # order is important
    combine=("min", "argmin"),
    reduction_type="argreduce",
    fill_value=(dtypes.INF, -1),
    final_fill_value=-1,
    finalize=lambda *x: x[1],
    dtype=np.intp,
)

first = Aggregation("first", chunk="first", combine="first", fill_value=0)
last = Aggregation("last", chunk="last", combine="last", fill_value=0)
nanfirst = Aggregation("nanfirst", chunk="nanfirst", combine="nanfirst", fill_value=np.nan)
nanlast = Aggregation("nanlast", chunk="nanlast", combine="nanlast", fill_value=np.nan)

all = Aggregation(
    "all", chunk="all", combine="all", fill_value=True, final_fill_value=False, dtype=bool
)
any = Aggregation(
    "any", chunk="any", combine="any", fill_value=False, final_fill_value=False, dtype=bool
)

# numpy_groupies does not support median
# And the dask version is really hard!
# median = Aggregation("median", chunk=None, combine=None, fill_value=None)
# nanmedian = Aggregation("nanmedian", chunk=None, combine=None, fill_value=None)
