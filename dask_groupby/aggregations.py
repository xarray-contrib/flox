from functools import partial

import numpy as np


def _atleast_1d(inp):
    if isinstance(inp, str):
        inp = (inp,)
    return inp


class Aggregation:
    def __init__(
        self, name, chunk, combine, aggregate=None, finalize=None, fill_value=None, dtype=None
    ):
        self.name = name
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
        self.fill_value = fill_value
        self.dtype = dtype

    def __repr__(self):
        return "\n".join(
            (
                f"{self.name}, fill: {self.fill_value}, dtype: {self.dtype}",
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
    data[data == fv] = np.nan
    return data


_min_finalize = partial(_minmax_finalize, fv=np.inf)
_max_finalize = partial(_minmax_finalize, fv=-np.inf)

min = Aggregation("min", chunk="min", combine="min", fill_value=np.inf, finalize=_min_finalize)
nanmin = Aggregation(
    "nanmin", chunk="nanmin", combine="min", fill_value=np.inf, finalize=_min_finalize
)
max = Aggregation("max", chunk="max", combine="max", fill_value=-np.inf, finalize=_max_finalize)
nanmax = Aggregation(
    "nanmax", chunk="nanmax", combine="max", fill_value=-np.inf, finalize=_max_finalize
)
