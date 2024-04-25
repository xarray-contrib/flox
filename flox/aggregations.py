from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from . import aggregate_flox, aggregate_npg, xrutils
from . import xrdtypes as dtypes

if TYPE_CHECKING:
    FuncTuple = tuple[Callable | str, ...]
    OptionalFuncTuple = tuple[Callable | str | None, ...]


logger = logging.getLogger("flox")


def _is_arg_reduction(func: str | Aggregation) -> bool:
    if isinstance(func, str) and func in ["argmin", "argmax", "nanargmax", "nanargmin"]:
        return True
    if isinstance(func, Aggregation) and func.reduction_type == "argreduce":
        return True
    return False


class AggDtypeInit(TypedDict):
    final: DTypeLike | None
    intermediate: tuple[DTypeLike, ...]


class AggDtype(TypedDict):
    user: DTypeLike | None
    final: np.dtype
    numpy: tuple[np.dtype | type[np.intp], ...]
    intermediate: tuple[np.dtype | type[np.intp], ...]


def get_npg_aggregation(func, *, engine):
    try:
        method_ = getattr(aggregate_npg, func)
        method = partial(method_, engine=engine)
    except AttributeError:
        aggregate = aggregate_npg._get_aggregate(engine).aggregate
        method = partial(aggregate, func=func)
    return method


def generic_aggregate(
    group_idx,
    array,
    *,
    engine: str,
    func: str,
    axis=-1,
    size=None,
    fill_value=None,
    dtype=None,
    **kwargs,
):
    if engine == "flox":
        try:
            method = getattr(aggregate_flox, func)
        except AttributeError:
            logger.debug(f"Couldn't find {func} for engine='flox'. Falling back to numpy")
            method = get_npg_aggregation(func, engine="numpy")

    elif engine == "numbagg":
        from . import aggregate_numbagg

        try:
            if "var" in func or "std" in func:
                ddof = kwargs.get("ddof", 0)
                if aggregate_numbagg.NUMBAGG_SUPPORTS_DDOF or (ddof != 0):
                    method = getattr(aggregate_numbagg, func)
                else:
                    logger.debug(f"numbagg too old for ddof={ddof}. Falling back to numpy")
                    method = get_npg_aggregation(func, engine="numpy")
            else:
                method = getattr(aggregate_numbagg, func)

        except AttributeError:
            logger.debug(f"Couldn't find {func} for engine='numbagg'. Falling back to numpy")
            method = get_npg_aggregation(func, engine="numpy")

    elif engine in ["numpy", "numba"]:
        method = get_npg_aggregation(func, engine=engine)

    else:
        raise ValueError(
            f"Expected engine to be one of ['flox', 'numpy', 'numba', 'numbagg']. Received {engine} instead."
        )

    group_idx = np.asarray(group_idx, like=array)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        result = method(
            group_idx, array, axis=axis, size=size, fill_value=fill_value, dtype=dtype, **kwargs
        )
    return result


def _normalize_dtype(dtype: DTypeLike, array_dtype: np.dtype, fill_value=None) -> np.dtype:
    if dtype is None:
        dtype = array_dtype
    if dtype is np.floating:
        # mean, std, var always result in floating
        # but we preserve the array's dtype if it is floating
        if array_dtype.kind in "fcmM":
            dtype = array_dtype
        else:
            dtype = np.dtype("float64")
    elif not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    if fill_value not in [None, dtypes.INF, dtypes.NINF, dtypes.NA]:
        dtype = np.result_type(dtype, fill_value)
    return dtype


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


def _atleast_1d(inp, min_length: int = 1):
    if xrutils.is_scalar(inp):
        inp = (inp,) * min_length
    assert len(inp) >= min_length
    return inp


def returns_empty_tuple(*args, **kwargs):
    return ()


@dataclass
class Dim:
    values: ArrayLike
    name: str | None

    @cached_property
    def is_scalar(self) -> bool:
        return xrutils.is_scalar(self.values)

    @cached_property
    def size(self) -> int:
        return 0 if self.is_scalar else len(self.values)  # type: ignore[arg-type]


class Aggregation:
    def __init__(
        self,
        name,
        *,
        numpy: str | FuncTuple | None = None,
        chunk: str | FuncTuple | None,
        combine: str | FuncTuple | None,
        preprocess: Callable | None = None,
        finalize: Callable | None = None,
        fill_value=None,
        final_fill_value=dtypes.NA,
        dtypes=None,
        final_dtype: DTypeLike | None = None,
        reduction_type: Literal["reduce", "argreduce"] = "reduce",
        new_dims_func: Callable | None = None,
    ):
        """
        Blueprint for computing grouped aggregations.

        See aggregations.py for examples on how to specify reductions.

        Attributes
        ----------
        name : str
            Name of reduction.
        numpy : str or callable, optional
            Reduction function applied to numpy inputs. This function should
            compute the grouped reduction and must have a specific signature.
            If string, these must be "native" reductions implemented by the backend
            engines (numpy_groupies, flox, numbagg). If None, will be set to ``name``.
        chunk : None or str or tuple of str or callable or tuple of callable
            For dask inputs only. Either a single function or a list of
            functions to be applied blockwise on the input dask array. If None, will raise
            an error for dask inputs.
        combine : None or str or tuple of str or callbe or tuple of callable
            For dask inputs only. Functions applied when combining intermediate
            results from the blockwise stage (see ``chunk``). If None, will raise an error
            for dask inputs.
        finalize : callable
            For dask inputs only. Function that combines intermediate results to compute
            final result.
        preprocess : callable
            For dask inputs only. Preprocess inputs before ``chunk`` stage.
        reduction_type : {"reduce", "argreduce"}
            Type of reduction.
        fill_value : number or tuple(number), optional
            Value to use when a group has no members. If single value will be converted
            to tuple of same length as chunk. If appropriate, provide a different fill_value
            per reduction in ``chunk`` as a tuple.
        final_fill_value : optional
            fill_value for final result.
        dtypes : DType or tuple(DType), optional
            dtypes for intermediate results. If single value, will be converted to a tuple
            of same length as chunk. If appropriate, provide a different fill_value
            per reduction in ``chunk`` as a tuple.
        final_dtype : DType, optional
            DType for output. By default, uses dtype of array being reduced.
        new_dims_func: Callable
            Function that receives finalize_kwargs and returns a tupleof sizes of any new dimensions
            added by the reduction. For e.g. quantile for q=(0.5, 0.85) adds a new dimension of size 2,
            so returns (2,)
        """
        self.name = name
        # preprocess before blockwise
        self.preprocess = preprocess
        # Use "chunk_reduce" or "chunk_argreduce"
        self.reduction_type = reduction_type
        self.numpy: FuncTuple = (numpy,) if numpy else (self.name,)
        # initialize blockwise reduction
        self.chunk: OptionalFuncTuple = _atleast_1d(chunk)
        # how to aggregate results after first round of reduction
        self.combine: OptionalFuncTuple = _atleast_1d(combine)
        # simpler reductions used with the "simple combine" algorithm
        self.simple_combine: OptionalFuncTuple = ()
        # finalize results (see mean)
        self.finalize: Callable | None = finalize

        self.fill_value = {}
        # This is used for the final reindexing
        self.fill_value[name] = final_fill_value
        # Aggregation.fill_value is used to reindex to group labels
        # at the *intermediate* step.
        # They should make sense when aggregated together with results from other blocks
        self.fill_value["intermediate"] = self._normalize_dtype_fill_value(fill_value, "fill_value")

        self.dtype_init: AggDtypeInit = {
            "final": final_dtype,
            "intermediate": self._normalize_dtype_fill_value(dtypes, "dtype"),
        }
        self.dtype: AggDtype = None  # type: ignore[assignment]

        # The following are set by _initialize_aggregation
        self.finalize_kwargs: dict[Any, Any] = {}
        self.min_count: int = 0
        self.new_dims_func: Callable = (
            returns_empty_tuple if new_dims_func is None else new_dims_func
        )

    @cached_property
    def new_dims(self) -> tuple[Dim]:
        return self.new_dims_func(**self.finalize_kwargs)

    @cached_property
    def num_new_vector_dims(self) -> int:
        return len(tuple(dim for dim in self.new_dims if not dim.is_scalar))

    def _normalize_dtype_fill_value(self, value, name):
        value = _atleast_1d(value)
        if len(value) == 1 and len(value) < len(self.chunk):
            value = value * len(self.chunk)
        if len(value) != len(self.chunk):
            raise ValueError(f"Bad {name} specified for Aggregation {name}.")
        return value

    def __dask_tokenize__(self):
        return (
            Aggregation,
            self.name,
            self.preprocess,
            self.reduction_type,
            self.numpy,
            self.chunk,
            self.combine,
            self.finalize,
            self.fill_value,
            self.dtype,
        )

    def __repr__(self) -> str:
        return "\n".join(
            (
                f"{self.name!r}, fill: {self.fill_value.values()!r}, dtype: {self.dtype}",
                f"chunk: {self.chunk!r}",
                f"combine: {self.combine!r}",
                f"finalize: {self.finalize!r}",
                f"min_count: {self.min_count!r}",
            )
        )


count = Aggregation(
    "count",
    numpy="nanlen",
    chunk="nanlen",
    combine="sum",
    fill_value=0,
    final_fill_value=0,
    dtypes=np.intp,
    final_dtype=np.intp,
)

# note that the fill values are the result of np.func([np.nan, np.nan])
# final_fill_value is used for groups that don't exist. This is usually np.nan
sum_ = Aggregation("sum", chunk="sum", combine="sum", fill_value=0)
nansum = Aggregation("nansum", chunk="nansum", combine="sum", fill_value=0)
prod = Aggregation("prod", chunk="prod", combine="prod", fill_value=1, final_fill_value=1)
nanprod = Aggregation("nanprod", chunk="nanprod", combine="prod", fill_value=1)


def _mean_finalize(sum_, count):
    with np.errstate(invalid="ignore", divide="ignore"):
        return sum_ / count


mean = Aggregation(
    "mean",
    chunk=("sum", "nanlen"),
    combine=("sum", "sum"),
    finalize=_mean_finalize,
    fill_value=(0, 0),
    dtypes=(None, np.intp),
    final_dtype=np.floating,
)
nanmean = Aggregation(
    "nanmean",
    chunk=("nansum", "nanlen"),
    combine=("sum", "sum"),
    finalize=_mean_finalize,
    fill_value=(0, 0),
    dtypes=(None, np.intp),
    final_dtype=np.floating,
)


# TODO: fix this for complex numbers
def _var_finalize(sumsq, sum_, count, ddof=0):
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (sumsq - (sum_**2 / count)) / (count - ddof)
    result[count <= ddof] = np.nan
    return result


def _std_finalize(sumsq, sum_, count, ddof=0):
    return np.sqrt(_var_finalize(sumsq, sum_, count, ddof))


# var, std always promote to float, so we set nan
var = Aggregation(
    "var",
    chunk=("sum_of_squares", "sum", "nanlen"),
    combine=("sum", "sum", "sum"),
    finalize=_var_finalize,
    fill_value=0,
    final_fill_value=np.nan,
    dtypes=(None, None, np.intp),
    final_dtype=np.floating,
)
nanvar = Aggregation(
    "nanvar",
    chunk=("nansum_of_squares", "nansum", "nanlen"),
    combine=("sum", "sum", "sum"),
    finalize=_var_finalize,
    fill_value=0,
    final_fill_value=np.nan,
    dtypes=(None, None, np.intp),
    final_dtype=np.floating,
)
std = Aggregation(
    "std",
    chunk=("sum_of_squares", "sum", "nanlen"),
    combine=("sum", "sum", "sum"),
    finalize=_std_finalize,
    fill_value=0,
    final_fill_value=np.nan,
    dtypes=(None, None, np.intp),
    final_dtype=np.floating,
)
nanstd = Aggregation(
    "nanstd",
    chunk=("nansum_of_squares", "nansum", "nanlen"),
    combine=("sum", "sum", "sum"),
    finalize=_std_finalize,
    fill_value=0,
    final_fill_value=np.nan,
    dtypes=(None, None, np.intp),
    final_dtype=np.floating,
)


min_ = Aggregation("min", chunk="min", combine="min", fill_value=dtypes.INF)
nanmin = Aggregation("nanmin", chunk="nanmin", combine="nanmin", fill_value=np.nan)
max_ = Aggregation("max", chunk="max", combine="max", fill_value=dtypes.NINF)
nanmax = Aggregation("nanmax", chunk="nanmax", combine="nanmax", fill_value=np.nan)


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


def _pick_second(*x):
    return x[1]


argmax = Aggregation(
    "argmax",
    preprocess=argreduce_preprocess,
    chunk=("max", "argmax"),  # order is important
    combine=("max", "argmax"),
    reduction_type="argreduce",
    fill_value=(dtypes.NINF, 0),
    final_fill_value=-1,
    finalize=_pick_second,
    dtypes=(None, np.intp),
    final_dtype=np.intp,
)

argmin = Aggregation(
    "argmin",
    preprocess=argreduce_preprocess,
    chunk=("min", "argmin"),  # order is important
    combine=("min", "argmin"),
    reduction_type="argreduce",
    fill_value=(dtypes.INF, 0),
    final_fill_value=-1,
    finalize=_pick_second,
    dtypes=(None, np.intp),
    final_dtype=np.intp,
)

nanargmax = Aggregation(
    "nanargmax",
    preprocess=argreduce_preprocess,
    chunk=("nanmax", "nanargmax"),  # order is important
    combine=("max", "argmax"),
    reduction_type="argreduce",
    fill_value=(dtypes.NINF, 0),
    final_fill_value=-1,
    finalize=_pick_second,
    dtypes=(None, np.intp),
    final_dtype=np.intp,
)

nanargmin = Aggregation(
    "nanargmin",
    preprocess=argreduce_preprocess,
    chunk=("nanmin", "nanargmin"),  # order is important
    combine=("min", "argmin"),
    reduction_type="argreduce",
    fill_value=(dtypes.INF, 0),
    final_fill_value=-1,
    finalize=_pick_second,
    dtypes=(None, np.intp),
    final_dtype=np.intp,
)

first = Aggregation("first", chunk=None, combine=None, fill_value=0)
last = Aggregation("last", chunk=None, combine=None, fill_value=0)
nanfirst = Aggregation("nanfirst", chunk="nanfirst", combine="nanfirst", fill_value=np.nan)
nanlast = Aggregation("nanlast", chunk="nanlast", combine="nanlast", fill_value=np.nan)

all_ = Aggregation(
    "all",
    chunk="all",
    combine="all",
    fill_value=True,
    final_fill_value=False,
    dtypes=bool,
    final_dtype=bool,
)
any_ = Aggregation(
    "any",
    chunk="any",
    combine="any",
    fill_value=False,
    final_fill_value=False,
    dtypes=bool,
    final_dtype=bool,
)

# Support statistical quantities only blockwise
# The parallel versions will be approximate and are hard to implement!
median = Aggregation(
    name="median", fill_value=dtypes.NA, chunk=None, combine=None, final_dtype=np.float64
)
nanmedian = Aggregation(
    name="nanmedian", fill_value=dtypes.NA, chunk=None, combine=None, final_dtype=np.float64
)


def quantile_new_dims_func(q) -> tuple[Dim]:
    return (Dim(name="quantile", values=q),)


quantile = Aggregation(
    name="quantile",
    fill_value=dtypes.NA,
    chunk=None,
    combine=None,
    final_dtype=np.float64,
    new_dims_func=quantile_new_dims_func,
)
nanquantile = Aggregation(
    name="nanquantile",
    fill_value=dtypes.NA,
    chunk=None,
    combine=None,
    final_dtype=np.float64,
    new_dims_func=quantile_new_dims_func,
)
mode = Aggregation(name="mode", fill_value=dtypes.NA, chunk=None, combine=None)
nanmode = Aggregation(name="nanmode", fill_value=dtypes.NA, chunk=None, combine=None)

aggregations = {
    "any": any_,
    "all": all_,
    "count": count,
    "sum": sum_,
    "nansum": nansum,
    "prod": prod,
    "nanprod": nanprod,
    "mean": mean,
    "nanmean": nanmean,
    "var": var,
    "nanvar": nanvar,
    "std": std,
    "nanstd": nanstd,
    "max": max_,
    "nanmax": nanmax,
    "min": min_,
    "nanmin": nanmin,
    "argmax": argmax,
    "nanargmax": nanargmax,
    "argmin": argmin,
    "nanargmin": nanargmin,
    "first": first,
    "nanfirst": nanfirst,
    "last": last,
    "nanlast": nanlast,
    "median": median,
    "nanmedian": nanmedian,
    "quantile": quantile,
    "nanquantile": nanquantile,
    "mode": mode,
    "nanmode": nanmode,
}


def _initialize_aggregation(
    func: str | Aggregation,
    dtype,
    array_dtype,
    fill_value,
    min_count: int,
    finalize_kwargs: dict[Any, Any] | None,
) -> Aggregation:
    if not isinstance(func, Aggregation):
        try:
            # TODO: need better interface
            # we set dtype, fillvalue on reduction later. so deepcopy now
            agg = copy.deepcopy(aggregations[func])
        except KeyError:
            raise NotImplementedError(f"Reduction {func!r} not implemented yet")
    elif isinstance(func, Aggregation):
        # TODO: test that func is a valid Aggregation
        agg = copy.deepcopy(func)
        func = agg.name
    else:
        raise ValueError("Bad type for func. Expected str or Aggregation")

    # np.dtype(None) == np.dtype("float64")!!!
    # so check for not None
    dtype_: np.dtype | None = (
        np.dtype(dtype) if dtype is not None and not isinstance(dtype, np.dtype) else dtype
    )

    final_dtype = _normalize_dtype(dtype_ or agg.dtype_init["final"], array_dtype, fill_value)
    agg.dtype = {
        "user": dtype,  # Save to automatically choose an engine
        "final": final_dtype,
        "numpy": (final_dtype,),
        "intermediate": tuple(
            (
                _normalize_dtype(int_dtype, np.result_type(array_dtype, final_dtype), int_fv)
                if int_dtype is None
                else np.dtype(int_dtype)
            )
            for int_dtype, int_fv in zip(
                agg.dtype_init["intermediate"], agg.fill_value["intermediate"]
            )
        ),
    }

    # Replace sentinel fill values according to dtype
    agg.fill_value["user"] = fill_value
    agg.fill_value["intermediate"] = tuple(
        _get_fill_value(dt, fv)
        for dt, fv in zip(agg.dtype["intermediate"], agg.fill_value["intermediate"])
    )
    agg.fill_value[func] = _get_fill_value(agg.dtype["final"], agg.fill_value[func])

    fv = fill_value if fill_value is not None else agg.fill_value[agg.name]
    if _is_arg_reduction(agg):
        # this allows us to unravel_index easily. we have to do that nearly every time.
        agg.fill_value["numpy"] = (0,)
    else:
        agg.fill_value["numpy"] = (fv,)

    if finalize_kwargs is not None:
        assert isinstance(finalize_kwargs, dict)
        agg.finalize_kwargs = finalize_kwargs

    # This is needed for the dask pathway.
    # Because we use intermediate fill_value since a group could be
    # absent in one block, but present in another block
    # We set it for numpy to get nansum, nanprod tests to pass
    # where the identity element is 0, 1
    if min_count > 0:
        agg.min_count = min_count
        agg.numpy += ("nanlen",)
        if agg.chunk != (None,):
            agg.chunk += ("nanlen",)
            agg.combine += ("sum",)
        agg.fill_value["intermediate"] += (0,)
        agg.fill_value["numpy"] += (0,)
        agg.dtype["intermediate"] += (np.intp,)
        agg.dtype["numpy"] += (np.intp,)
    else:
        agg.min_count = 0

    simple_combine: list[Callable | None] = []
    for combine in agg.combine:
        if isinstance(combine, str):
            if combine in ["nanfirst", "nanlast"]:
                simple_combine.append(getattr(xrutils, combine))
            else:
                simple_combine.append(getattr(np, combine))
        else:
            simple_combine.append(combine)

    agg.simple_combine = tuple(simple_combine)

    return agg
