from __future__ import annotations

import copy
import itertools
import math
import operator
import sys
import warnings
from collections import namedtuple
from collections.abc import Sequence
from functools import partial, reduce
from numbers import Integral
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Union,
    overload,
)

import numpy as np
import numpy_groupies as npg
import pandas as pd
import toolz as tlz

from . import xrdtypes
from .aggregate_flox import _prepare_for_flox
from .aggregations import (
    Aggregation,
    _atleast_1d,
    _initialize_aggregation,
    generic_aggregate,
)
from .cache import memoize
from .xrutils import is_duck_array, is_duck_dask_array, isnull, module_available

HAS_NUMBAGG = module_available("numbagg", minversion="0.3.0")

if TYPE_CHECKING:
    try:
        if sys.version_info < (3, 11):
            from typing_extensions import Unpack
        else:
            from typing import Unpack
    except (ModuleNotFoundError, ImportError):
        Unpack: Any  # type: ignore[no-redef]

    import dask.array.Array as DaskArray
    from dask.typing import Graph

    T_DuckArray = Union[np.ndarray, DaskArray]  # Any ?
    T_By = T_DuckArray
    T_Bys = tuple[T_By, ...]
    T_ExpectIndex = Union[pd.Index]
    T_ExpectIndexTuple = tuple[T_ExpectIndex, ...]
    T_ExpectIndexOpt = Union[T_ExpectIndex, None]
    T_ExpectIndexOptTuple = tuple[T_ExpectIndexOpt, ...]
    T_Expect = Union[Sequence, np.ndarray, T_ExpectIndex]
    T_ExpectTuple = tuple[T_Expect, ...]
    T_ExpectOpt = Union[Sequence, np.ndarray, T_ExpectIndexOpt]
    T_ExpectOptTuple = tuple[T_ExpectOpt, ...]
    T_ExpectedGroups = Union[T_Expect, T_ExpectOptTuple]
    T_ExpectedGroupsOpt = Union[T_ExpectedGroups, None]
    T_Func = Union[str, Callable]
    T_Funcs = Union[T_Func, Sequence[T_Func]]
    T_Agg = Union[str, Aggregation]
    T_Axis = int
    T_Axes = tuple[T_Axis, ...]
    T_AxesOpt = Union[T_Axis, T_Axes, None]
    T_Dtypes = Union[np.typing.DTypeLike, Sequence[np.typing.DTypeLike], None]
    T_FillValues = Union[np.typing.ArrayLike, Sequence[np.typing.ArrayLike], None]
    T_Engine = Literal["flox", "numpy", "numba", "numbagg"]
    T_EngineOpt = None | T_Engine
    T_Method = Literal["map-reduce", "blockwise", "cohorts"]
    T_IsBins = Union[bool | Sequence[bool]]


IntermediateDict = dict[Union[str, Callable], Any]
FinalResultsDict = dict[str, Union["DaskArray", np.ndarray]]
FactorProps = namedtuple("FactorProps", "offset_group nan_sentinel nanmask")

# This dummy axis is inserted using np.expand_dims
# and then reduced over during the combine stage by
# _simple_combine.
DUMMY_AXIS = -2


def _postprocess_numbagg(result, *, func, fill_value, size, seen_groups):
    """Account for numbagg not providing a fill_value kwarg."""
    from .aggregate_numbagg import DEFAULT_FILL_VALUE

    if not isinstance(func, str) or func not in DEFAULT_FILL_VALUE:
        return result
    # The condition needs to be
    # len(found_groups) < size; if so we mask with fill_value (?)
    default_fv = DEFAULT_FILL_VALUE[func]
    needs_masking = fill_value is not None and not np.array_equal(
        fill_value, default_fv, equal_nan=True
    )
    groups = np.arange(size)
    if needs_masking:
        mask = np.isin(groups, seen_groups, assume_unique=True, invert=True)
        if mask.any():
            result[..., groups[mask]] = fill_value
    return result


def _issorted(arr: np.ndarray) -> bool:
    return bool((arr[:-1] <= arr[1:]).all())


def _is_arg_reduction(func: T_Agg) -> bool:
    if isinstance(func, str) and func in ["argmin", "argmax", "nanargmax", "nanargmin"]:
        return True
    if isinstance(func, Aggregation) and func.reduction_type == "argreduce":
        return True
    return False


def _is_minmax_reduction(func: T_Agg) -> bool:
    return not _is_arg_reduction(func) and (
        isinstance(func, str) and ("max" in func or "min" in func)
    )


def _is_first_last_reduction(func: T_Agg) -> bool:
    return isinstance(func, str) and func in ["nanfirst", "nanlast", "first", "last"]


def _get_expected_groups(by: T_By, sort: bool) -> T_ExpectIndex:
    if is_duck_dask_array(by):
        raise ValueError("Please provide expected_groups if not grouping by a numpy array.")
    flatby = by.reshape(-1)
    expected = pd.unique(flatby[~isnull(flatby)])
    return _convert_expected_groups_to_index((expected,), isbin=(False,), sort=sort)[0]


def _get_chunk_reduction(reduction_type: Literal["reduce", "argreduce"]) -> Callable:
    if reduction_type == "reduce":
        return chunk_reduce
    elif reduction_type == "argreduce":
        return chunk_argreduce
    else:
        raise ValueError(f"Unknown reduction type: {reduction_type}")


def is_nanlen(reduction: T_Func) -> bool:
    return isinstance(reduction, str) and reduction == "nanlen"


def _move_reduce_dims_to_end(arr: np.ndarray, axis: T_Axes) -> np.ndarray:
    """Transpose `arr` by moving `axis` to the end."""
    axis = tuple(axis)
    order = tuple(ax for ax in np.arange(arr.ndim) if ax not in axis) + axis
    arr = arr.transpose(order)
    return arr


def _collapse_axis(arr: np.ndarray, naxis: int) -> np.ndarray:
    """Reshape so that the last `naxis` axes are collapsed to one axis."""
    newshape = arr.shape[:-naxis] + (math.prod(arr.shape[-naxis:]),)
    return arr.reshape(newshape)


@memoize
def _get_optimal_chunks_for_groups(chunks, labels):
    chunkidx = np.cumsum(chunks) - 1
    # what are the groups at chunk boundaries
    labels_at_chunk_bounds = _unique(labels[chunkidx])
    # what's the last index of all groups
    last_indexes = npg.aggregate_numpy.aggregate(labels, np.arange(len(labels)), func="last")
    # what's the last index of groups at the chunk boundaries.
    lastidx = last_indexes[labels_at_chunk_bounds]

    if len(chunkidx) == len(lastidx) and (chunkidx == lastidx).all():
        return chunks

    first_indexes = npg.aggregate_numpy.aggregate(labels, np.arange(len(labels)), func="first")
    firstidx = first_indexes[labels_at_chunk_bounds]

    newchunkidx = [0]
    for c, f, l in zip(chunkidx, firstidx, lastidx):  # noqa
        Δf = abs(c - f)
        Δl = abs(c - l)
        if c == 0 or newchunkidx[-1] > l:
            continue
        if Δf < Δl and f > newchunkidx[-1]:
            newchunkidx.append(f)
        else:
            newchunkidx.append(l + 1)
    if newchunkidx[-1] != chunkidx[-1] + 1:
        newchunkidx.append(chunkidx[-1] + 1)
    newchunks = np.diff(newchunkidx)

    assert sum(newchunks) == sum(chunks)
    return tuple(newchunks)


def _unique(a: np.ndarray) -> np.ndarray:
    """Much faster to use pandas unique and sort the results.
    np.unique sorts before uniquifying and is slow."""
    return np.sort(pd.unique(a.reshape(-1)))


@memoize
def find_group_cohorts(labels, chunks, merge: bool = True) -> dict:
    """
    Finds groups labels that occur together aka "cohorts"

    If available, results are cached in a 1MB cache managed by `cachey`.
    This allows us to be quick when repeatedly calling groupby_reduce
    for arrays with the same chunking (e.g. an xarray Dataset).

    Parameters
    ----------
    labels : np.ndarray
        mD Array of group labels
    chunks : tuple
        nD array that is being reduced
    merge : bool, optional
        Attempt to merge cohorts when one cohort's chunks are a subset
        of another cohort's chunks.

    Returns
    -------
    cohorts: dict_values
        Iterable of cohorts
    """
    import dask

    # To do this, we must have values in memory so casting to numpy should be safe
    labels = np.asarray(labels)

    # Build an array with the shape of labels, but where every element is the "chunk number"
    # 1. First subset the array appropriately
    axis = range(-labels.ndim, 0)
    # Easier to create a dask array and use the .blocks property
    array = dask.array.empty(tuple(sum(c) for c in chunks), chunks=chunks)
    labels = np.broadcast_to(labels, array.shape[-labels.ndim :])

    #  Iterate over each block and create a new block of same shape with "chunk number"
    shape = tuple(array.blocks.shape[ax] for ax in axis)
    # Use a numpy object array to enable assignment in the loop
    # TODO: is it possible to just use a nested list?
    #       That is what we need for `np.block`
    blocks = np.empty(shape, dtype=object)
    array_chunks = tuple(np.array(c) for c in array.chunks)
    for idx, blockindex in enumerate(np.ndindex(array.numblocks)):
        chunkshape = tuple(c[i] for c, i in zip(array_chunks, blockindex))
        blocks[blockindex] = np.full(chunkshape, idx)
    which_chunk = np.block(blocks.tolist()).reshape(-1)

    raveled = labels.reshape(-1)
    # these are chunks where a label is present
    label_chunks = pd.Series(which_chunk).groupby(raveled).unique()

    # These invert the label_chunks mapping so we know which labels occur together.
    def invert(x) -> tuple[np.ndarray, ...]:
        arr = label_chunks.get(x)
        return tuple(arr)  # type: ignore [arg-type] # pandas issue?

    chunks_cohorts = tlz.groupby(invert, label_chunks.keys())

    # If our dataset has chunksize one along the axis,
    # then no merging is possible.
    single_chunks = all((ac == 1).all() for ac in array_chunks)

    if merge and not single_chunks:
        # First sort by number of chunks occupied by cohort
        sorted_chunks_cohorts = dict(
            sorted(chunks_cohorts.items(), key=lambda kv: len(kv[0]), reverse=True)
        )

        items = tuple(sorted_chunks_cohorts.items())

        merged_cohorts = {}
        merged_keys = []

        # Now we iterate starting with the longest number of chunks,
        # and then merge in cohorts that are present in a subset of those chunks
        # I think this is suboptimal and must fail at some point.
        # But it might work for most cases. There must be a better way...
        for idx, (k1, v1) in enumerate(items):
            if k1 in merged_keys:
                continue
            merged_cohorts[k1] = copy.deepcopy(v1)
            for k2, v2 in items[idx + 1 :]:
                if k2 in merged_keys:
                    continue
                if set(k2).issubset(set(k1)):
                    merged_cohorts[k1].extend(v2)
                    merged_keys.append(k2)

        # make sure each cohort is sorted after merging
        sorted_merged_cohorts = {k: sorted(v) for k, v in merged_cohorts.items()}
        # sort by first label in cohort
        # This will help when sort=True (default)
        # and we have to resort the dask array
        return dict(sorted(sorted_merged_cohorts.items(), key=lambda kv: kv[1][0]))

    else:
        return chunks_cohorts


def rechunk_for_cohorts(
    array: DaskArray,
    axis: T_Axis,
    labels: np.ndarray,
    force_new_chunk_at: Sequence,
    chunksize: int | None = None,
    ignore_old_chunks: bool = False,
    debug: bool = False,
) -> DaskArray:
    """
    Rechunks array so that each new chunk contains groups that always occur together.

    Parameters
    ----------
    array : dask.array.Array
        array to rechunk
    axis : int
        Axis to rechunk
    labels : np.array
        1D Group labels to align chunks with. This routine works
        well when ``labels`` has repeating patterns: e.g.
        ``1, 2, 3, 1, 2, 3, 4, 1, 2, 3`` though there is no requirement
        that the pattern must contain sequences.
    force_new_chunk_at : Sequence
        Labels at which we always start a new chunk. For
        the example ``labels`` array, this would be `1`.
    chunksize : int, optional
        nominal chunk size. Chunk size is exceeded when the label
        in ``force_new_chunk_at`` is less than ``chunksize//2`` elements away.
        If None, uses median chunksize along axis.

    Returns
    -------
    dask.array.Array
        rechunked array
    """
    if chunksize is None:
        chunksize = np.median(array.chunks[axis]).astype(int)

    if len(labels) != array.shape[axis]:
        raise ValueError(
            "labels must be equal to array.shape[axis]. "
            f"Received length {len(labels)}.  Expected length {array.shape[axis]}"
        )

    force_new_chunk_at = _atleast_1d(force_new_chunk_at)
    oldchunks = array.chunks[axis]
    oldbreaks = np.insert(np.cumsum(oldchunks), 0, 0)
    if debug:
        labels_at_breaks = labels[oldbreaks[:-1]]
        print(labels_at_breaks[:40])

    isbreak = np.isin(labels, force_new_chunk_at)
    if not np.any(isbreak):
        raise ValueError("One or more labels in ``force_new_chunk_at`` not present in ``labels``.")

    divisions = []
    counter = 1
    for idx, lab in enumerate(labels):
        if lab in force_new_chunk_at or idx == 0:
            divisions.append(idx)
            counter = 1
            continue

        next_break = np.nonzero(isbreak[idx:])[0]
        if next_break.any():
            next_break_is_close = next_break[0] <= chunksize // 2
        else:
            next_break_is_close = False

        if (not ignore_old_chunks and idx in oldbreaks) or (
            counter >= chunksize and not next_break_is_close
        ):
            divisions.append(idx)
            counter = 1
            continue

        counter += 1

    divisions.append(len(labels))
    if debug:
        labels_at_breaks = labels[divisions[:-1]]
        print(labels_at_breaks[:40])

    newchunks = tuple(np.diff(divisions))
    if debug:
        print(divisions[:10], newchunks[:10])
        print(divisions[-10:], newchunks[-10:])
    assert sum(newchunks) == len(labels)

    if newchunks == array.chunks[axis]:
        return array
    else:
        return array.rechunk({axis: newchunks})


def rechunk_for_blockwise(array: DaskArray, axis: T_Axis, labels: np.ndarray) -> DaskArray:
    """
    Rechunks array so that group boundaries line up with chunk boundaries, allowing
    embarrassingly parallel group reductions.

    This only works when the groups are sequential
    (e.g. labels = ``[0,0,0,1,1,1,1,2,2]``).
    Such patterns occur when using ``.resample``.

    Parameters
    ----------
    array : DaskArray
        Array to rechunk
    axis : int
        Axis along which to rechunk the array.
    labels : np.ndarray
        Group labels

    Returns
    -------
    DaskArray
        Rechunked array
    """
    labels = factorize_((labels,), axes=())[0]
    chunks = array.chunks[axis]
    newchunks = _get_optimal_chunks_for_groups(chunks, labels)
    if newchunks == chunks:
        return array
    else:
        return array.rechunk({axis: newchunks})


def reindex_(
    array: np.ndarray,
    from_,
    to,
    fill_value: Any = None,
    axis: T_Axis = -1,
    promote: bool = False,
) -> np.ndarray:
    if not isinstance(to, pd.Index):
        if promote:
            to = pd.Index(to)
        else:
            raise ValueError("reindex requires a pandas.Index or promote=True")

    if to.ndim > 1:
        raise ValueError(f"Cannot reindex to a multidimensional array: {to}")

    if array.shape[axis] == 0:
        # all groups were NaN
        reindexed = np.full(array.shape[:-1] + (len(to),), fill_value, dtype=array.dtype)
        return reindexed

    from_ = pd.Index(from_)
    # short-circuit for trivial case
    if from_.equals(to):
        return array

    if from_.dtype.kind == "O" and isinstance(from_[0], tuple):
        raise NotImplementedError(
            "Currently does not support reindexing with object arrays of tuples. "
            "These occur when grouping by multi-indexed variables in xarray."
        )
    idx = from_.get_indexer(to)
    indexer = [slice(None, None)] * array.ndim
    indexer[axis] = idx
    reindexed = array[tuple(indexer)]
    if any(idx == -1):
        if fill_value is None:
            raise ValueError("Filling is required. fill_value cannot be None.")
        indexer[axis] = idx == -1
        # This allows us to match xarray's type promotion rules
        if fill_value is xrdtypes.NA or isnull(fill_value):
            new_dtype, fill_value = xrdtypes.maybe_promote(reindexed.dtype)
            reindexed = reindexed.astype(new_dtype, copy=False)
        reindexed[tuple(indexer)] = fill_value
    return reindexed


def offset_labels(labels: np.ndarray, ngroups: int) -> tuple[np.ndarray, int]:
    """
    Offset group labels by dimension. This is used when we
    reduce over a subset of the dimensions of by. It assumes that the reductions
    dimensions have been flattened in the last dimension
    Copied from xhistogram &
    https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    """
    assert labels.ndim > 1
    offset: np.ndarray = (
        labels + np.arange(math.prod(labels.shape[:-1])).reshape((*labels.shape[:-1], -1)) * ngroups
    )
    # -1 indicates NaNs. preserve these otherwise we aggregate in the wrong groups!
    offset[labels == -1] = -1
    size: int = math.prod(labels.shape[:-1]) * ngroups
    return offset, size


@overload
def factorize_(
    by: T_Bys,
    axes: T_Axes,
    *,
    fastpath: Literal[True],
    expected_groups: T_ExpectIndexOptTuple | None = None,
    reindex: bool = False,
    sort: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, ...], int, int, None]:
    ...


@overload
def factorize_(
    by: T_Bys,
    axes: T_Axes,
    *,
    expected_groups: T_ExpectIndexOptTuple | None = None,
    reindex: bool = False,
    sort: bool = True,
    fastpath: Literal[False] = False,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, ...], int, int, FactorProps]:
    ...


@overload
def factorize_(
    by: T_Bys,
    axes: T_Axes,
    *,
    expected_groups: T_ExpectIndexOptTuple | None = None,
    reindex: bool = False,
    sort: bool = True,
    fastpath: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, ...], int, int, FactorProps | None]:
    ...


def factorize_(
    by: T_Bys,
    axes: T_Axes,
    *,
    expected_groups: T_ExpectIndexOptTuple | None = None,
    reindex: bool = False,
    sort: bool = True,
    fastpath: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, ...], int, int, FactorProps | None]:
    """
    Returns an array of integer  codes  for groups (and associated data)
    by wrapping pd.cut and pd.factorize (depending on isbin).
    This method handles reindex and sort so that we don't spend time reindexing / sorting
    a possibly large results array. Instead we set up the appropriate integer codes (group_idx)
    so that the results come out in the appropriate order.
    """
    if expected_groups is None:
        expected_groups = (None,) * len(by)

    factorized = []
    found_groups = []
    for groupvar, expect in zip(by, expected_groups):
        flat = groupvar.reshape(-1)
        if isinstance(expect, pd.RangeIndex):
            # idx is a view of the original `by` array
            # copy here so we don't have a race condition with the
            # group_idx[nanmask] = nan_sentinel assignment later
            # this is important in shared-memory parallelism with dask
            # TODO: figure out how to avoid this
            idx = flat.copy()
            found_groups.append(np.array(expect))
            # TODO: fix by using masked integers
            idx[idx > expect[-1]] = -1

        elif isinstance(expect, pd.IntervalIndex):
            if expect.closed == "both":
                raise NotImplementedError
            bins = np.concatenate([expect.left.to_numpy(), expect.right.to_numpy()[[-1]]])

            # digitize is 0 or idx.max() for values outside the bounds of all intervals
            # make it behave like pd.cut which uses -1:
            if len(bins) > 1:
                right = expect.closed_right
                idx = np.digitize(
                    flat,
                    bins=bins.view(np.int64) if bins.dtype.kind == "M" else bins,
                    right=right,
                )
                idx -= 1
                within_bins = flat <= bins.max() if right else flat < bins.max()
                idx[~within_bins] = -1
            else:
                idx = np.zeros_like(flat, dtype=np.intp) - 1

            found_groups.append(np.array(expect))
        else:
            if expect is not None and reindex:
                sorter = np.argsort(expect)
                groups = expect[(sorter,)] if sort else expect
                idx = np.searchsorted(expect, flat, sorter=sorter)
                mask = ~np.isin(flat, expect) | isnull(flat) | (idx == len(expect))
                if not sort:
                    # idx is the index in to the sorted array.
                    # if we didn't want sorting, unsort it back
                    idx[(idx == len(expect),)] = -1
                    idx = sorter[(idx,)]
                idx[mask] = -1
            else:
                idx, groups = pd.factorize(flat, sort=sort)  # type: ignore[arg-type]

            found_groups.append(np.array(groups))
        factorized.append(idx.reshape(groupvar.shape))

    grp_shape = tuple(len(grp) for grp in found_groups)
    ngroups = math.prod(grp_shape)
    if len(by) > 1:
        group_idx = np.ravel_multi_index(factorized, grp_shape, mode="wrap")
        # NaNs; as well as values outside the bins are coded by -1
        # Restore these after the raveling
        nan_by_mask = reduce(np.logical_or, [(f == -1) for f in factorized])
        group_idx[nan_by_mask] = -1
    else:
        group_idx = factorized[0]

    if fastpath:
        return group_idx, tuple(found_groups), grp_shape, ngroups, ngroups, None

    if len(axes) == 1 and groupvar.ndim > 1:
        # Not reducing along all dimensions of by
        # this is OK because for 3D by and axis=(1,2),
        # we collapse to a 2D by and axis=-1
        offset_group = True
        group_idx, size = offset_labels(group_idx.reshape(by[0].shape), ngroups)
    else:
        size = ngroups
        offset_group = False

    # numpy_groupies cannot deal with group_idx = -1
    # so we'll add use ngroups as the sentinel
    # note we cannot simply remove the NaN locations;
    # that would mess up argmax, argmin
    nan_sentinel = size if offset_group else ngroups
    nanmask = group_idx == -1
    if nanmask.any():
        # bump it up so there's a place to assign values to the nan_sentinel index
        size += 1
    group_idx[nanmask] = nan_sentinel

    props = FactorProps(offset_group, nan_sentinel, nanmask)
    return group_idx, tuple(found_groups), grp_shape, ngroups, size, props


def chunk_argreduce(
    array_plus_idx: tuple[np.ndarray, ...],
    by: np.ndarray,
    func: T_Funcs,
    expected_groups: pd.Index | None,
    axis: T_AxesOpt,
    fill_value: T_FillValues,
    dtype: T_Dtypes = None,
    reindex: bool = False,
    engine: T_Engine = "numpy",
    sort: bool = True,
    user_dtype=None,
) -> IntermediateDict:
    """
    Per-chunk arg reduction.

    Expects a tuple of (array, index along reduction axis). Inspired by
    dask.array.reductions.argtopk
    """
    array, idx = array_plus_idx
    by = np.broadcast_to(by, array.shape)

    results = chunk_reduce(
        array,
        by,
        func,
        expected_groups=None,
        axis=axis,
        fill_value=fill_value,
        dtype=dtype,
        engine=engine,
        sort=sort,
        user_dtype=user_dtype,
    )
    if not isnull(results["groups"]).all():
        idx = np.broadcast_to(idx, array.shape)

        # array, by get flattened to 1D before passing to npg
        # so the indexes need to be unraveled
        newidx = np.unravel_index(results["intermediates"][1], array.shape)

        # Now index into the actual "global" indexes `idx`
        results["intermediates"][1] = idx[newidx]

    if reindex and expected_groups is not None:
        results["intermediates"][1] = reindex_(
            results["intermediates"][1], results["groups"].squeeze(), expected_groups, fill_value=0
        )

    assert results["intermediates"][0].shape == results["intermediates"][1].shape

    return results


def chunk_reduce(
    array: np.ndarray,
    by: np.ndarray,
    func: T_Funcs,
    expected_groups: pd.Index | None,
    axis: T_AxesOpt = None,
    fill_value: T_FillValues = None,
    dtype: T_Dtypes = None,
    reindex: bool = False,
    engine: T_Engine = "numpy",
    kwargs: Sequence[dict] | None = None,
    sort: bool = True,
    user_dtype=None,
) -> IntermediateDict:
    """
    Wrapper for numpy_groupies aggregate that supports nD ``array`` and
    mD ``by``.

    Core groupby reduction using numpy_groupies. Uses ``pandas.factorize`` to factorize
    ``by``. Offsets the groups if not reducing along all dimensions of ``by``.
    Always ravels ``by`` to 1D, flattens appropriate dimensions of array.

    When dask arrays are passed to groupby_reduce, this function is called on every
    block.

    Parameters
    ----------
    array : numpy.ndarray
        Array of values to reduced
    by : numpy.ndarray
        Array to group by.
    func : str or Callable or Sequence[str] or Sequence[Callable]
        Name of reduction or function, passed to numpy_groupies.
        Supports multiple reductions.
    axis : (optional) int or Sequence[int]
        If None, reduce along all dimensions of array.
        Else reduce along specified axes.

    Returns
    -------
    dict
    """

    if not (isinstance(func, str) or callable(func)):
        funcs = func
    else:
        funcs = (func,)
    nfuncs = len(funcs)

    if isinstance(dtype, Sequence):
        dtypes = dtype
    else:
        dtypes = (dtype,) * nfuncs
    assert len(dtypes) >= nfuncs

    if isinstance(fill_value, Sequence):
        fill_values = fill_value
    else:
        fill_values = (fill_value,) * nfuncs
    assert len(fill_values) >= nfuncs

    if isinstance(kwargs, Sequence):
        kwargss = kwargs
    else:
        kwargss = ({},) * nfuncs
    assert len(kwargss) >= nfuncs

    if isinstance(axis, Sequence):
        axes: T_Axes = axis
        nax = len(axes)
    else:
        nax = by.ndim
        if axis is None:
            axes = ()
        else:
            axes = (axis,) * nax

    assert by.ndim <= array.ndim

    final_array_shape = array.shape[:-nax] + (1,) * (nax - 1)
    final_groups_shape = (1,) * (nax - 1)

    if 1 < nax < by.ndim:
        # when axis is a tuple
        # collapse and move reduction dimensions to the end
        by = _collapse_axis(by, nax)
        array = _collapse_axis(array, nax)
        axes = (-1,)
        nax = 1

    # if indices=[2,2,2], npg assumes groups are (0, 1, 2);
    # and will return a result that is bigger than necessary
    # avoid by factorizing again so indices=[2,2,2] is changed to
    # indices=[0,0,0]. This is necessary when combining block results
    # factorize can handle strings etc unlike digitize
    group_idx, grps, found_groups_shape, _, size, props = factorize_(
        (by,), axes, expected_groups=(expected_groups,), reindex=reindex, sort=sort
    )
    (groups,) = grps

    # do this *before* possible broadcasting below.
    # factorize_ has already taken care of offsetting
    seen_groups = _unique(group_idx)

    order = "C"
    if nax > 1:
        needs_broadcast = any(
            group_idx.shape[ax] != array.shape[ax] and group_idx.shape[ax] == 1
            for ax in range(-nax, 0)
        )
        if needs_broadcast:
            # This is the dim=... case, it's a lot faster to ravel group_idx
            # in fortran order since group_idx is then sorted
            # I'm seeing 400ms -> 23ms for engine="flox"
            # Of course we are slower to ravel `array` but we avoid argsorting
            # both `array` *and* `group_idx` in _prepare_for_flox
            group_idx = np.broadcast_to(group_idx, array.shape[-by.ndim :])
            if engine == "flox":
                group_idx = group_idx.reshape(-1, order="F")
                order = "F"
    # always reshape to 1D along group dimensions
    newshape = array.shape[: array.ndim - by.ndim] + (math.prod(array.shape[-by.ndim :]),)
    array = array.reshape(newshape, order=order)  # type: ignore[call-overload]
    group_idx = group_idx.reshape(-1)

    assert group_idx.ndim == 1

    empty = np.all(props.nanmask)

    results: IntermediateDict = {"groups": [], "intermediates": []}
    if reindex and expected_groups is not None:
        # TODO: what happens with binning here?
        results["groups"] = expected_groups.to_numpy()
    else:
        if empty:
            results["groups"] = np.array([np.nan])
        else:
            results["groups"] = groups

    # npg's argmax ensures that index of first "max" is returned assuming there
    # are many elements equal to the "max". Sorting messes this up totally.
    # so we skip this for argreductions
    if engine == "flox":
        # is_arg_reduction = any("arg" in f for f in func if isinstance(f, str))
        # if not is_arg_reduction:
        group_idx, array = _prepare_for_flox(group_idx, array)

    final_array_shape += results["groups"].shape
    final_groups_shape += results["groups"].shape

    # we commonly have func=(..., "nanlen", "nanlen") when
    # counts are needed for the final result as well as for masking
    # optimize that out.
    previous_reduction: T_Func = ""
    for reduction, fv, kw, dt in zip(funcs, fill_values, kwargss, dtypes):
        if empty:
            result = np.full(shape=final_array_shape, fill_value=fv)
        elif is_nanlen(reduction) and is_nanlen(previous_reduction):
            result = results["intermediates"][-1]
        else:
            # fill_value here is necessary when reducing with "offset" groups
            kw_func = dict(size=size, dtype=dt, fill_value=fv)
            kw_func.update(kw)

            if callable(reduction):
                # passing a custom reduction for npg to apply per-group is really slow!
                # So this `reduction` has to do the groupby-aggregation
                result = reduction(group_idx, array, **kw_func)
            else:
                result = generic_aggregate(
                    group_idx, array, axis=-1, engine=engine, func=reduction, **kw_func
                ).astype(dt, copy=False)
            if engine == "numbagg":
                result = _postprocess_numbagg(
                    result,
                    func=reduction,
                    size=size,
                    fill_value=fv,
                    # Unfortunately, we cannot reuse found_groups, it has not
                    # been "offset" and is really expected_groups in nearly all cases
                    seen_groups=seen_groups,
                )
            if np.any(props.nanmask):
                # remove NaN group label which should be last
                result = result[..., :-1]
            result = result.reshape(final_array_shape[:-1] + found_groups_shape)
        results["intermediates"].append(result)
        previous_reduction = reduction

    results["groups"] = np.broadcast_to(results["groups"], final_groups_shape)
    return results


def _squeeze_results(results: IntermediateDict, axis: T_Axes) -> IntermediateDict:
    # at the end we squeeze out extra dims
    groups = results["groups"]
    newresults: IntermediateDict = {"groups": [], "intermediates": []}
    newresults["groups"] = np.squeeze(
        groups, axis=tuple(ax for ax in range(groups.ndim - 1) if groups.shape[ax] == 1)
    )
    for v in results["intermediates"]:
        squeeze_ax = tuple(ax for ax in sorted(axis)[:-1] if v.shape[ax] == 1)
        newresults["intermediates"].append(np.squeeze(v, axis=squeeze_ax) if squeeze_ax else v)
    return newresults


def _finalize_results(
    results: IntermediateDict,
    agg: Aggregation,
    axis: T_Axes,
    expected_groups: pd.Index | None,
    fill_value: Any,
    reindex: bool,
) -> FinalResultsDict:
    """Finalize results by
    1. Squeezing out dummy dimensions
    2. Calling agg.finalize with intermediate results
    3. Mask using counts and fill with user-provided fill_value.
    4. reindex to expected_groups
    """
    squeezed = _squeeze_results(results, axis)

    min_count = agg.min_count
    if min_count > 0:
        counts = squeezed["intermediates"][-1]
        squeezed["intermediates"] = squeezed["intermediates"][:-1]

    # finalize step
    finalized: FinalResultsDict = {}
    if agg.finalize is None:
        finalized[agg.name] = squeezed["intermediates"][0]
    else:
        finalized[agg.name] = agg.finalize(*squeezed["intermediates"], **agg.finalize_kwargs)

    if min_count > 0:
        count_mask = counts < min_count
        if count_mask.any():
            # For one count_mask.any() prevents promoting bool to dtype(fill_value) unless
            # necessary
            if fill_value is None:
                raise ValueError("Filling is required but fill_value is None.")
            # This allows us to match xarray's type promotion rules
            if fill_value is xrdtypes.NA:
                new_dtype, fill_value = xrdtypes.maybe_promote(finalized[agg.name].dtype)
                finalized[agg.name] = finalized[agg.name].astype(new_dtype)
            finalized[agg.name] = np.where(count_mask, fill_value, finalized[agg.name])

    # Final reindexing has to be here to be lazy
    if not reindex and expected_groups is not None:
        finalized[agg.name] = reindex_(
            finalized[agg.name], squeezed["groups"], expected_groups, fill_value=fill_value
        )
        finalized["groups"] = expected_groups.to_numpy()
    else:
        finalized["groups"] = squeezed["groups"]

    finalized[agg.name] = finalized[agg.name].astype(agg.dtype["final"], copy=False)
    return finalized


def _aggregate(
    x_chunk,
    combine: Callable,
    agg: Aggregation,
    expected_groups: pd.Index | None,
    axis: T_Axes,
    keepdims,
    fill_value: Any,
    reindex: bool,
) -> FinalResultsDict:
    """Final aggregation step of tree reduction"""
    results = combine(x_chunk, agg, axis, keepdims, is_aggregate=True)
    return _finalize_results(results, agg, axis, expected_groups, fill_value, reindex)


def _expand_dims(results: IntermediateDict) -> IntermediateDict:
    results["intermediates"] = tuple(
        np.expand_dims(array, DUMMY_AXIS) for array in results["intermediates"]
    )
    return results


def _find_unique_groups(x_chunk) -> np.ndarray:
    from dask.base import flatten
    from dask.utils import deepmap

    unique_groups = _unique(np.asarray(tuple(flatten(deepmap(listify_groups, x_chunk)))))
    unique_groups = unique_groups[~isnull(unique_groups)]

    if len(unique_groups) == 0:
        unique_groups = np.array([np.nan])
    return unique_groups


def _simple_combine(
    x_chunk,
    agg: Aggregation,
    axis: T_Axes,
    keepdims: bool,
    reindex: bool,
    is_aggregate: bool = False,
) -> IntermediateDict:
    """
    'Simple' combination of blockwise results.

    1. After the blockwise groupby-reduce, all blocks contain a value for all possible groups,
       and are of the same shape; i.e. reindex must have been True
    2. _expand_dims was used to insert an extra axis DUMMY_AXIS
    3. Here we concatenate along DUMMY_AXIS, and then call the combine function along
       DUMMY_AXIS
    4. At the final aggregate step, we squeeze out DUMMY_AXIS
    """
    from dask.array.core import deepfirst
    from dask.utils import deepmap

    if not reindex:
        # We didn't reindex at the blockwise step
        # So now reindex before combining by reducing along DUMMY_AXIS
        unique_groups = _find_unique_groups(x_chunk)
        x_chunk = deepmap(
            partial(reindex_intermediates, agg=agg, unique_groups=unique_groups), x_chunk
        )
    else:
        unique_groups = deepfirst(x_chunk)["groups"]

    results: IntermediateDict = {"groups": unique_groups}
    results["intermediates"] = []
    axis_ = axis[:-1] + (DUMMY_AXIS,)
    for idx, combine in enumerate(agg.simple_combine):
        array = _conc2(x_chunk, key1="intermediates", key2=idx, axis=axis_)
        assert array.ndim >= 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            assert callable(combine)
            result = combine(array, axis=axis_, keepdims=True)
        if is_aggregate:
            # squeeze out DUMMY_AXIS if this is the last step i.e. called from _aggregate
            result = result.squeeze(axis=DUMMY_AXIS)
        results["intermediates"].append(result)
    return results


def _conc2(x_chunk, key1, key2=slice(None), axis: T_Axes | None = None) -> np.ndarray:
    """copied from dask.array.reductions.mean_combine"""
    from dask.array.core import _concatenate2
    from dask.utils import deepmap

    mapped = deepmap(lambda x: x[key1][key2], x_chunk)
    return _concatenate2(mapped, axes=axis)

    # This doesn't seem to improve things at all; and some tests fail...
    # from dask.array.core import concatenate3
    # for _ in range(mapped[0].ndim-1):
    #    mapped = [mapped]
    # return concatenate3(mapped)


def reindex_intermediates(x: IntermediateDict, agg: Aggregation, unique_groups) -> IntermediateDict:
    new_shape = x["groups"].shape[:-1] + (len(unique_groups),)
    newx: IntermediateDict = {"groups": np.broadcast_to(unique_groups, new_shape)}
    newx["intermediates"] = tuple(
        reindex_(
            v, from_=np.atleast_1d(x["groups"].squeeze()), to=pd.Index(unique_groups), fill_value=f
        )
        for v, f in zip(x["intermediates"], agg.fill_value["intermediate"])
    )
    return newx


def listify_groups(x: IntermediateDict):
    return list(np.atleast_1d(x["groups"].squeeze()))


def _grouped_combine(
    x_chunk,
    agg: Aggregation,
    axis: T_Axes,
    keepdims: bool,
    engine: T_Engine,
    is_aggregate: bool = False,
    sort: bool = True,
) -> IntermediateDict:
    """Combine intermediates step of tree reduction."""
    from dask.utils import deepmap

    combine = agg.combine

    if isinstance(x_chunk, dict):
        # Only one block at final step; skip one extra groupby
        return x_chunk

    if len(axis) != 1:
        # when there's only a single axis of reduction, we can just concatenate later,
        # reindexing is unnecessary
        # I bet we can minimize the amount of reindexing for mD reductions too, but it's complicated
        unique_groups = _find_unique_groups(x_chunk)
        x_chunk = deepmap(
            partial(reindex_intermediates, agg=agg, unique_groups=unique_groups), x_chunk
        )

    # these are negative axis indices useful for concatenating the intermediates
    neg_axis = tuple(range(-len(axis), 0))

    groups = _conc2(x_chunk, "groups", axis=neg_axis)

    if agg.reduction_type == "argreduce":
        # If "nanlen" was added for masking later, we need to account for that
        if agg.chunk[-1] == "nanlen":
            slicer = slice(None, -1)
        else:
            slicer = slice(None, None)

        # We need to send the intermediate array values & indexes at the same time
        # intermediates are (value e.g. max, index e.g. argmax, counts)
        array_idx = tuple(
            _conc2(x_chunk, key1="intermediates", key2=idx, axis=axis) for idx in (0, 1)
        )

        # for a single element along axis, we don't want to run the argreduction twice
        # This happens when we are reducing along an axis with a single chunk.
        avoid_reduction = array_idx[0].shape[axis[0]] == 1
        if avoid_reduction:
            results: IntermediateDict = {"groups": groups, "intermediates": list(array_idx)}
        else:
            results = chunk_argreduce(
                array_idx,
                groups,
                # count gets treated specially next
                func=combine[slicer],  # type: ignore[arg-type]
                axis=axis,
                expected_groups=None,
                fill_value=agg.fill_value["intermediate"][slicer],
                dtype=agg.dtype["intermediate"][slicer],
                engine=engine,
                sort=sort,
            )

        if agg.chunk[-1] == "nanlen":
            counts = _conc2(x_chunk, key1="intermediates", key2=2, axis=axis)

            if avoid_reduction:
                results["intermediates"].append(counts)
            else:
                # sum the counts
                results["intermediates"].append(
                    chunk_reduce(
                        counts,
                        groups,
                        func="sum",
                        axis=axis,
                        expected_groups=None,
                        fill_value=(0,),
                        dtype=(np.intp,),
                        engine=engine,
                        sort=sort,
                        user_dtype=agg.dtype["user"],
                    )["intermediates"][0]
                )

    elif agg.reduction_type == "reduce":
        # Here we reduce the intermediates individually
        results = {"groups": None, "intermediates": []}
        for idx, (combine_, fv, dtype) in enumerate(
            zip(combine, agg.fill_value["intermediate"], agg.dtype["intermediate"])
        ):
            assert combine_ is not None
            array = _conc2(x_chunk, key1="intermediates", key2=idx, axis=axis)
            if array.shape[-1] == 0:
                # all empty when combined
                results["intermediates"].append(
                    np.empty(shape=(1,) * (len(axis) - 1) + (0,), dtype=dtype)
                )
                results["groups"] = np.empty(
                    shape=(1,) * (len(neg_axis) - 1) + (0,), dtype=groups.dtype
                )
            else:
                _results = chunk_reduce(
                    array,
                    groups,
                    func=combine_,
                    axis=axis,
                    expected_groups=None,
                    fill_value=(fv,),
                    dtype=(dtype,),
                    engine=engine,
                    sort=sort,
                    user_dtype=agg.dtype["user"],
                )
                results["intermediates"].append(*_results["intermediates"])
                results["groups"] = _results["groups"]
    return results


def _reduce_blockwise(
    array,
    by,
    agg: Aggregation,
    *,
    axis: T_Axes,
    expected_groups,
    fill_value,
    engine: T_Engine,
    sort,
    reindex,
) -> FinalResultsDict:
    """
    Blockwise groupby reduction that produces the final result. This code path is
    also used for non-dask array aggregations.
    """
    # for pure numpy grouping, we just use npg directly and avoid "finalizing"
    # (agg.finalize = None). We still need to do the reindexing step in finalize
    # so that everything matches the dask version.
    agg.finalize = None

    assert agg.finalize_kwargs is not None
    finalize_kwargs_: tuple[dict[Any, Any], ...] = (agg.finalize_kwargs,) + ({},) + ({},)

    results = chunk_reduce(
        array,
        by,
        func=agg.numpy,
        axis=axis,
        expected_groups=expected_groups,
        # This fill_value should only apply to groups that only contain NaN observations
        # BUT there is funkiness when axis is a subset of all possible values
        # (see below)
        fill_value=agg.fill_value["numpy"],
        dtype=agg.dtype["numpy"],
        kwargs=finalize_kwargs_,
        engine=engine,
        sort=sort,
        reindex=reindex,
        user_dtype=agg.dtype["user"],
    )

    if _is_arg_reduction(agg):
        results["intermediates"][0] = np.unravel_index(results["intermediates"][0], array.shape)[-1]

    result = _finalize_results(
        results, agg, axis, expected_groups, fill_value=fill_value, reindex=reindex
    )
    return result


def _normalize_indexes(array: DaskArray, flatblocks, blkshape) -> tuple:
    """
    .blocks accessor can only accept one iterable at a time,
    but can handle multiple slices.
    To minimize tasks and layers, we normalize to produce slices
    along as many axes as possible, and then repeatedly apply
    any remaining iterables in a loop.

    TODO: move this upstream
    """
    unraveled = np.unravel_index(flatblocks, blkshape)

    normalized: list[int | slice | list[int]] = []
    for ax, idx in enumerate(unraveled):
        i = _unique(idx).squeeze()
        if i.ndim == 0:
            normalized.append(i.item())
        else:
            if np.array_equal(i, np.arange(blkshape[ax])):
                normalized.append(slice(None))
            elif np.array_equal(i, np.arange(i[0], i[-1] + 1)):
                normalized.append(slice(i[0], i[-1] + 1))
            else:
                normalized.append(list(i))
    full_normalized = (slice(None),) * (array.ndim - len(normalized)) + tuple(normalized)

    # has no iterables
    noiter = list(i if not hasattr(i, "__len__") else slice(None) for i in full_normalized)
    # has all iterables
    alliter = {ax: i for ax, i in enumerate(full_normalized) if hasattr(i, "__len__")}

    mesh = dict(zip(alliter.keys(), np.ix_(*alliter.values())))

    full_tuple = tuple(i if ax not in mesh else mesh[ax] for ax, i in enumerate(noiter))

    return full_tuple


def subset_to_blocks(
    array: DaskArray, flatblocks: Sequence[int], blkshape: tuple[int] | None = None
) -> DaskArray:
    """
    Advanced indexing of .blocks such that we always get a regular array back.

    Parameters
    ----------
    array : dask.array
    flatblocks : flat indices of blocks to extract
    blkshape : shape of blocks with which to unravel flatblocks

    Returns
    -------
    dask.array
    """
    import dask.array
    from dask.array.slicing import normalize_index
    from dask.base import tokenize
    from dask.highlevelgraph import HighLevelGraph

    if blkshape is None:
        blkshape = array.blocks.shape

    index = _normalize_indexes(array, flatblocks, blkshape)

    if all(not isinstance(i, np.ndarray) and i == slice(None) for i in index):
        return array

    # These rest is copied from dask.array.core.py with slight modifications
    index = normalize_index(index, array.numblocks)
    index = tuple(slice(k, k + 1) if isinstance(k, Integral) else k for k in index)

    name = "blocks-" + tokenize(array, index)
    new_keys = array._key_array[index]

    squeezed = tuple(np.squeeze(i) if isinstance(i, np.ndarray) else i for i in index)
    chunks = tuple(tuple(np.array(c)[i].tolist()) for c, i in zip(array.chunks, squeezed))

    keys = itertools.product(*(range(len(c)) for c in chunks))
    layer: Graph = {(name,) + key: tuple(new_keys[key].tolist()) for key in keys}
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[array])

    return dask.array.Array(graph, name, chunks, meta=array)


def _extract_unknown_groups(reduced, dtype) -> tuple[DaskArray]:
    import dask.array
    from dask.highlevelgraph import HighLevelGraph

    groups_token = f"group-{reduced.name}"
    first_block = reduced.ndim * (0,)
    layer: Graph = {
        (groups_token, *first_block): (operator.getitem, (reduced.name, *first_block), "groups")
    }
    groups: tuple[DaskArray] = (
        dask.array.Array(
            HighLevelGraph.from_collections(groups_token, layer, dependencies=[reduced]),
            groups_token,
            chunks=((np.nan,),),
            meta=np.array([], dtype=dtype),
        ),
    )

    return groups


def dask_groupby_agg(
    array: DaskArray,
    by: T_By,
    agg: Aggregation,
    expected_groups: T_ExpectIndexOpt,
    axis: T_Axes = (),
    fill_value: Any = None,
    method: T_Method = "map-reduce",
    reindex: bool = False,
    engine: T_Engine = "numpy",
    sort: bool = True,
    chunks_cohorts=None,
) -> tuple[DaskArray, tuple[np.ndarray | DaskArray]]:
    import dask.array
    from dask.array.core import slices_from_chunks

    # I think _tree_reduce expects this
    assert isinstance(axis, Sequence)
    assert all(ax >= 0 for ax in axis)

    inds = tuple(range(array.ndim))
    name = f"groupby_{agg.name}"
    token = dask.base.tokenize(array, by, agg, expected_groups, axis)

    if expected_groups is None and reindex:
        expected_groups = _get_expected_groups(by, sort=sort)
    if method == "cohorts":
        assert reindex is False

    by_input = by

    # Unifying chunks is necessary for argreductions.
    # We need to rechunk before zipping up with the index
    # let's always do it anyway
    if not is_duck_dask_array(by):
        # chunk numpy arrays like the input array
        # This removes an extra rechunk-merge layer that would be
        # added otherwise
        chunks = tuple(array.chunks[ax] if by.shape[ax] != 1 else (1,) for ax in range(-by.ndim, 0))

        by = dask.array.from_array(by, chunks=chunks)
    _, (array, by) = dask.array.unify_chunks(array, inds, by, inds[-by.ndim :])

    # preprocess the array:
    #   - for argreductions, this zips the index together with the array block
    #   - not necessary for blockwise with argreductions
    #   - if this is needed later, we can fix this then
    if agg.preprocess and method != "blockwise":
        array = agg.preprocess(array, axis=axis)

    # 1. We first apply the groupby-reduction blockwise to generate "intermediates"
    # 2. These intermediate results are combined to generate the final result using a
    #    "map-reduce" or "tree reduction" approach.
    #    There are two ways:
    #    a. "_simple_combine": Where it makes sense, we tree-reduce the reduction,
    #        NOT the groupby-reduction for a speed boost. This is what xhistogram does (effectively),
    #        It requires that all blocks contain all groups after the initial blockwise step (1) i.e.
    #        reindex=True, and we must know expected_groups
    #    b. "_grouped_combine": A more general solution where we tree-reduce the groupby reduction.
    #       This allows us to discover groups at compute time, support argreductions, lower intermediate
    #       memory usage (but method="cohorts" would also work to reduce memory in some cases)
    do_simple_combine = not _is_arg_reduction(agg)

    if method == "blockwise":
        #  use the "non dask" code path, but applied blockwise
        blockwise_method = partial(
            _reduce_blockwise, agg=agg, fill_value=fill_value, reindex=reindex
        )
    else:
        # choose `chunk_reduce` or `chunk_argreduce`
        blockwise_method = partial(
            _get_chunk_reduction(agg.reduction_type),
            func=agg.chunk,
            fill_value=agg.fill_value["intermediate"],
            dtype=agg.dtype["intermediate"],
            reindex=reindex,
            user_dtype=agg.dtype["user"],
        )
        if do_simple_combine:
            # Add a dummy dimension that then gets reduced over
            blockwise_method = tlz.compose(_expand_dims, blockwise_method)

    # apply reduction on chunk
    intermediate = dask.array.blockwise(
        partial(
            blockwise_method,
            axis=axis,
            expected_groups=expected_groups if reindex else None,
            engine=engine,
            sort=sort,
        ),
        # output indices are the same as input indices
        # Unlike xhistogram, we don't always know what the size of the group
        # dimension will be unless reindex=True
        inds,
        array,
        inds,
        by,
        inds[-by.ndim :],
        concatenate=False,
        dtype=array.dtype,  # this is purely for show
        meta=array._meta,
        align_arrays=False,
        name=f"{name}-chunk-{token}",
    )

    group_chunks: tuple[tuple[int | float, ...]]

    if method in ["map-reduce", "cohorts"]:
        combine: Callable[..., IntermediateDict]
        if do_simple_combine:
            combine = partial(_simple_combine, reindex=reindex)
            combine_name = "simple-combine"
        else:
            combine = partial(_grouped_combine, engine=engine, sort=sort)
            combine_name = "grouped-combine"

        tree_reduce = partial(
            dask.array.reductions._tree_reduce,
            name=f"{name}-reduce-{method}-{combine_name}",
            dtype=array.dtype,
            axis=axis,
            keepdims=True,
            concatenate=False,
        )
        aggregate = partial(_aggregate, combine=combine, agg=agg, fill_value=fill_value)

        # Each chunk of `reduced`` is really a dict mapping
        # 1. reduction name to array
        # 2. "groups" to an array of group labels
        # Note: it does not make sense to interpret axis relative to
        # shape of intermediate results after the blockwise call
        if method == "map-reduce":
            reduced = tree_reduce(
                intermediate,
                combine=partial(combine, agg=agg),
                aggregate=partial(aggregate, expected_groups=expected_groups, reindex=reindex),
            )
            if is_duck_dask_array(by_input) and expected_groups is None:
                groups = _extract_unknown_groups(reduced, dtype=by.dtype)
                group_chunks = ((np.nan,),)
            else:
                if expected_groups is None:
                    expected_groups_ = _get_expected_groups(by_input, sort=sort)
                else:
                    expected_groups_ = expected_groups
                groups = (expected_groups_.to_numpy(),)
                group_chunks = ((len(expected_groups_),),)

        elif method == "cohorts":
            chunks_cohorts = find_group_cohorts(
                by_input, [array.chunks[ax] for ax in axis], merge=True
            )
            reduced_ = []
            groups_ = []
            for blks, cohort in chunks_cohorts.items():
                index = pd.Index(cohort)
                subset = subset_to_blocks(intermediate, blks, array.blocks.shape[-len(axis) :])
                reindexed = dask.array.map_blocks(
                    reindex_intermediates, subset, agg=agg, unique_groups=index, meta=subset._meta
                )
                # now that we have reindexed, we can set reindex=True explicitlly
                reduced_.append(
                    tree_reduce(
                        reindexed,
                        combine=partial(combine, agg=agg, reindex=True),
                        aggregate=partial(aggregate, expected_groups=index, reindex=True),
                    )
                )
                # This is done because pandas promotes to 64-bit types when an Index is created
                # So we use the index to generate the return value for consistency with "map-reduce"
                # This is important on windows
                groups_.append(index.values)

            reduced = dask.array.concatenate(reduced_, axis=-1)
            groups = (np.concatenate(groups_),)
            group_chunks = (tuple(len(cohort) for cohort in groups_),)

    elif method == "blockwise":
        reduced = intermediate
        if reindex:
            if TYPE_CHECKING:
                assert expected_groups is not None
            # TODO: we could have `expected_groups` be a dask array with appropriate chunks
            # for now, we have a numpy array that is interpreted as listing all group labels
            # that are present in every chunk
            groups = (expected_groups,)
            group_chunks = ((len(expected_groups),),)
        else:
            # Here one input chunk → one output chunks
            # find number of groups in each chunk, this is needed for output chunks
            # along the reduced axis
            # TODO: this logic is very specialized for the resampling case
            slices = slices_from_chunks(tuple(array.chunks[ax] for ax in axis))
            groups_in_block = tuple(_unique(by_input[slc]) for slc in slices)
            groups = (np.concatenate(groups_in_block),)
            ngroups_per_block = tuple(len(grp) for grp in groups_in_block)
            group_chunks = (ngroups_per_block,)
    else:
        raise ValueError(f"Unknown method={method}.")

    out_inds = inds[: -len(axis)] + (inds[-1],)
    output_chunks = reduced.chunks[: -len(axis)] + group_chunks
    if method == "blockwise" and len(axis) > 1:
        # The final results are available but the blocks along axes
        # need to be reshaped to axis=-1
        # I don't know that this is possible with blockwise
        # All other code paths benefit from an unmaterialized Blockwise layer
        reduced = _collapse_blocks_along_axes(reduced, axis, group_chunks)

    # Can't use map_blocks because it forces concatenate=True along drop_axes,
    result = dask.array.blockwise(
        _extract_result,
        out_inds,
        reduced,
        inds,
        adjust_chunks=dict(zip(out_inds, output_chunks)),
        dtype=agg.dtype["final"],
        key=agg.name,
        name=f"{name}-{token}",
        concatenate=False,
    )

    return (result, groups)


def _collapse_blocks_along_axes(reduced: DaskArray, axis: T_Axes, group_chunks) -> DaskArray:
    import dask.array
    from dask.highlevelgraph import HighLevelGraph

    nblocks = tuple(reduced.numblocks[ax] for ax in axis)
    output_chunks = reduced.chunks[: -len(axis)] + ((1,) * (len(axis) - 1),) + group_chunks

    # extract results from the dict
    ochunks = tuple(range(len(chunks_v)) for chunks_v in output_chunks)
    layer2: dict[tuple, tuple] = {}
    name = f"reshape-{reduced.name}"

    for ochunk in itertools.product(*ochunks):
        inchunk = ochunk[: -len(axis)] + np.unravel_index(ochunk[-1], nblocks)
        layer2[(name, *ochunk)] = (reduced.name, *inchunk)

    layer2: Graph
    return dask.array.Array(
        HighLevelGraph.from_collections(name, layer2, dependencies=[reduced]),
        name,
        chunks=output_chunks,
        dtype=reduced.dtype,
    )


def _extract_result(result_dict: FinalResultsDict, key) -> np.ndarray:
    from dask.array.core import deepfirst

    # deepfirst should be not be needed here but sometimes we receive a list of dict?
    return deepfirst(result_dict)[key]


def _validate_reindex(
    reindex: bool | None,
    func,
    method: T_Method,
    expected_groups,
    any_by_dask: bool,
    is_dask_array: bool,
) -> bool:
    all_numpy = not is_dask_array and not any_by_dask
    if reindex is True and not all_numpy:
        if _is_arg_reduction(func):
            raise NotImplementedError
        if method == "cohorts" or (method == "blockwise" and not any_by_dask):
            raise ValueError(
                "reindex=True is not a valid choice for method='blockwise' or method='cohorts'."
            )
        if func in ["first", "last"]:
            raise ValueError("reindex must be None or False when func is 'first' or 'last.")

    if reindex is None:
        if all_numpy:
            return True

        if func in ["first", "last"]:
            # have to do the grouped_combine since there's no good fill_value
            reindex = False

        if method == "blockwise":
            # for grouping by dask arrays, we set reindex=True
            reindex = any_by_dask

        elif _is_arg_reduction(func):
            reindex = False

        elif method == "cohorts":
            reindex = False

        elif method == "map-reduce":
            if expected_groups is None and any_by_dask:
                reindex = False
            else:
                reindex = True

    assert isinstance(reindex, bool)

    return reindex


def _assert_by_is_aligned(shape: tuple[int, ...], by: T_Bys) -> None:
    assert all(b.ndim == by[0].ndim for b in by[1:])
    for idx, b in enumerate(by):
        if not all(j in [i, 1] for i, j in zip(shape[-b.ndim :], b.shape)):
            raise ValueError(
                "`array` and `by` arrays must be 'aligned' "
                "so that such that by_ is broadcastable to array.shape[-by.ndim:] "
                "for every array `by_` in `by`. "
                "Either array.shape[-by_.ndim :] == by_.shape or the only differences "
                "should be size-1 dimensions in by_."
                f"Received array of shape {shape} but "
                f"array {idx} in `by` has shape {b.shape}."
            )


@overload
def _convert_expected_groups_to_index(
    expected_groups: tuple[None, ...], isbin: Sequence[bool], sort: bool
) -> tuple[None, ...]:
    ...


@overload
def _convert_expected_groups_to_index(
    expected_groups: T_ExpectTuple, isbin: Sequence[bool], sort: bool
) -> T_ExpectIndexTuple:
    ...


def _convert_expected_groups_to_index(
    expected_groups: T_ExpectOptTuple, isbin: Sequence[bool], sort: bool
) -> T_ExpectIndexOptTuple:
    out: list[T_ExpectIndexOpt] = []
    for ex, isbin_ in zip(expected_groups, isbin):
        if isinstance(ex, pd.IntervalIndex) or (isinstance(ex, pd.Index) and not isbin_):
            if sort:
                out.append(ex.sort_values())
            else:
                out.append(ex)
        elif ex is not None:
            if isbin_:
                out.append(pd.IntervalIndex.from_breaks(ex))
            else:
                if sort:
                    ex = np.sort(ex)
                out.append(pd.Index(ex))
        else:
            assert ex is None
            out.append(None)
    return tuple(out)


def _lazy_factorize_wrapper(*by: T_By, **kwargs) -> np.ndarray:
    group_idx, *rest = factorize_(by, **kwargs)
    return group_idx


def _factorize_multiple(
    by: T_Bys,
    expected_groups: T_ExpectIndexOptTuple,
    any_by_dask: bool,
    reindex: bool,
    sort: bool = True,
) -> tuple[tuple[np.ndarray], tuple[np.ndarray, ...], tuple[int, ...]]:
    if any_by_dask:
        import dask.array

        # unifying chunks will make sure all arrays in `by` are dask arrays
        # with compatible chunks, even if there was originally a numpy array
        inds = tuple(range(by[0].ndim))
        chunks, by_ = dask.array.unify_chunks(*itertools.chain(*zip(by, (inds,) * len(by))))

        group_idx = dask.array.map_blocks(
            _lazy_factorize_wrapper,
            *by_,
            chunks=tuple(chunks.values()),
            meta=np.array((), dtype=np.int64),
            axes=(),  # always (), we offset later if necessary.
            expected_groups=expected_groups,
            fastpath=True,
            reindex=reindex,
            sort=sort,
        )

        fg, gs = [], []
        for by_, expect in zip(by, expected_groups):
            if expect is None:
                if is_duck_dask_array(by_):
                    raise ValueError(
                        "Please provide expected_groups when grouping by a dask array."
                    )

                found_group = pd.unique(by_.reshape(-1))
            else:
                found_group = expect.to_numpy()

            fg.append(found_group)
            gs.append(len(found_group))

        found_groups = tuple(fg)
        grp_shape = tuple(gs)
    else:
        group_idx, found_groups, grp_shape, ngroups, size, props = factorize_(
            by,
            axes=(),  # always (), we offset later if necessary.
            expected_groups=expected_groups,
            fastpath=True,
            reindex=reindex,
            sort=sort,
        )

    return (group_idx,), found_groups, grp_shape


@overload
def _validate_expected_groups(nby: int, expected_groups: None) -> tuple[None, ...]:
    ...


@overload
def _validate_expected_groups(nby: int, expected_groups: T_ExpectedGroups) -> T_ExpectTuple:
    ...


def _validate_expected_groups(nby: int, expected_groups: T_ExpectedGroupsOpt) -> T_ExpectOptTuple:
    if expected_groups is None:
        return (None,) * nby

    if nby == 1 and not isinstance(expected_groups, tuple):
        if isinstance(expected_groups, (pd.Index, np.ndarray)):
            return (expected_groups,)
        else:
            array = np.asarray(expected_groups)
            if np.issubdtype(array.dtype, np.integer):
                # preserve default dtypes
                # on pandas 1.5/2, on windows
                # when a list is passed
                array = array.astype(np.int64)
            return (array,)

    if nby > 1 and not isinstance(expected_groups, tuple):  # TODO: test for list
        raise ValueError(
            "When grouping by multiple variables, expected_groups must be a tuple "
            "of either arrays or objects convertible to an array (like lists). "
            "For example `expected_groups=(np.array([1, 2, 3]), ['a', 'b', 'c'])`."
            f"Received a {type(expected_groups).__name__} instead. "
            "When grouping by a single variable, you can pass an array or something "
            "convertible to an array for convenience: `expected_groups=['a', 'b', 'c']`."
        )

    if TYPE_CHECKING:
        assert isinstance(expected_groups, tuple)

    if len(expected_groups) != nby:
        raise ValueError(
            f"Must have same number of `expected_groups` (received {len(expected_groups)}) "
            f" and variables to group by (received {nby})."
        )

    return expected_groups


def _choose_engine(by, agg: Aggregation):
    dtype = agg.dtype["user"]

    not_arg_reduce = not _is_arg_reduction(agg)

    # numbagg only supports nan-skipping reductions
    # without dtype specified
    has_blockwise_nan_skipping = (agg.chunk[0] is None and "nan" in agg.name) or any(
        (isinstance(func, str) and "nan" in func) for func in agg.chunk
    )
    if HAS_NUMBAGG:
        if agg.name in ["all", "any"] or (
            not_arg_reduce and has_blockwise_nan_skipping and dtype is None
        ):
            return "numbagg"

    if not_arg_reduce and (not is_duck_dask_array(by) and _issorted(by)):
        return "flox"
    else:
        return "numpy"


def groupby_reduce(
    array: np.ndarray | DaskArray,
    *by: T_By,
    func: T_Agg,
    expected_groups: T_ExpectedGroupsOpt = None,
    sort: bool = True,
    isbin: T_IsBins = False,
    axis: T_AxesOpt = None,
    fill_value=None,
    dtype: np.typing.DTypeLike = None,
    min_count: int | None = None,
    method: T_Method = "map-reduce",
    engine: T_EngineOpt = None,
    reindex: bool | None = None,
    finalize_kwargs: dict[Any, Any] | None = None,
) -> tuple[DaskArray, Unpack[tuple[np.ndarray | DaskArray, ...]]]:  # type: ignore[misc]  # Unpack not in mypy yet
    """
    GroupBy reductions using tree reductions for dask.array

    Parameters
    ----------
    array : ndarray or DaskArray
        Array to be reduced, possibly nD
    *by : ndarray or DaskArray
        Array of labels to group over. Must be aligned with ``array`` so that
        ``array.shape[-by.ndim :] == by.shape`` or any disagreements in that
        equality check are for dimensions of size 1 in `by`.
    func : {"all", "any", "count", "sum", "nansum", "mean", "nanmean", \
            "max", "nanmax", "min", "nanmin", "argmax", "nanargmax", "argmin", "nanargmin", \
            "quantile", "nanquantile", "median", "nanmedian", "mode", "nanmode", \
            "first", "nanfirst", "last", "nanlast"} or Aggregation
        Single function name or an Aggregation instance
    expected_groups : (optional) Sequence
        Expected unique labels.
    isbin : bool, optional
        Are ``expected_groups`` bin edges?
    sort : bool, optional
        Whether groups should be returned in sorted order. Only applies for dask
        reductions when ``method`` is not ``"map-reduce"``. For ``"map-reduce"``, the groups
        are always sorted.
    axis : None or int or Sequence[int], optional
        If None, reduce across all dimensions of by
        Else, reduce across corresponding axes of array
        Negative integers are normalized using array.ndim
    fill_value : Any
        Value to assign when a label in ``expected_groups`` is not present.
    dtype : data-type , optional
        DType for the output. Can be anything that is accepted by ``np.dtype``.
    min_count : int, default: None
        The required number of valid values to perform the operation. If
        fewer than min_count non-NA values are present the result will be
        NA. Only used if skipna is set to True or defaults to True for the
        array's dtype.
    method : {"map-reduce", "blockwise", "cohorts"}, optional
        Strategy for reduction of dask arrays only:
          * ``"map-reduce"``:
            First apply the reduction blockwise on ``array``, then
            combine a few newighbouring blocks, apply the reduction.
            Continue until finalizing. Usually, ``func`` will need
            to be an Aggregation instance for this method to work.
            Common aggregations are implemented.
          * ``"blockwise"``:
            Only reduce using blockwise and avoid aggregating blocks
            together. Useful for resampling-style reductions where group
            members are always together. If  `by` is 1D,  `array` is automatically
            rechunked so that chunk boundaries line up with group boundaries
            i.e. each block contains all members of any group present
            in that block. For nD `by`, you must make sure that all members of a group
            are present in a single block.
          * ``"cohorts"``:
            Finds group labels that tend to occur together ("cohorts"),
            indexes out cohorts and reduces that subset using "map-reduce",
            repeat for all cohorts. This works well for many time groupings
            where the group labels repeat at regular intervals like 'hour',
            'month', dayofyear' etc. Optimize chunking ``array`` for this
            method by first rechunking using ``rechunk_for_cohorts``
            (for 1D ``by`` only).
    engine : {"flox", "numpy", "numba", "numbagg"}, optional
        Algorithm to compute the groupby reduction on non-dask arrays and on each dask chunk:
          * ``"numpy"``:
            Use the vectorized implementations in ``numpy_groupies.aggregate_numpy``.
            This is the default choice because it works for most array types.
          * ``"flox"``:
            Use an internal implementation where the data is sorted so that
            all members of a group occur sequentially, and then numpy.ufunc.reduceat
            is to used for the reduction. This will fall back to ``numpy_groupies.aggregate_numpy``
            for a reduction that is not yet implemented.
          * ``"numba"``:
            Use the implementations in ``numpy_groupies.aggregate_numba``.
          * ``"numbagg"``:
            Use the reductions supported by ``numbagg.grouped``. This will fall back to ``numpy_groupies.aggregate_numpy``
            for a reduction that is not yet implemented.
    reindex : bool, optional
        Whether to "reindex" the blockwise results to ``expected_groups`` (possibly automatically detected).
        If True, the intermediate result of the blockwise groupby-reduction has a value for all expected groups,
        and the final result is a simple reduction of those intermediates. In nearly all cases, this is a significant
        boost in computation speed. For cases like time grouping, this may result in large intermediates relative to the
        original block size. Avoid that by using ``method="cohorts"``. By default, it is turned off for argreductions.
    finalize_kwargs : dict, optional
        Kwargs passed to finalize the reduction such as ``ddof`` for var, std or ``q`` for quantile.

    Returns
    -------
    result
        Aggregated result
    *groups
        Group labels

    See Also
    --------
    xarray.xarray_reduce
    """

    if engine == "flox" and _is_arg_reduction(func):
        raise NotImplementedError(
            "argreductions not supported for engine='flox' yet."
            "Try engine='numpy' or engine='numba' instead."
        )

    if engine == "numbagg" and dtype is not None:
        raise NotImplementedError(
            "numbagg does not support the `dtype` kwarg. Either cast your "
            "input arguments to `dtype` or use a different `engine`: "
            "'flox' or 'numpy' or 'numba'. "
            "See https://github.com/numbagg/numbagg/issues/121."
        )

    if func == "quantile" and (finalize_kwargs is None or "q" not in finalize_kwargs):
        raise ValueError("Please pass `q` for quantile calculations.")

    bys: T_Bys = tuple(np.asarray(b) if not is_duck_array(b) else b for b in by)
    nby = len(bys)
    by_is_dask = tuple(is_duck_dask_array(b) for b in bys)
    any_by_dask = any(by_is_dask)

    if (
        engine == "numbagg"
        and _is_arg_reduction(func)
        and (any_by_dask or is_duck_dask_array(array))
    ):
        # There is only one test that fails, but I can't figure
        # out why without deep debugging.
        # just disable for now.
        # test_groupby_reduce_axis_subset_against_numpy
        # for array is 3D dask, by is 3D dask, axis=2
        # We are falling back to numpy for the arg reduction,
        # so presumably something is going wrong
        raise NotImplementedError(
            "argreductions not supported for engine='numbagg' yet."
            "Try engine='numpy' or engine='numba' instead."
        )

    if method == "cohorts" and any_by_dask:
        raise ValueError(f"method={method!r} can only be used when grouping by numpy arrays.")

    reindex = _validate_reindex(
        reindex, func, method, expected_groups, any_by_dask, is_duck_dask_array(array)
    )

    if not is_duck_array(array):
        array = np.asarray(array)
    is_bool_array = np.issubdtype(array.dtype, bool)
    array = array.astype(int) if is_bool_array else array

    if isinstance(isbin, Sequence):
        isbins = isbin
    else:
        isbins = (isbin,) * nby

    _assert_by_is_aligned(array.shape, bys)

    expected_groups = _validate_expected_groups(nby, expected_groups)

    for idx, (expect, is_dask) in enumerate(zip(expected_groups, by_is_dask)):
        if is_dask and (reindex or nby > 1) and expect is None:
            raise ValueError(
                f"`expected_groups` for array {idx} in `by` cannot be None since it is a dask.array."
            )

    # We convert to pd.Index since that lets us know if we are binning or not
    # (pd.IntervalIndex or not)
    expected_groups = _convert_expected_groups_to_index(expected_groups, isbins, sort)

    # Don't factorize "early only when
    # grouping by dask arrays, and not having expected_groups
    factorize_early = not (
        # can't do it if we are grouping by dask array but don't have expected_groups
        any(is_dask and ex_ is None for is_dask, ex_ in zip(by_is_dask, expected_groups))
    )
    if factorize_early:
        bys, final_groups, grp_shape = _factorize_multiple(
            bys,
            expected_groups,
            any_by_dask=any_by_dask,
            # This is the only way it makes sense I think.
            # reindex controls what's actually allocated in chunk_reduce
            # At this point, we care about an accurate conversion to codes.
            reindex=True,
            sort=sort,
        )
        expected_groups = (pd.RangeIndex(math.prod(grp_shape)),)

    assert len(bys) == 1
    (by_,) = bys
    (expected_groups,) = expected_groups

    if axis is None:
        axis_ = tuple(array.ndim + np.arange(-by_.ndim, 0))
    else:
        # TODO: How come this function doesn't exist according to mypy?
        axis_ = np.core.numeric.normalize_axis_tuple(axis, array.ndim)  # type: ignore[attr-defined]
    nax = len(axis_)

    has_dask = is_duck_dask_array(array) or is_duck_dask_array(by_)

    if _is_first_last_reduction(func):
        if has_dask and nax != 1:
            raise ValueError(
                "For dask arrays: first, last, nanfirst, nanlast reductions are "
                "only supported along a single axis. Please reshape appropriately."
            )

        elif nax not in [1, by_.ndim]:
            raise ValueError(
                "first, last, nanfirst, nanlast reductions are only supported "
                "along a single axis or when reducing across all dimensions of `by`."
            )

    # TODO: make sure expected_groups is unique
    if nax == 1 and by_.ndim > 1 and expected_groups is None:
        if not any_by_dask:
            expected_groups = _get_expected_groups(by_, sort)
        else:
            # When we reduce along all axes, we are guaranteed to see all
            # groups in the final combine stage, so everything works.
            # This is not necessarily true when reducing along a subset of axes
            # (of by)
            # TODO: Does this depend on chunking of by?
            # For e.g., we could relax this if there is only one chunk along all
            # by dim != axis?
            raise NotImplementedError(
                "Please provide ``expected_groups`` when not reducing along all axes."
            )

    assert nax <= by_.ndim
    if nax < by_.ndim:
        by_ = _move_reduce_dims_to_end(by_, tuple(-array.ndim + ax + by_.ndim for ax in axis_))
        array = _move_reduce_dims_to_end(array, axis_)
        axis_ = tuple(array.ndim + np.arange(-nax, 0))
        nax = len(axis_)

    # When axis is a subset of possible values; then npg will
    # apply the fill_value to groups that don't exist along a particular axis (for e.g.)
    # since these count as a group that is absent. thoo!
    # fill_value applies to all-NaN groups as well as labels in expected_groups that are not found.
    #     The only way to do this consistently is mask out using min_count
    #     Consider np.sum([np.nan]) = np.nan, np.nansum([np.nan]) = 0
    if min_count is None:
        if nax < by_.ndim or fill_value is not None:
            min_count_: int = 1
        else:
            min_count_ = 0
    else:
        min_count_ = min_count

    # TODO: set in xarray?
    if min_count_ > 0 and func in ["nansum", "nanprod"] and fill_value is None:
        # nansum, nanprod have fill_value=0, 1
        # overwrite than when min_count is set
        fill_value = np.nan

    kwargs = dict(axis=axis_, fill_value=fill_value)
    agg = _initialize_aggregation(func, dtype, array.dtype, fill_value, min_count_, finalize_kwargs)

    # Need to set this early using `agg`
    # It cannot be done in the core loop of chunk_reduce
    # since we "prepare" the data for flox.
    kwargs["engine"] = _choose_engine(by_, agg) if engine is None else engine

    groups: tuple[np.ndarray | DaskArray, ...]
    if not has_dask:
        results = _reduce_blockwise(
            array, by_, agg, expected_groups=expected_groups, reindex=reindex, sort=sort, **kwargs
        )
        groups = (results["groups"],)
        result = results[agg.name]

    else:
        if TYPE_CHECKING:
            # TODO: How else to narrow that array.chunks is there?
            assert isinstance(array, DaskArray)

        if agg.chunk[0] is None and method != "blockwise":
            raise NotImplementedError(
                f"Aggregation {agg.name!r} is only implemented for dask arrays when method='blockwise'."
                f"Received method={method!r}"
            )

        if method in ["blockwise", "cohorts"] and nax != by_.ndim:
            raise NotImplementedError(
                "Must reduce along all dimensions of `by` when method != 'map-reduce'."
                f"Received method={method!r}"
            )

        # TODO: just do this in dask_groupby_agg
        # we always need some fill_value (see above) so choose the default if needed
        if kwargs["fill_value"] is None:
            kwargs["fill_value"] = agg.fill_value[agg.name]

        partial_agg = partial(dask_groupby_agg, **kwargs)

        if method == "blockwise" and by_.ndim == 1:
            array = rechunk_for_blockwise(array, axis=-1, labels=by_)

        result, groups = partial_agg(
            array,
            by_,
            expected_groups=expected_groups,
            agg=agg,
            reindex=reindex,
            method=method,
            sort=sort,
        )

        if sort and method != "map-reduce":
            assert len(groups) == 1
            sorted_idx = np.argsort(groups[0])
            # This optimization helps specifically with resampling
            if not _issorted(sorted_idx):
                result = result[..., sorted_idx]
                groups = (groups[0][sorted_idx],)

    if factorize_early:
        # nan group labels are factorized to -1, and preserved
        # now we get rid of them by reindexing
        # This also handles bins with no data
        result = reindex_(
            result, from_=groups[0], to=expected_groups, fill_value=fill_value
        ).reshape(result.shape[:-1] + grp_shape)
        groups = final_groups

    if is_bool_array and (_is_minmax_reduction(func) or _is_first_last_reduction(func)):
        result = result.astype(bool)
    return (result, *groups)  # type: ignore[return-value]  # Unpack not in mypy yet
