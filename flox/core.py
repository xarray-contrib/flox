import copy
import itertools
import operator
from collections import namedtuple
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy_groupies as npg
import pandas as pd

from . import aggregations, xrdtypes
from .aggregations import Aggregation, _atleast_1d, _get_fill_value, generic_aggregate
from .cache import memoize
from .xrutils import is_duck_array, is_duck_dask_array, isnull

if TYPE_CHECKING:
    import dask.array.Array as DaskArray


IntermediateDict = Dict[Union[str, Callable], Any]
FinalResultsDict = Dict[str, Union["DaskArray", np.ndarray]]
FactorProps = namedtuple("FactorProps", "offset_group nan_sentinel nanmask")


def _prepare_for_flox(group_idx, array):
    """
    Sort the input array once to save time.
    """
    issorted = (group_idx[:-1] <= group_idx[1:]).all()
    if issorted:
        ordered_array = array
    else:
        perm = group_idx.argsort(kind="stable")
        group_idx = group_idx[..., perm]
        ordered_array = array[..., perm]
    return group_idx, ordered_array


def _normalize_dtype(dtype, array_dtype, fill_value=None):
    if dtype is None:
        if fill_value is not None and np.isnan(fill_value):
            dtype = np.floating
        else:
            dtype = array_dtype
    elif dtype is np.floating:
        # mean, std, var always result in floating
        # but we preserve the array's dtype if it is floating
        if array_dtype.kind in "fcmM":
            dtype = array_dtype
        else:
            dtype = np.dtype("float64")
    elif not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    return dtype


def _get_chunk_reduction(reduction_type: str) -> Callable:
    if reduction_type == "reduce":
        return chunk_reduce
    elif reduction_type == "argreduce":
        return chunk_argreduce
    else:
        raise ValueError(f"Unknown reduction type: {reduction_type}")


def _move_reduce_dims_to_end(arr: np.ndarray, axis: Sequence) -> np.ndarray:
    """Transpose `arr` by moving `axis` to the end."""
    axis = tuple(axis)
    order = tuple(ax for ax in np.arange(arr.ndim) if ax not in axis) + axis
    arr = arr.transpose(order)
    return arr


def _collapse_axis(arr: np.ndarray, naxis: int) -> np.ndarray:
    """Reshape so that the last `naxis` axes are collapsed to one axis."""
    newshape = arr.shape[:-naxis] + (np.prod(arr.shape[-naxis:]),)
    return arr.reshape(newshape)


def _get_optimal_chunks_for_groups(chunks, labels):
    chunkidx = np.cumsum(chunks) - 1
    # what are the groups at chunk boundaries
    labels_at_chunk_bounds = np.unique(labels[chunkidx])
    # what's the last index of all groups
    last_indexes = npg.aggregate_numpy.aggregate(labels, np.arange(len(labels)), func="last")
    # what's the last index of groups at the chunk boundaries.
    lastidx = last_indexes[labels_at_chunk_bounds]

    if len(chunkidx) == len(lastidx) and (chunkidx == lastidx).all():
        return chunks

    first_indexes = npg.aggregate_numpy.aggregate(labels, np.arange(len(labels)), func="first")
    firstidx = first_indexes[labels_at_chunk_bounds]

    newchunkidx = [0]
    for c, f, l in zip(chunkidx, firstidx, lastidx):
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


@memoize
def find_group_cohorts(labels, chunks, merge=True, method="cohorts"):
    """
    Finds groups labels that occur together aka "cohorts"

    If available, results are cached in a 1MB cache managed by `cachey`.
    This allows us to be quick when repeatedly calling groupby_reduce
    for arrays with the same chunking (e.g. an xarray Dataset).

    Parameters
    ----------
    labels: np.ndarray
        mD Array of group labels
    array: tuple
        nD array that is being reduced
    merge: bool, optional
        Attempt to merge cohorts when one cohort's chunks are a subset
        of another cohort's chunks.
    method: ["split-reduce", "cohorts"], optional
        Which method are we using?

    Returns
    -------
    cohorts: dict_values
        Iterable of cohorts
    """
    import copy

    import dask
    import toolz as tlz

    if method == "split-reduce":
        return np.unique(labels).reshape(-1, 1).tolist()

    # To do this, we must have values in memory so casting to numpy should be safe
    labels = np.asarray(labels)

    # Build an array with the shape of labels, but where every element is the "chunk number"
    # 1. First subset the array appropriately
    axis = range(-labels.ndim, 0)
    # Easier to create a dask array and use the .blocks property
    array = dask.array.ones(tuple(sum(c) for c in chunks), chunks=chunks)

    #  Iterate over each block and create a new block of same shape with "chunk number"
    shape = tuple(array.blocks.shape[ax] for ax in axis)
    blocks = np.empty(np.prod(shape), dtype=object)
    for idx, block in enumerate(array.blocks.ravel()):
        blocks[idx] = np.full(tuple(block.shape[ax] for ax in axis), idx)
    which_chunk = np.block(blocks.reshape(shape).tolist()).ravel()

    # these are chunks where a label is present
    label_chunks = {
        lab: tuple(np.unique(which_chunk[labels.ravel() == lab])) for lab in np.unique(labels)
    }
    # These invert the label_chunks mapping so we know which labels occur together.
    chunks_cohorts = tlz.groupby(label_chunks.get, label_chunks.keys())

    if merge:
        # First sort by number of chunks occupied by cohort
        sorted_chunks_cohorts = dict(
            reversed(sorted(chunks_cohorts.items(), key=lambda kv: len(kv[0])))
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

        return merged_cohorts.values()
    else:
        return chunks_cohorts.values()


def rechunk_for_cohorts(array, axis, labels, force_new_chunk_at, chunksize=None):
    """
    Rechunks array so that each new chunk contains groups that always occur together.

    Parameters
    ----------
    array: dask.array.Array
        array to rechunk
    axis: int
        Axis to rechunk
    labels: np.array
        1D Group labels to align chunks with. This routine works
        well when ``labels`` has repeating patterns: e.g.
        ``1, 2, 3, 1, 2, 3, 4, 1, 2, 3`` though there is no requirement
        that the pattern must contain sequences.
    force_new_chunk_at:
        label at which we always start a new chunk. For
        the example ``labels`` array, this would be `1``.
    chunksize: int, optional
        nominal chunk size. Chunk size is exceded when the label
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

    isbreak = np.isin(labels, force_new_chunk_at)
    if not np.any(isbreak):
        raise ValueError("One or more labels in ``force_new_chunk_at`` not present in ``labels``.")

    divisions = []
    counter = 1
    for idx, lab in enumerate(labels):
        if lab in force_new_chunk_at:
            divisions.append(idx)
            counter = 1
            continue

        next_break = np.nonzero(isbreak[idx:])[0]
        if next_break.any():
            next_break_is_close = next_break[0] <= chunksize // 2
        else:
            next_break_is_close = False

        if idx in oldbreaks or (counter >= chunksize and not next_break_is_close):
            divisions.append(idx)
            counter = 1
            continue
        counter += 1

    divisions.append(len(labels))
    newchunks = tuple(np.diff(divisions))
    assert sum(newchunks) == len(labels)

    if newchunks == array.chunks[axis]:
        return array
    else:
        return array.rechunk({axis: newchunks})


def rechunk_for_blockwise(array, axis, labels):
    """
    Rechunks array so that group boundaries line up with chunk boundaries, allowing
    parallel group reductions.

    This only works when the groups are sequential (e.g. labels = [0,0,0,1,1,1,1,2,2]).
    Such patterns occur when using ``.resample``.
    """
    labels = factorize_((labels,), axis=None)[0]
    chunks = array.chunks[axis]
    # TODO: lru_cache this?
    newchunks = _get_optimal_chunks_for_groups(chunks, labels)
    if newchunks == chunks:
        return array
    else:
        return array.rechunk({axis: newchunks})


def reindex_(array: np.ndarray, from_, to, fill_value=None, axis: int = -1) -> np.ndarray:

    assert axis in (0, -1)

    from_ = np.atleast_1d(from_)
    to = np.atleast_1d(to)
    # short-circuit for trivial case
    if len(from_) == len(to) and np.all(from_ == to):
        return array

    if array.shape[axis] == 0:
        # all groups were NaN
        reindexed = np.full(array.shape[:-1] + (len(to),), fill_value, dtype=array.dtype)
        return reindexed

    if from_.dtype.kind == "O" and isinstance(from_[0], tuple):
        raise NotImplementedError(
            "Currently does not support reindexing with object arrays of tuples. "
            "These occur when grouping by multi-indexed variables in xarray."
        )
    idx = np.array(
        [np.argwhere(np.array(from_) == label)[0, 0] if label in from_ else -1 for label in to]
    )
    indexer = [slice(None, None)] * array.ndim
    indexer[axis] = idx  # type: ignore
    reindexed = array[tuple(indexer)]
    if any(idx == -1):
        if fill_value is None:
            raise ValueError("Filling is required. fill_value cannot be None.")
        if axis == 0:
            loc = (idx == -1, ...)
        else:
            loc = (..., idx == -1)
        # This allows us to match xarray's type promotion rules
        if fill_value is xrdtypes.NA or np.isnan(fill_value):
            new_dtype, fill_value = xrdtypes.maybe_promote(reindexed.dtype)
            reindexed = reindexed.astype(new_dtype, copy=False)
        reindexed[loc] = fill_value
    return reindexed


def offset_labels(labels: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Offset group labels by dimension. This is used when we
    reduce over a subset of the dimensions of by. It assumes that the reductions
    dimensions have been flattened in the last dimension
    Copied from xhistogram &
    https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    """
    ngroups: int = labels.max() + 1  # type: ignore
    offset: np.ndarray = (
        labels + np.arange(np.prod(labels.shape[:-1])).reshape((*labels.shape[:-1], -1)) * ngroups
    )
    # -1 indicates NaNs. preserve these otherwise we aggregate in the wrong groups!
    offset[labels == -1] = -1
    size: int = np.prod(labels.shape[:-1]) * ngroups  # type: ignore
    return offset, ngroups, size


def factorize_(by: Tuple, axis, expected_groups: Tuple = None, isbin: Tuple = None):
    if not isinstance(by, tuple):
        raise ValueError(f"Expected `by` to be a tuple. Received {type(by)} instead")

    if isbin is None:
        isbin = (False,) * len(by)
    if expected_groups is None:
        expected_groups = (None,) * len(by)

    factorized = []
    found_groups = []
    for groupvar, expect, tobin in zip(by, expected_groups, isbin):
        if tobin:
            # when binning we change expected groups to integers marking the interval
            # this makes the reindexing logic simpler.
            if expect is None:
                raise ValueError("Please pass bin edges in expected_groups.")
            # idx = np.digitize(groupvar.ravel(), expect) - 1
            idx = pd.cut(groupvar.ravel(), bins=expect, labels=False)
            # same sentinel value as factorize
            idx[np.isnan(idx)] = -1
            idx = idx.astype(int, copy=False)
            found_groups.append(np.arange(len(expect) - 1))
        else:
            idx, groups = pd.factorize(groupvar.ravel())
            found_groups.append(np.array(groups))
        factorized.append(idx)

    grp_shape = tuple(len(grp) for grp in found_groups)
    ngroups = np.prod(grp_shape)
    if len(by) > 1:
        group_idx = np.ravel_multi_index(factorized, grp_shape).reshape(by[0].shape)
    else:
        group_idx = factorized[0]

    if np.isscalar(axis) and groupvar.ndim > 1:
        # Not reducing along all dimensions of by
        offset_group = True
        group_idx, ngroups, size = offset_labels(group_idx.reshape(by[0].shape))
        group_idx = group_idx.ravel()
    else:
        size = ngroups
        offset_group = False

    # numpy_groupies cannot deal with group_idx = -1
    # so we'll add use (ngroups+1) as the sentinel
    # note we cannot simply remove the NaN locations;
    # that would mess up argmax, argmin
    nan_sentinel = size if offset_group else ngroups
    nanmask = group_idx == -1
    if nanmask.any() and not offset_group:
        size += 1
    group_idx[nanmask] = nan_sentinel

    props = FactorProps(offset_group, nan_sentinel, nanmask)
    return group_idx, found_groups, grp_shape, ngroups, size, props


def chunk_argreduce(
    array_plus_idx: Tuple[np.ndarray, ...],
    by: np.ndarray,
    func: Sequence[str],
    expected_groups: Optional[Union[Sequence, np.ndarray]],
    axis: Union[int, Sequence[int]],
    fill_value: Mapping[Union[str, Callable], Any],
    dtype=None,
    reindex: bool = False,
    isbin: bool = False,
    engine: str = "numpy",
) -> IntermediateDict:
    """
    Per-chunk arg reduction.

    Expects a tuple of (array, index along reduction axis). Inspired by
    dask.array.reductions.argtopk
    """
    array, idx = array_plus_idx

    results = chunk_reduce(
        array,
        by,
        func,
        expected_groups=None,
        axis=axis,
        fill_value=fill_value,
        isbin=isbin,
        dtype=dtype,
        engine=engine,
    )
    if not np.isnan(results["groups"]).all():
        # will not work for empty groups...
        # glorious
        # TODO: npg bug
        results["intermediates"][1] = results["intermediates"][1].astype(int)
        newidx = np.broadcast_to(idx, array.shape)[
            np.unravel_index(results["intermediates"][1], array.shape)
        ]
        results["intermediates"][1] = newidx

    if reindex and expected_groups is not None:
        results["intermediates"][1] = reindex_(
            results["intermediates"][1], results["groups"].squeeze(), expected_groups, fill_value=0
        )

    return results


def chunk_reduce(
    array: np.ndarray,
    by: np.ndarray,
    func: Union[str, Callable, Sequence[str], Sequence[Callable]],
    expected_groups: Union[Sequence, np.ndarray] = None,
    axis: Union[int, Sequence[int]] = None,
    fill_value: Mapping[Union[str, Callable], Any] = None,
    dtype=None,
    reindex: bool = False,
    isbin: bool = False,
    engine: str = "numpy",
    kwargs=None,
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
    array: numpy.ndarray
        Array of values to reduced
    by: numpy.ndarray
        Array to group by.
    func: str or Callable or Sequence[str] or Sequence[Callable]
        Name of reduction or function, passed to numpy_groupies.
        Supports multiple reductions.
    axis: (optional) int or Sequence[int]
        If None, reduce along all dimensions of array.
        Else reduce along specified axes.

    Returns
    -------
    dict
    """

    if dtype is not None:
        assert isinstance(dtype, Sequence)
    if fill_value is not None:
        assert isinstance(fill_value, Sequence)

    if isinstance(func, str) or callable(func):
        func = (func,)  # type: ignore

    func: Union[Sequence[str], Sequence[Callable]]

    nax = len(axis) if isinstance(axis, Sequence) else by.ndim
    final_array_shape = array.shape[:-nax] + (1,) * (nax - 1)
    final_groups_shape = (1,) * (nax - 1)

    if isinstance(axis, Sequence) and len(axis) == 1:
        axis = next(iter(axis))

    if not isinstance(fill_value, Sequence):
        fill_value = (fill_value,)

    if kwargs is None:
        kwargs = ({},) * len(func)

    # when axis is a tuple
    # collapse and move reduction dimensions to the end
    if isinstance(axis, Sequence) and len(axis) < by.ndim:
        by = _collapse_axis(by, len(axis))
        array = _collapse_axis(array, len(axis))
        axis = -1

    if by.ndim == 1:
        # TODO: This assertion doesn't work with dask reducing across all dimensions
        # when by.ndim == array.ndim
        # the intermediates are 1D but axis=range(array.ndim)
        # assert axis in (0, -1, array.ndim - 1, None)
        axis = -1

    # if indices=[2,2,2], npg assumes groups are (0, 1, 2);
    # and will return a result that is bigger than necessary
    # avoid by factorizing again so indices=[2,2,2] is changed to
    # indices=[0,0,0]. This is necessary when combining block results
    # factorize can handle strings etc unlike digitize
    group_idx, groups, _, ngroups, size, props = factorize_(
        (by,), axis, expected_groups=(expected_groups,), isbin=(isbin,)
    )
    groups = groups[0]

    # always reshape to 1D along group dimensions
    newshape = array.shape[: array.ndim - by.ndim] + (np.prod(array.shape[-by.ndim :]),)
    array = array.reshape(newshape)

    assert group_idx.ndim == 1
    empty = np.all(props.nanmask) or np.prod(by.shape) == 0

    results: IntermediateDict = {"groups": [], "intermediates": []}
    if reindex and expected_groups is not None:
        results["groups"] = np.array(expected_groups)
    else:
        if empty:
            results["groups"] = np.array([np.nan])
        else:
            if (groups[:-1] <= groups[1:]).all():
                sortidx = slice(None)
            else:
                sortidx = groups.argsort()
            results["groups"] = groups[sortidx]

    # npg's argmax ensures that index of first "max" is returned assuming there
    # are many elements equal to the "max". Sorting messes this up totally.
    # so we skip this for argreductions
    if engine == "flox":
        # is_arg_reduction = any("arg" in f for f in func if isinstance(f, str))
        # if not is_arg_reduction:
        group_idx, array = _prepare_for_flox(group_idx, array)

    final_array_shape += results["groups"].shape
    final_groups_shape += results["groups"].shape

    for reduction, fv, kw, dt in zip(func, fill_value, kwargs, dtype):
        if empty:
            result = np.full(shape=final_array_shape, fill_value=fv)
        else:
            if callable(reduction):
                # passing a custom reduction for npg to apply per-group is really slow!
                # So this `reduction` has to do the groupby-aggregation
                result = reduction(
                    group_idx,
                    array,
                    size=size,
                    # important when reducing with "offset" groups
                    fill_value=fv,
                    dtype=dt,
                    **kw,
                )
            else:
                result = generic_aggregate(
                    group_idx,
                    array,
                    axis=-1,
                    engine=engine,
                    func=reduction,
                    size=size,
                    # important when reducing with "offset" groups
                    fill_value=fv,
                    dtype=dt,
                    **kw,
                ).astype(dt, copy=False)
            if np.any(props.nanmask):
                # remove NaN group label which should be last
                result = result[..., :-1]
            if props.offset_group:
                result = result.reshape(*final_array_shape[:-1], ngroups)
            if reindex and expected_groups is not None:
                result = reindex_(result, groups, expected_groups, fill_value=fv)
            else:
                result = result[..., sortidx]
            result = result.reshape(final_array_shape)
        results["intermediates"].append(result)
    if final_groups_shape:
        # This happens when to_group is broadcasted, and we reduce along the broadcast
        # dimension
        results["groups"] = np.broadcast_to(results["groups"], final_groups_shape)
    return results


def _squeeze_results(results: IntermediateDict, axis: Sequence) -> IntermediateDict:
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


def _split_groups(array, j, slicer):
    """Slices out chunks when split_out > 1"""
    results = {"groups": array["groups"][..., slicer]}
    results["intermediates"] = [v[..., slicer] for v in array["intermediates"]]
    return results


def _finalize_results(
    results: IntermediateDict,
    agg: Aggregation,
    axis: Sequence[int],
    expected_groups: Union[Sequence, np.ndarray, None],
    fill_value: Any,
    min_count: Optional[int] = None,
    finalize_kwargs: Optional[Mapping] = None,
):
    """Finalize results by
    1. Squeezing out dummy dimensions
    2. Calling agg.finalize with intermediate results
    3. Mask using counts and fill with user-provided fill_value.
    4. reindex to expected_groups
    """
    squeezed = _squeeze_results(results, axis)

    # finalize step
    result: Dict[str, Union["DaskArray", np.ndarray]] = {}
    if agg.finalize is None:
        if min_count is not None:
            counts = squeezed["intermediates"][-1]
            squeezed["intermediates"] = squeezed["intermediates"][:-1]
        result[agg.name] = squeezed["intermediates"][0]
        if min_count is not None and np.any(counts < min_count):
            if fill_value is None:
                raise ValueError("Filling is required but fill_value is None.")
            # This allows us to match xarray's type promotion rules
            if fill_value is xrdtypes.NA:
                new_dtype, fill_value = xrdtypes.maybe_promote(result[agg.name].dtype)
                result[agg.name] = result[agg.name].astype(new_dtype)
            result[agg.name] = np.where(counts >= min_count, result[agg.name], fill_value)
    else:
        if fill_value is not None:
            counts = squeezed["intermediates"][-1]
            squeezed["intermediates"] = squeezed["intermediates"][:-1]
        # This is needed for the dask pathway.
        # Because we use intermediate fill_value since a group could be
        # absent in one block, but present in another block
        if min_count is None:
            min_count = 1
        if finalize_kwargs is None:
            finalize_kwargs = {}
        result[agg.name] = agg.finalize(*squeezed["intermediates"], **finalize_kwargs)
        if fill_value is not None:
            count_mask = counts < min_count
            if count_mask.any():
                # For one this check prevents promoting bool to dtype(fill_value) unless
                # necessary
                result[agg.name] = np.where(count_mask, fill_value, result[agg.name])

    # Final reindexing has to be here to be lazy
    if expected_groups is not None:
        result[agg.name] = reindex_(
            result[agg.name], squeezed["groups"], expected_groups, fill_value=fill_value
        )
        result["groups"] = expected_groups

    return result


def _npg_aggregate(
    x_chunk,
    agg: Aggregation,
    expected_groups: Union[Sequence, np.ndarray, None],
    axis: Sequence,
    keepdims,
    neg_axis: Sequence,
    fill_value: Any = None,
    min_count: Optional[int] = None,
    engine: str = "numpy",
    finalize_kwargs: Optional[Mapping] = None,
) -> FinalResultsDict:
    """Final aggregation step of tree reduction"""
    results = _npg_combine(x_chunk, agg, axis, keepdims, neg_axis, engine)
    return _finalize_results(
        results, agg, axis, expected_groups, fill_value, min_count, finalize_kwargs
    )


def _conc2(x_chunk, key1, key2=None, axis=None) -> np.ndarray:
    """copied from dask.array.reductions.mean_combine"""
    from dask.array.core import _concatenate2
    from dask.utils import deepmap

    if key2 is not None:
        mapped = deepmap(lambda x: x[key1][key2], x_chunk)
    else:
        mapped = deepmap(lambda x: x[key1], x_chunk)
    return _concatenate2(mapped, axes=axis)

    # This doesn't seem to improve things at all; and some tests fail...
    # from dask.array.core import concatenate3
    # for _ in range(mapped[0].ndim-1):
    #    mapped = [mapped]
    # return concatenate3(mapped)


def _npg_combine(
    x_chunk,
    agg: Aggregation,
    axis: Sequence,
    keepdims: bool,
    neg_axis: Sequence,
    engine: str,
) -> IntermediateDict:
    """Combine intermediates step of tree reduction."""
    from dask.base import flatten
    from dask.utils import deepmap

    if not isinstance(x_chunk, list):
        x_chunk = [x_chunk]

    if len(axis) != 1:
        # when there's only a single axis of reduction, we can just concatenate later,
        # reindexing is unnecessary
        # I bet we can minimize the amount of reindexing for mD reductions too, but it's complicated
        unique_groups = np.unique(
            tuple(flatten(deepmap(lambda x: list(np.atleast_1d(x["groups"].squeeze())), x_chunk)))
        )

        def reindex_intermediates(x):
            new_shape = x["groups"].shape[:-1] + (len(unique_groups),)
            newx = {"groups": np.broadcast_to(unique_groups, new_shape)}
            newx["intermediates"] = tuple(
                reindex_(v, from_=x["groups"].squeeze(), to=unique_groups, fill_value=f)
                for v, f in zip(x["intermediates"], agg.fill_value["intermediate"])
            )
            return newx

        x_chunk = deepmap(reindex_intermediates, x_chunk)

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
        results = chunk_argreduce(
            array_idx,
            groups,
            func=agg.combine[slicer],  # count gets treated specially next
            axis=axis,
            expected_groups=None,
            fill_value=agg.fill_value["intermediate"][slicer],
            dtype=agg.dtype["intermediate"][slicer],
            engine=engine,
        )

        if agg.chunk[-1] == "nanlen":
            counts = _conc2(x_chunk, key1="intermediates", key2=2, axis=axis)
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
                )["intermediates"][0]
            )

    elif agg.reduction_type == "reduce":
        # Here we reduce the intermediates individually
        results = {"groups": None, "intermediates": []}
        for idx, (combine, fv, dtype) in enumerate(
            zip(agg.combine, agg.fill_value["intermediate"], agg.dtype["intermediate"])
        ):
            array = _conc2(x_chunk, key1="intermediates", key2=idx, axis=axis)
            if array.shape[-1] == 0:
                # all empty when combined
                results["intermediates"].append(
                    np.empty(shape=(1,) * (len(axis) - 1) + (0,), dtype=agg.dtype)
                )
                results["groups"] = np.empty(
                    shape=(1,) * (len(neg_axis) - 1) + (0,), dtype=groups.dtype
                )
            else:
                _results = chunk_reduce(
                    array,
                    groups,
                    func=combine,
                    axis=axis,
                    expected_groups=None,
                    fill_value=(fv,),
                    dtype=(dtype,),
                    engine=engine,
                )
                results["intermediates"].append(*_results["intermediates"])
                results["groups"] = _results["groups"]
    return results


def split_blocks(applied, split_out, expected_groups, split_name):
    import dask.array
    from dask.array.core import normalize_chunks
    from dask.highlevelgraph import HighLevelGraph

    chunk_tuples = tuple(itertools.product(*tuple(range(n) for n in applied.numblocks)))
    ngroups = len(expected_groups)
    group_chunks = normalize_chunks(np.ceil(ngroups / split_out), (ngroups,))[0]
    idx = tuple(np.cumsum((0,) + group_chunks))

    # split each block into `split_out` chunks
    dsk = {}
    for i in chunk_tuples:
        for j in range(split_out):
            dsk[(split_name, *i, j)] = (
                _split_groups,
                (applied.name, *i),
                j,
                slice(idx[j], idx[j + 1]),
            )

    # now construct an array that can be passed to _tree_reduce
    intergraph = HighLevelGraph.from_collections(split_name, dsk, dependencies=(applied,))
    intermediate = dask.array.Array(
        intergraph,
        name=split_name,
        chunks=applied.chunks + ((1,) * split_out,),
        meta=applied._meta,
    )
    return intermediate, group_chunks


def groupby_agg(
    array: "DaskArray",
    by: Union["DaskArray", np.ndarray],
    agg: Aggregation,
    expected_groups: Optional[Union[Sequence, np.ndarray]],
    axis: Sequence = None,
    split_out: int = 1,
    fill_value: Any = None,
    method: str = "map-reduce",
    min_count: Optional[int] = None,
    isbin: bool = False,
    reindex: bool = False,
    engine: str = "numpy",
    finalize_kwargs: Optional[Mapping] = None,
) -> Tuple["DaskArray", Union[np.ndarray, "DaskArray"]]:

    import dask.array
    from dask.highlevelgraph import HighLevelGraph

    # I think _tree_reduce expects this
    assert isinstance(axis, Sequence)
    assert all(ax >= 0 for ax in axis)

    # these are negative axis indices useful for concatenating the intermediates
    neg_axis = range(-len(axis), 0)

    inds = tuple(range(array.ndim))
    name = f"groupby_{agg.name}"
    token = dask.base.tokenize(array, by, agg, expected_groups, axis, split_out)

    # This is necessary for argreductions.
    # We need to rechunk before zipping up with the index
    # let's always do it anyway
    # but first save by if blockwise is True.
    if method == "blockwise":
        by_maybe_numpy = by
    _, (array, by) = dask.array.unify_chunks(array, inds, by, inds[-by.ndim :])

    # preprocess the array
    if agg.preprocess:
        array = agg.preprocess(array, axis=axis)

    # apply reduction on chunk
    applied = dask.array.blockwise(
        partial(
            _get_chunk_reduction(agg.reduction_type),
            func=agg.chunk,  # type: ignore
            axis=axis,
            # with the current implementation we want reindexing at the blockwise step
            # only reindex to groups present at combine stage
            expected_groups=expected_groups if reindex or split_out > 1 or isbin else None,
            fill_value=agg.fill_value["intermediate"],
            dtype=agg.dtype["intermediate"],
            isbin=isbin,
            reindex=reindex or (split_out > 1),
            engine=engine,
        ),
        inds,
        array,
        inds,
        by,
        inds[-by.ndim :],
        concatenate=False,
        dtype=array.dtype,  # this is purely for show
        meta=array._meta,
        align_arrays=False,
        token=f"{name}-chunk-{token}",
    )

    if split_out > 1:
        if expected_groups is None:
            # This could be implemented using the "hash_split" strategy
            # from dask.dataframe
            raise NotImplementedError

        intermediate, group_chunks = split_blocks(
            applied, split_out, expected_groups, split_name=f"{name}-split-{token}"
        )
        expected_agg = None
    else:
        intermediate = applied
        # from this point on, we just work with bin indexes when binning
        if isbin:
            expected_groups = np.arange(len(expected_groups) - 1)
        group_chunks = (len(expected_groups),) if expected_groups is not None else (np.nan,)
        expected_agg = expected_groups

    agg_kwargs = dict(
        neg_axis=neg_axis,
        fill_value=fill_value,
        min_count=min_count,
        engine=engine,
        finalize_kwargs=finalize_kwargs,
    )

    if method == "map-reduce":
        # reduced is really a dict mapping reduction name to array
        # and "groups" to an array of group labels
        # Note: it does not make sense to interpret axis relative to
        # shape of intermediate results after the blockwise call
        reduced = dask.array.reductions._tree_reduce(
            intermediate,
            aggregate=partial(
                _npg_aggregate,
                agg=agg,
                expected_groups=expected_agg,
                **agg_kwargs,
            ),
            combine=partial(_npg_combine, agg=agg, neg_axis=neg_axis, engine=engine),
            name=f"{name}-reduce",
            dtype=array.dtype,
            axis=axis,
            keepdims=True,
            concatenate=False,
        )
        output_chunks = reduced.chunks[: -(len(axis) + int(split_out > 1))] + (group_chunks,)
    elif method == "blockwise":
        # Blockwise apply the aggregation step so that one input chunk → one output chunk
        # TODO: We could combine this with the chunk reduction and do everything in one task.
        #       This would also optimize the single block along reduced-axis case.
        if expected_groups is None or split_out > 1 or not isinstance(by_maybe_numpy, np.ndarray):
            raise NotImplementedError

        reduced = dask.array.blockwise(
            partial(
                _npg_aggregate,
                agg=agg,
                expected_groups=None,
                **agg_kwargs,
                axis=axis,
                keepdims=True,
            ),
            inds,
            intermediate,
            inds,
            concatenate=False,
            dtype=array.dtype,
            meta=array._meta,
            align_arrays=False,
            name=f"{name}-blockwise-{token}",
        )

        # find number of groups in each chunk, this is needed for output chunks
        # along the reduced axis
        from dask.array.core import slices_from_chunks

        slices = slices_from_chunks(tuple(array.chunks[ax] for ax in axis))
        if expected_groups is None:
            groups_in_block = tuple(np.unique(by_maybe_numpy[slc]) for slc in slices)
        else:
            # For cohorts, we could be indexing a block with groups that
            # are not in the cohort (usually for nD `by`)
            # Only keep the expected groups.
            groups_in_block = tuple(
                np.intersect1d(by_maybe_numpy[slc], expected_groups) for slc in slices
            )
        ngroups_per_block = tuple(len(groups) for groups in groups_in_block)
        output_chunks = reduced.chunks[: -(len(axis))] + (ngroups_per_block,)
    else:
        raise ValueError(f"Unknown method={method}.")

    def _getitem(d, key1, key2):
        return d[key1][key2]

    # extract results from the dict
    result: Dict = {}
    layer: Dict[Tuple, Tuple] = {}
    ochunks = tuple(range(len(chunks_v)) for chunks_v in output_chunks)
    if expected_groups is None:
        groups_name = f"groups-{name}-{token}"
        # we've used keepdims=True, so _tree_reduce preserves some dummy dimensions
        first_block = len(ochunks) * (0,)
        layer[(groups_name, *first_block)] = (
            operator.getitem,
            (reduced.name, *first_block),
            "groups",
        )
        groups = (
            dask.array.Array(
                HighLevelGraph.from_collections(groups_name, layer, dependencies=[reduced]),
                groups_name,
                chunks=(group_chunks,),
                dtype=by.dtype,
            ),
        )
    else:
        if method == "map-reduce":
            groups = (expected_groups,)
        else:
            groups = (np.concatenate(groups_in_block),)

    layer: Dict[Tuple, Tuple] = {}  # type: ignore
    agg_name = f"{name}-{token}"
    for ochunk in itertools.product(*ochunks):
        if method == "blockwise":
            if len(axis) == 1:
                inchunk = ochunk
            else:
                nblocks = tuple(len(array.chunks[ax]) for ax in axis)
                inchunk = ochunk[:-1] + np.unravel_index(ochunk[-1], nblocks)
        else:
            inchunk = ochunk[:-1] + (0,) * len(axis) + (ochunk[-1],) * int(split_out > 1)
        layer[(agg_name, *ochunk)] = (
            operator.getitem,
            (reduced.name, *inchunk),
            agg.name,
        )
    result = dask.array.Array(
        HighLevelGraph.from_collections(agg_name, layer, dependencies=[reduced]),
        agg_name,
        chunks=output_chunks,
        dtype=agg.dtype[agg.name],
    )

    return (result, *groups)


def groupby_reduce(
    array: Union[np.ndarray, "DaskArray"],
    by: Union[np.ndarray, "DaskArray"],
    func: Union[str, Aggregation],
    *,
    expected_groups: Union[Sequence, np.ndarray] = None,
    sort: bool = True,
    isbin: bool = False,
    axis=None,
    fill_value=None,
    min_count: Optional[int] = None,
    split_out: int = 1,
    method: str = "map-reduce",
    engine: str = "flox",
    finalize_kwargs: Optional[Mapping] = None,
) -> Tuple["DaskArray", Union[np.ndarray, "DaskArray"]]:
    """
    GroupBy reductions using tree reductions for dask.array

    Parameters
    ----------
    array : ndarray or DaskArray
        Array to be reduced, possibly nD
    by : ndarray or DaskArray
        Array of labels to group over. Must be aligned with ``array`` so that
            ``array.shape[-by.ndim :] == by.shape``
    func : str or Aggregation
        Single function name or an Aggregation instance
    expected_groups : (optional) Sequence
        Expected unique labels.
    isbin : bool, optional
        Are ``expected_groups`` bin edges?
    sort : (optional), bool
        Whether groups should be returned in sorted order. Only applies for dask
        reductions when ``method`` is not `"map-reduce"`. For ``"map-reduce", the groups
        are always sorted.
    axis : (optional) None or int or Sequence[int]
        If None, reduce across all dimensions of by
        Else, reduce across corresponding axes of array
        Negative integers are normalized using array.ndim
    fill_value : Any
        Value to assign when a label in ``expected_groups`` is not present.
    min_count : int, default: None
        The required number of valid values to perform the operation. If
        fewer than min_count non-NA values are present the result will be
        NA. Only used if skipna is set to True or defaults to True for the
        array's dtype.
    split_out : int, optional
        Number of chunks along group axis in output (last axis)
    method : {"map-reduce", "blockwise", "cohorts", "split-reduce"}, optional
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
          * ``"split-reduce"``:
            Break out each group into its own array and then ``"map-reduce"``.
            This is implemented by having each group be its own cohort,
            and is identical to xarray's default strategy.

    engine : {"flox", numpy", "numba"}, optional
        Underlying algorithm to compute the groupby reduction on non-dask arrays
        and on each dask chunk at compute-time.
          * ``"flox"``:
            Use an internal implementation where the data is sorted so that
            all members of a group occur sequentially, and then numpy.ufunc.reduceat
            is to used for the reduction. This will fall back to ``numpy_groupies.aggregate_numpy``
            for a reduction that is not yet implemented.
          * ``"numpy"``:
            Use the vectorized implementations in ``numpy_groupies.aggregate_numpy``.
          * ``"numba"``:
            Use the implementations in ``numpy_groupies.aggregate_numba``.

    finalize_kwargs : dict, optional
        Kwargs passed to finalize the reduction such as ``ddof`` for var, std.

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

    if engine == "flox" and isinstance(func, str) and "arg" in func:
        raise NotImplementedError(
            "argreductions not supported for engine='flox' yet."
            "Try engine='numpy' or engine='numba' instead."
        )

    if not is_duck_array(by):
        by = np.asarray(by)
    if not is_duck_array(array):
        array = np.asarray(array)
    if array.shape[-by.ndim :] != by.shape:
        raise ValueError(
            "array and by must be aligned i.e. array.shape[-by.ndim :] == by.shape. "
            f"Received array of shape {array.shape} and by of shape {by.shape}"
        )

    if min_count is not None and min_count > 1:
        if func not in ["nansum", "nanprod"]:
            raise ValueError("min_count can be > 1 only for nansum, nanprod.")

    if axis is None:
        axis = tuple(array.ndim + np.arange(-by.ndim, 0))
    else:
        axis = np.core.numeric.normalize_axis_tuple(axis, array.ndim)  # type: ignore

    if method in ["blockwise", "cohorts", "split-reduce"] and len(axis) != by.ndim:
        raise NotImplementedError(
            "Must reduce along all dimensions of `by` when method != 'map-reduce'."
        )

    if expected_groups is None and isinstance(by, np.ndarray):
        flatby = by.ravel()
        expected_groups = np.unique(flatby[~isnull(flatby)])

    # TODO: make sure expected_groups is unique
    if len(axis) == 1 and by.ndim > 1 and expected_groups is None:
        # When we reduce along all axes, it guarantees that we will see all
        # groups in the final combine stage, so everything works.
        # This is not necessarily true when reducing along a subset of axes
        # (of by)
        # TODO: depends on chunking of by?
        # we could relax this if there is only one chunk along all
        # by dim != axis?
        raise NotImplementedError(
            "Please provide ``expected_groups`` when not reducing along all axes."
        )

    if isinstance(axis, Sequence) and len(axis) < by.ndim:
        by = _move_reduce_dims_to_end(by, -array.ndim + np.array(axis) + by.ndim)
        array = _move_reduce_dims_to_end(array, axis)
        axis = tuple(array.ndim + np.arange(-len(axis), 0))

    if not isinstance(func, Aggregation):
        try:
            # TODO: need better interface
            # we set dtype, fillvalue on reduction later. so deepcopy now
            reduction = copy.deepcopy(getattr(aggregations, func))
        except AttributeError:
            raise NotImplementedError(f"Reduction {func!r} not implemented yet")
    else:
        # TODO: test that func is a valid Aggregation
        reduction = copy.deepcopy(func)
        func = reduction.name

    reduction.dtype[func] = _normalize_dtype(reduction.dtype[func], array.dtype, fill_value)
    reduction.dtype["intermediate"] = [
        _normalize_dtype(dtype, array.dtype) for dtype in reduction.dtype["intermediate"]
    ]

    # Replace sentinel fill values according to dtype
    reduction.fill_value["intermediate"] = tuple(
        _get_fill_value(dt, fv)
        for dt, fv in zip(reduction.dtype["intermediate"], reduction.fill_value["intermediate"])
    )
    reduction.fill_value[func] = _get_fill_value(reduction.dtype[func], reduction.fill_value[func])

    if min_count is not None:
        # Let this pass so that xarray can keep return np.nan for bins with
        # no observations. The restriction of min_count to nansum, nanprod
        # seems to be an Xarray limitation so there's no reason we need to copy it.
        # nansum, nanprod have fill_value=0, 1
        # overwrite than when min_count is set
        if func in ["nansum", "nanprod"] and fill_value is None:
            fill_value = np.nan

    if not is_duck_dask_array(array) and not is_duck_dask_array(by):
        # for pure numpy grouping, we just use npg directly and avoid "finalizing"
        # (agg.finalize = None). We still need to do the reindexing step in finalize
        # so that everything matches the dask version.
        reduction.finalize = None
        # xarray's count is npg's nanlen
        func = (reduction.numpy, "nanlen")
        if finalize_kwargs is None:
            finalize_kwargs = {}
        if isinstance(finalize_kwargs, Mapping):
            finalize_kwargs = (finalize_kwargs,)
        finalize_kwargs = finalize_kwargs + ({},) + ({},)

        results = chunk_reduce(
            array,
            by,
            func=func,
            axis=axis,
            expected_groups=expected_groups if isbin else None,
            # This fill_value should only apply to groups that only contain NaN observations
            # BUT there is funkiness when axis is a subset of all possible values
            # (see below)
            fill_value=(reduction.fill_value[reduction.name], 0),
            dtype=(reduction.dtype[reduction.name], np.intp),
            isbin=isbin,
            kwargs=finalize_kwargs,
            engine=engine,
        )  # type: ignore

        if reduction.name in ["argmin", "argmax", "nanargmax", "nanargmin"]:
            if array.ndim > 1:
                # default fill_value is -1; we can't unravel that;
                # so replace -1 with 0; unravel; then replace 0 with -1
                # UGH!
                idx = results["intermediates"][0]
                mask = idx == -1
                idx[mask] = 0
                # Fix npg bug where argmax with nD array, 1D group_idx, axis=-1
                # will return wrong indices
                idx = np.unravel_index(idx, array.shape)[-1]
                idx[mask] = -1
                results["intermediates"][0] = idx
        elif reduction.name in ["nanvar", "nanstd"]:
            # Fix npg bug where all-NaN rows are 0 instead of NaN
            value, counts = results["intermediates"]
            mask = counts <= 0
            value[mask] = np.nan
            results["intermediates"][0] = value

        if isbin:
            expected_groups = np.arange(len(expected_groups) - 1)

        # When axis is a subset of possible values; then npg will
        # apply it to groups that don't exist along a particular axis (for e.g.)
        # since these count as a group that is absent. thoo!
        # TODO: the "count" bit is a hack to make tests pass.
        if len(axis) < by.ndim and min_count is None and reduction.name != "count":
            min_count = 1

        if fill_value is None:
            if reduction.name in ["any", "all"]:
                fill_value = False
            elif "arg" not in reduction.name:
                fill_value = xrdtypes.NA

        result = _finalize_results(
            results,
            reduction,
            axis,
            expected_groups,
            # This fill_value applies to members of expected_groups not seen in groups
            # or when the min_count threshold is not satisfied
            # Use xarray's dtypes.NA to match type promotion rules
            fill_value=fill_value,
            min_count=min_count,
        )
        groups = (result["groups"],)
        result = result[reduction.name]
    else:
        if func in ["first", "last"]:
            raise NotImplementedError("first, last not implemented for dask arrays")

        # we need to explicitly track counts so that we can mask at the end
        if fill_value is not None or min_count is not None:
            reduction.chunk += ("nanlen",)
            reduction.combine += ("sum",)
            reduction.fill_value["intermediate"] += (0,)
            reduction.dtype["intermediate"] += (np.intp,)

        partial_agg = partial(
            groupby_agg,
            agg=reduction,
            axis=axis,
            split_out=split_out,
            fill_value=fill_value,
            min_count=min_count,
            isbin=isbin,
            engine=engine,
            finalize_kwargs=finalize_kwargs,
        )

        if method in ["split-reduce", "cohorts"]:
            cohorts = find_group_cohorts(
                by, [array.chunks[ax] for ax in axis], merge=True, method=method
            )

            results = []
            groups_ = []
            for cohort in cohorts:
                cohort = sorted(cohort)
                # equivalent of xarray.DataArray.where(mask, drop=True)
                mask = np.isin(by, cohort)
                indexer = [np.unique(v) for v in np.nonzero(mask)]
                array_subset = array
                for ax, idxr in zip(range(-by.ndim, 0), indexer):
                    array_subset = np.take(array_subset, idxr, axis=ax)
                numblocks = np.prod([len(array_subset.chunks[ax]) for ax in axis])

                # get final result for these groups
                r, *g = partial_agg(
                    array_subset,
                    by[np.ix_(*indexer)],
                    expected_groups=cohort,
                    # reindex to expected_groups at the blockwise step.
                    # this approach avoids replacing non-cohort members with
                    # np.nan or some other sentinel value, and preserves dtypes
                    reindex=True,
                    # if only a single block along axis, we can just work blockwise
                    # inspired by https://github.com/dask/dask/issues/8361
                    method="blockwise" if numblocks == 1 and len(axis) == by.ndim else "map-reduce",
                )
                results.append(r)
                groups_.append(cohort)

            # concatenate results together,
            # sort to make sure we match expected output
            groups = (np.hstack(groups_),)
            result = np.concatenate(results, axis=-1)
        else:
            if method == "blockwise":
                if by.ndim == 1:
                    array = rechunk_for_blockwise(array, axis=-1, labels=by)

            # TODO: test with mixed array kinds (numpy array + dask by)
            result, *groups = partial_agg(
                array,
                by,
                expected_groups=expected_groups,
                method=method,
            )
        if sort and method != "map-reduce":
            assert len(groups) == 1
            sorted_idx = np.argsort(groups[0])
            result = result[..., sorted_idx]
            groups = (groups[0][sorted_idx],)

    return (result, *groups)
