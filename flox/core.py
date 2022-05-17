from __future__ import annotations

import copy
import itertools
import operator
from collections import namedtuple
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Sequence, Union

import numpy as np
import numpy_groupies as npg
import pandas as pd
import toolz as tlz

from . import xrdtypes
from .aggregations import (
    Aggregation,
    _atleast_1d,
    _initialize_aggregation,
    generic_aggregate,
)
from .cache import memoize
from .xrutils import is_duck_array, is_duck_dask_array, isnull

if TYPE_CHECKING:
    import dask.array.Array as DaskArray


IntermediateDict = Dict[Union[str, Callable], Any]
FinalResultsDict = Dict[str, Union["DaskArray", np.ndarray]]
FactorProps = namedtuple("FactorProps", "offset_group nan_sentinel nanmask")

# This dummy axis is inserted using np.expand_dims
# and then reduced over during the combine stage by
# _simple_combine.
DUMMY_AXIS = -2


def _is_arg_reduction(func: str | Aggregation) -> bool:
    if isinstance(func, str) and func in ["argmin", "argmax", "nanargmax", "nanargmin"]:
        return True
    if isinstance(func, Aggregation) and func.reduction_type == "argreduce":
        return True
    return False


def _prepare_for_flox(group_idx, array):
    """
    Sort the input array once to save time.
    """
    assert array.shape[-1] == group_idx.shape[0]
    issorted = (group_idx[:-1] <= group_idx[1:]).all()
    if issorted:
        ordered_array = array
    else:
        perm = group_idx.argsort(kind="stable")
        group_idx = group_idx[..., perm]
        ordered_array = array[..., perm]
    return group_idx, ordered_array


def _get_expected_groups(by, sort, *, raise_if_dask=True) -> pd.Index | None:
    if is_duck_dask_array(by):
        if raise_if_dask:
            raise ValueError("Please provide expected_groups if not grouping by a numpy array.")
        return None
    flatby = by.ravel()
    expected = pd.unique(flatby[~isnull(flatby)])
    return _convert_expected_groups_to_index((expected,), isbin=(False,), sort=sort)[0]


def _get_chunk_reduction(reduction_type: str) -> Callable:
    if reduction_type == "reduce":
        return chunk_reduce
    elif reduction_type == "argreduce":
        return chunk_argreduce
    else:
        raise ValueError(f"Unknown reduction type: {reduction_type}")


def is_nanlen(reduction: str | Callable) -> bool:
    return isinstance(reduction, str) and reduction == "nanlen"


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


@memoize
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
    labels : np.ndarray
        mD Array of group labels
    chunks : tuple
        nD array that is being reduced
    merge : bool, optional
        Attempt to merge cohorts when one cohort's chunks are a subset
        of another cohort's chunks.
    method : ["split-reduce", "cohorts"], optional
        Which method are we using?

    Returns
    -------
    cohorts: dict_values
        Iterable of cohorts
    """
    import dask

    # To do this, we must have values in memory so casting to numpy should be safe
    labels = np.asarray(labels)

    if method == "split-reduce":
        return _get_expected_groups(labels, sort=False).values.reshape(-1, 1).tolist()

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

    # We always drop NaN; np.unique also considers every NaN to be different so
    # it's really important we get rid of them.
    raveled = labels.ravel()
    unique_labels = np.unique(raveled[~isnull(raveled)])
    # these are chunks where a label is present
    label_chunks = {lab: tuple(np.unique(which_chunk[raveled == lab])) for lab in unique_labels}
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


def rechunk_for_cohorts(
    array, axis, labels, force_new_chunk_at, chunksize=None, ignore_old_chunks=False, debug=False
):
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
    if debug:
        labels_at_breaks = labels[oldbreaks[:-1]]
        print(labels_at_breaks[:40])

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
    assert sum(newchunks) == len(labels)

    if newchunks == array.chunks[axis]:
        return array
    else:
        return array.rechunk({axis: newchunks})


def rechunk_for_blockwise(array, axis, labels):
    """
    Rechunks array so that group boundaries line up with chunk boundaries, allowing
    embarassingly parallel group reductions.

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
    labels = factorize_((labels,), axis=None)[0]
    chunks = array.chunks[axis]
    newchunks = _get_optimal_chunks_for_groups(chunks, labels)
    if newchunks == chunks:
        return array
    else:
        return array.rechunk({axis: newchunks})


def reindex_(
    array: np.ndarray, from_, to, fill_value=None, axis: int = -1, promote: bool = False
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
    indexer[axis] = idx  # type: ignore
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
        labels + np.arange(np.prod(labels.shape[:-1])).reshape((*labels.shape[:-1], -1)) * ngroups
    )
    # -1 indicates NaNs. preserve these otherwise we aggregate in the wrong groups!
    offset[labels == -1] = -1
    size: int = np.prod(labels.shape[:-1]) * ngroups  # type: ignore
    return offset, size


def factorize_(
    by: tuple,
    axis,
    expected_groups: tuple[pd.Index, ...] = None,
    reindex=False,
    sort=True,
    fastpath=False,
):
    """
    Returns an array of integer  codes  for groups (and associated data)
    by wrapping pd.cut and pd.factorize (depending on isbin).
    This method handles reindex and sort so that we don't spend time reindexing / sorting
    a possibly large results array. Instead we set up the appropriate integer codes (group_idx)
    so that the results come out in the appropriate order.
    """
    if not isinstance(by, tuple):
        raise ValueError(f"Expected `by` to be a tuple. Received {type(by)} instead")

    if expected_groups is None:
        expected_groups = (None,) * len(by)

    factorized = []
    found_groups = []
    for groupvar, expect in zip(by, expected_groups):
        if isinstance(expect, pd.IntervalIndex):
            # when binning we change expected groups to integers marking the interval
            # this makes the reindexing logic simpler.
            if expect is None:
                raise ValueError("Please pass bin edges in expected_groups.")
            # TODO: fix for binning
            found_groups.append(expect)
            # pd.cut with bins = IntervalIndex[datetime64] doesn't work...
            if groupvar.dtype.kind == "M":
                expect = np.concatenate([expect.left.to_numpy(), [expect.right[-1].to_numpy()]])
            idx = pd.cut(groupvar.ravel(), bins=expect).codes.copy()
        else:
            if expect is not None and reindex:
                groups = expect
                if not sort:
                    sorter = np.argsort(expect)
                else:
                    sorter = None
                idx = np.searchsorted(expect, groupvar.ravel(), sorter=sorter)
                mask = isnull(groupvar.ravel()) | (idx == len(expect))
                # TODO: optimize?
                idx[mask] = -1
                if not sort:
                    idx = sorter[idx]
                    idx[mask] = -1
            else:
                idx, groups = pd.factorize(groupvar.ravel(), sort=sort)

            found_groups.append(np.array(groups))
        factorized.append(idx)

    grp_shape = tuple(len(grp) for grp in found_groups)
    ngroups = np.prod(grp_shape)
    if len(by) > 1:
        group_idx = np.ravel_multi_index(factorized, grp_shape, mode="wrap").reshape(by[0].shape)
        nan_by_mask = reduce(np.logical_or, [isnull(b) for b in by])
        group_idx[nan_by_mask] = -1
    else:
        group_idx = factorized[0]

    if fastpath:
        return group_idx, found_groups, grp_shape

    if np.isscalar(axis) and groupvar.ndim > 1:
        # Not reducing along all dimensions of by
        # this is OK because for 3D by and axis=(1,2),
        # we collapse to a 2D by and axis=-1
        offset_group = True
        group_idx, size = offset_labels(group_idx.reshape(by[0].shape), ngroups)
        group_idx = group_idx.ravel()
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
    return group_idx, found_groups, grp_shape, ngroups, size, props


def chunk_argreduce(
    array_plus_idx: tuple[np.ndarray, ...],
    by: np.ndarray,
    func: Sequence[str],
    expected_groups: pd.Index | None,
    axis: int | Sequence[int],
    fill_value: Mapping[str | Callable, Any],
    dtype=None,
    reindex: bool = False,
    engine: str = "numpy",
    sort: bool = True,
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
    func: str | Callable | Sequence[str] | Sequence[Callable],
    expected_groups: pd.Index | None,
    axis: int | Sequence[int] = None,
    fill_value: Mapping[str | Callable, Any] = None,
    dtype=None,
    reindex: bool = False,
    engine: str = "numpy",
    kwargs=None,
    sort=True,
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

    if dtype is not None:
        assert isinstance(dtype, Sequence)
    if fill_value is not None:
        assert isinstance(fill_value, Sequence)

    if isinstance(func, str) or callable(func):
        func = (func,)  # type: ignore

    func: Sequence[str] | Sequence[Callable]

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

    # if indices=[2,2,2], npg assumes groups are (0, 1, 2);
    # and will return a result that is bigger than necessary
    # avoid by factorizing again so indices=[2,2,2] is changed to
    # indices=[0,0,0]. This is necessary when combining block results
    # factorize can handle strings etc unlike digitize
    group_idx, groups, found_groups_shape, ngroups, size, props = factorize_(
        (by,), axis, expected_groups=(expected_groups,), reindex=reindex, sort=sort
    )
    groups = groups[0]

    # always reshape to 1D along group dimensions
    newshape = array.shape[: array.ndim - by.ndim] + (np.prod(array.shape[-by.ndim :]),)
    array = array.reshape(newshape)

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
    previous_reduction = None
    for param in (fill_value, kwargs, dtype):
        assert len(param) >= len(func)
    for reduction, fv, kw, dt in zip(func, fill_value, kwargs, dtype):
        if empty:
            result = np.full(shape=final_array_shape, fill_value=fv)
        else:
            if is_nanlen(reduction) and is_nanlen(previous_reduction):
                result = results["intermediates"][-1]

            # fill_value here is necessary when reducing with "offset" groups
            kwargs = dict(size=size, dtype=dt, fill_value=fv)
            kwargs.update(kw)

            if callable(reduction):
                # passing a custom reduction for npg to apply per-group is really slow!
                # So this `reduction` has to do the groupby-aggregation
                result = reduction(group_idx, array, **kwargs)
            else:
                result = generic_aggregate(
                    group_idx, array, axis=-1, engine=engine, func=reduction, **kwargs
                ).astype(dt, copy=False)
            if np.any(props.nanmask):
                # remove NaN group label which should be last
                result = result[..., :-1]
            result = result.reshape(final_array_shape[:-1] + found_groups_shape)
        results["intermediates"].append(result)

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
    expected_groups: pd.Index | None,
    fill_value: Any,
    reindex: bool,
):
    """Finalize results by
    1. Squeezing out dummy dimensions
    2. Calling agg.finalize with intermediate results
    3. Mask using counts and fill with user-provided fill_value.
    4. reindex to expected_groups
    """
    squeezed = _squeeze_results(results, axis)

    if agg.min_count is not None:
        counts = squeezed["intermediates"][-1]
        squeezed["intermediates"] = squeezed["intermediates"][:-1]

    # finalize step
    finalized: dict[str, DaskArray | np.ndarray] = {}
    if agg.finalize is None:
        finalized[agg.name] = squeezed["intermediates"][0]
    else:
        finalized[agg.name] = agg.finalize(*squeezed["intermediates"], **agg.finalize_kwargs)

    if agg.min_count is not None:
        count_mask = counts < agg.min_count
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

    return finalized


def _aggregate(
    x_chunk,
    combine: Callable,
    agg: Aggregation,
    expected_groups: pd.Index | None,
    axis: Sequence,
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


def _simple_combine(
    x_chunk, agg: Aggregation, axis: Sequence, keepdims: bool, is_aggregate: bool = False
) -> IntermediateDict:
    """
    'Simple' combination of blockwise results.

    1. After the blockwise groupby-reduce, all blocks contain a value for all possible groups,
       and are of the same shape; i.e. reindex must have been True
    2. _expand_dims was used to insert an extra axis DUMMY_AXIS
    3. Here we concatenate along DUMMY_AXIS, and then call the combine function along
       DUMMY_AXIS
    4. At the final agggregate step, we squeeze out DUMMY_AXIS
    """
    from dask.array.core import deepfirst

    results = {"groups": deepfirst(x_chunk)["groups"]}
    results["intermediates"] = []
    for idx, combine in enumerate(agg.combine):
        array = _conc2(x_chunk, key1="intermediates", key2=idx, axis=axis[:-1] + (DUMMY_AXIS,))
        assert array.ndim >= 2
        result = getattr(np, combine)(array, axis=axis[:-1] + (DUMMY_AXIS,), keepdims=True)
        if is_aggregate:
            # squeeze out DUMMY_AXIS if this is the last step i.e. called from _aggregate
            result = result.squeeze(axis=DUMMY_AXIS)
        results["intermediates"].append(result)
    return results


def _conc2(x_chunk, key1, key2=slice(None), axis=None) -> np.ndarray:
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


def reindex_intermediates(x, agg, unique_groups):
    new_shape = x["groups"].shape[:-1] + (len(unique_groups),)
    newx = {"groups": np.broadcast_to(unique_groups, new_shape)}
    newx["intermediates"] = tuple(
        reindex_(
            v, from_=np.atleast_1d(x["groups"].squeeze()), to=pd.Index(unique_groups), fill_value=f
        )
        for v, f in zip(x["intermediates"], agg.fill_value["intermediate"])
    )
    return newx


def listify_groups(x):
    return list(np.atleast_1d(x["groups"].squeeze()))


def _grouped_combine(
    x_chunk,
    agg: Aggregation,
    axis: Sequence,
    keepdims: bool,
    neg_axis: Sequence,
    engine: str,
    is_aggregate: bool = False,
    sort: bool = True,
) -> IntermediateDict:
    """Combine intermediates step of tree reduction."""
    from dask.base import flatten
    from dask.utils import deepmap

    if isinstance(x_chunk, dict):
        # Only one block at final step; skip one extra groupby
        return x_chunk

    if len(axis) != 1:
        # when there's only a single axis of reduction, we can just concatenate later,
        # reindexing is unnecessary
        # I bet we can minimize the amount of reindexing for mD reductions too, but it's complicated
        unique_groups = np.unique(tuple(flatten(deepmap(listify_groups, x_chunk))))
        unique_groups = unique_groups[~isnull(unique_groups)]
        if len(unique_groups) == 0:
            unique_groups = [np.nan]

        x_chunk = deepmap(
            partial(reindex_intermediates, agg=agg, unique_groups=unique_groups), x_chunk
        )

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
            results = {"groups": groups, "intermediates": list(array_idx)}
        else:
            results = chunk_argreduce(
                array_idx,
                groups,
                func=agg.combine[slicer],  # count gets treated specially next
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
                    sort=sort,
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
    group_chunks = normalize_chunks(np.ceil(ngroups / split_out), (ngroups,))
    idx = tuple(np.cumsum((0,) + group_chunks[0]))

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


def _reduce_blockwise(array, by, agg, *, axis, expected_groups, fill_value, engine, sort, reindex):
    """
    Blockwise groupby reduction that produces the final result. This code path is
    also used for non-dask array aggregations.
    """
    # for pure numpy grouping, we just use npg directly and avoid "finalizing"
    # (agg.finalize = None). We still need to do the reindexing step in finalize
    # so that everything matches the dask version.
    agg.finalize = None

    assert agg.finalize_kwargs is not None
    finalize_kwargs = agg.finalize_kwargs
    if isinstance(finalize_kwargs, Mapping):
        finalize_kwargs = (finalize_kwargs,)
    finalize_kwargs = finalize_kwargs + ({},) + ({},)

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
        kwargs=finalize_kwargs,
        engine=engine,
        sort=sort,
        reindex=reindex,
    )  # type: ignore

    if _is_arg_reduction(agg):
        results["intermediates"][0] = np.unravel_index(results["intermediates"][0], array.shape)[-1]

    result = _finalize_results(
        results, agg, axis, expected_groups, fill_value=fill_value, reindex=reindex
    )
    return result


def dask_groupby_agg(
    array: DaskArray,
    by: DaskArray | np.ndarray,
    agg: Aggregation,
    expected_groups: pd.Index | None,
    axis: Sequence = None,
    split_out: int = 1,
    fill_value: Any = None,
    method: str = "map-reduce",
    reindex: bool = False,
    engine: str = "numpy",
    sort: bool = True,
) -> tuple[DaskArray, np.ndarray | DaskArray]:

    import dask.array
    from dask.array.core import slices_from_chunks
    from dask.highlevelgraph import HighLevelGraph

    # I think _tree_reduce expects this
    assert isinstance(axis, Sequence)
    assert all(ax >= 0 for ax in axis)

    if method == "blockwise" and (split_out > 1 or not isinstance(by, np.ndarray)):
        raise NotImplementedError

    if split_out > 1 and expected_groups is None:
        # This could be implemented using the "hash_split" strategy
        # from dask.dataframe
        raise NotImplementedError

    inds = tuple(range(array.ndim))
    name = f"groupby_{agg.name}"
    token = dask.base.tokenize(array, by, agg, expected_groups, axis, split_out)

    if expected_groups is None and (reindex or split_out > 1):
        expected_groups = _get_expected_groups(by, sort=sort)

    by_input = by

    # Unifying chunks is necessary for argreductions.
    # We need to rechunk before zipping up with the index
    # let's always do it anyway
    if not is_duck_dask_array(by):
        # chunk numpy arrays like the input array
        # This removes an extra rechunk-merge layer that would be
        # added otherwise
        by = dask.array.from_array(by, chunks=tuple(array.chunks[ax] for ax in range(-by.ndim, 0)))
    _, (array, by) = dask.array.unify_chunks(array, inds, by, inds[-by.ndim :])

    # preprocess the array: for argreductions, this zips the index together with the array block
    if agg.preprocess:
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

    do_simple_combine = (
        method != "blockwise" and reindex and not _is_arg_reduction(agg) and split_out == 1
    )
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
            reindex=reindex or (split_out > 1),
        )
        if do_simple_combine:
            # Add a dummy dimension that then gets reduced over
            blockwise_method = tlz.compose(_expand_dims, blockwise_method)

    # apply reduction on chunk
    applied = dask.array.blockwise(
        partial(
            blockwise_method,
            axis=axis,
            expected_groups=expected_groups,
            engine=engine,
            sort=sort,
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
        intermediate, group_chunks = split_blocks(
            applied, split_out, expected_groups, split_name=f"{name}-split-{token}"
        )
    else:
        intermediate = applied
        if expected_groups is None:
            expected_groups = _get_expected_groups(by_input, sort=sort, raise_if_dask=False)
        group_chunks = ((len(expected_groups),) if expected_groups is not None else (np.nan,),)

    if method == "map-reduce":
        # these are negative axis indices useful for concatenating the intermediates
        neg_axis = tuple(range(-len(axis), 0))

        combine = (
            _simple_combine
            if do_simple_combine
            else partial(_grouped_combine, engine=engine, neg_axis=neg_axis, sort=sort)
        )

        # reduced is really a dict mapping reduction name to array
        # and "groups" to an array of group labels
        # Note: it does not make sense to interpret axis relative to
        # shape of intermediate results after the blockwise call
        reduced = dask.array.reductions._tree_reduce(
            intermediate,
            aggregate=partial(
                _aggregate,
                combine=combine,
                agg=agg,
                expected_groups=None if split_out > 1 else expected_groups,
                fill_value=fill_value,
                reindex=reindex,
            ),
            combine=partial(combine, agg=agg),
            name=f"{name}-reduce",
            dtype=array.dtype,
            axis=axis,
            keepdims=True,
            concatenate=False,
        )
        output_chunks = reduced.chunks[: -(len(axis) + int(split_out > 1))] + group_chunks
    elif method == "blockwise":
        reduced = intermediate
        # Here one input chunk → one output chunka
        # find number of groups in each chunk, this is needed for output chunks
        # along the reduced axis
        slices = slices_from_chunks(tuple(array.chunks[ax] for ax in axis))
        if expected_groups is None:
            groups_in_block = tuple(np.unique(by_input[slc]) for slc in slices)
        else:
            # For cohorts, we could be indexing a block with groups that
            # are not in the cohort (usually for nD `by`)
            # Only keep the expected groups.
            groups_in_block = tuple(
                np.intersect1d(by_input[slc], expected_groups) for slc in slices
            )
        ngroups_per_block = tuple(len(groups) for groups in groups_in_block)
        output_chunks = reduced.chunks[: -(len(axis))] + (ngroups_per_block,)
    else:
        raise ValueError(f"Unknown method={method}.")

    # extract results from the dict
    result: dict = {}
    layer: dict[tuple, tuple] = {}
    ochunks = tuple(range(len(chunks_v)) for chunks_v in output_chunks)
    if is_duck_dask_array(by_input) and expected_groups is None:
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
                chunks=group_chunks,
                dtype=by.dtype,
            ),
        )
    else:
        if method == "map-reduce":
            if expected_groups is None:
                expected_groups = _get_expected_groups(by_input, sort=sort)
            groups = (expected_groups.values,)
        else:
            groups = (np.concatenate(groups_in_block),)

    layer: dict[tuple, tuple] = {}  # type: ignore
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
        layer[(agg_name, *ochunk)] = (operator.getitem, (reduced.name, *inchunk), agg.name)

    result = dask.array.Array(
        HighLevelGraph.from_collections(agg_name, layer, dependencies=[reduced]),
        agg_name,
        chunks=output_chunks,
        dtype=agg.dtype[agg.name],
    )

    return (result, *groups)


def _validate_reindex(reindex: bool, func, method, expected_groups) -> bool:
    if reindex is True and _is_arg_reduction(func):
        raise NotImplementedError

    if method == "blockwise" and reindex is True:
        raise NotImplementedError

    if method == "blockwise" or _is_arg_reduction(func):
        reindex = False

    if reindex is None and expected_groups is not None:
        reindex = True

    if method in ["split-reduce", "cohorts"] and reindex is False:
        raise NotImplementedError

    return reindex


def _assert_by_is_aligned(shape, by):
    for idx, b in enumerate(by):
        if shape[-b.ndim :] != b.shape:
            raise ValueError(
                "`array` and `by` arrays must be aligned "
                "i.e. array.shape[-by.ndim :] == by.shape. "
                "for every array in `by`."
                f"Received array of shape {shape} but "
                f"array {idx} in `by` has shape {b.shape}."
            )


def _convert_expected_groups_to_index(
    expected_groups: tuple, isbin: bool, sort: bool
) -> pd.Index | None:
    out = []
    for ex, isbin_ in zip(expected_groups, isbin):
        if isinstance(ex, pd.IntervalIndex) or (isinstance(ex, pd.Index) and not isbin):
            if sort:
                ex = ex.sort_values()
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


def _lazy_factorize_wrapper(*by, **kwargs):
    group_idx, *rest = factorize_(by, **kwargs)
    return group_idx


def _factorize_multiple(by, expected_groups, by_is_dask):
    kwargs = dict(
        expected_groups=expected_groups,
        axis=None,  # always None, we offset later if necessary.
        fastpath=True,
    )
    if by_is_dask:
        import dask.array

        group_idx = dask.array.map_blocks(
            _lazy_factorize_wrapper,
            *np.broadcast_arrays(*by),
            meta=np.array((), dtype=np.int64),
            **kwargs,
        )
        found_groups = tuple(None if is_duck_dask_array(b) else pd.unique(b) for b in by)
        grp_shape = tuple(len(e) for e in expected_groups)
    else:
        group_idx, found_groups, grp_shape = factorize_(by, **kwargs)

    final_groups = tuple(
        found if expect is None else expect.to_numpy()
        for found, expect in zip(found_groups, expected_groups)
    )

    if any(grp is None for grp in final_groups):
        raise ValueError("Please provide expected_groups when grouping by a dask array.")
    return (group_idx,), final_groups, grp_shape


def groupby_reduce(
    array: np.ndarray | DaskArray,
    *by: np.ndarray | DaskArray,
    func: str | Aggregation,
    expected_groups: Sequence | np.ndarray | None = None,
    sort: bool = True,
    isbin: bool = False,
    axis=None,
    fill_value=None,
    min_count: int | None = None,
    split_out: int = 1,
    method: str = "map-reduce",
    engine: str = "flox",
    reindex: bool | None = None,
    finalize_kwargs: Mapping | None = None,
) -> tuple[DaskArray, np.ndarray | DaskArray]:
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
    engine : {"flox", "numpy", "numba"}, optional
        Algorithm to compute the groupby reduction on non-dask arrays and on each dask chunk:
          * ``"flox"``:
            Use an internal implementation where the data is sorted so that
            all members of a group occur sequentially, and then numpy.ufunc.reduceat
            is to used for the reduction. This will fall back to ``numpy_groupies.aggregate_numpy``
            for a reduction that is not yet implemented.
          * ``"numpy"``:
            Use the vectorized implementations in ``numpy_groupies.aggregate_numpy``.
          * ``"numba"``:
            Use the implementations in ``numpy_groupies.aggregate_numba``.
    reindex : bool, optional
        Whether to "reindex" the blockwise results to `expected_groups` (possibly automatically detected).
        If True, the intermediate result of the blockwise groupby-reduction has a value for all expected groups,
        and the final result is a simple reduction of those intermediates. In nearly all cases, this is a significant
        boost in computation speed. For cases like time grouping, this may result in large intermediates relative to the
        original block size. Avoid that by using method="cohorts". By default, it is turned off for argreductions.
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

    if engine == "flox" and _is_arg_reduction(func):
        raise NotImplementedError(
            "argreductions not supported for engine='flox' yet."
            "Try engine='numpy' or engine='numba' instead."
        )
    reindex = _validate_reindex(reindex, func, method, expected_groups)

    by: tuple = tuple(np.asarray(b) if not is_duck_array(b) else b for b in by)
    nby = len(by)
    by_is_dask = any(is_duck_dask_array(b) for b in by)

    if method in ["split-reduce", "cohorts"] and by_is_dask:
        raise ValueError(f"method={method!r} can only be used when grouping by numpy arrays.")

    if not is_duck_array(array):
        array = np.asarray(array)
    array = array.astype(int) if np.issubdtype(array.dtype, bool) else array

    if isinstance(isbin, bool):
        isbin = (isbin,) * len(by)
    if expected_groups is None:
        expected_groups = (None,) * len(by)

    _assert_by_is_aligned(array.shape, by)

    if len(by) == 1 and not isinstance(expected_groups, tuple):
        expected_groups = (np.asarray(expected_groups),)
    elif len(expected_groups) != len(by):
        raise ValueError("len(expected_groups) != len(by)")

    # We convert to pd.Index since that lets us know if we are binning or not
    # (pd.IntervalIndex or not)
    expected_groups = _convert_expected_groups_to_index(expected_groups, isbin, sort)

    # TODO: could restrict this to dask-only
    factorize_early = (nby > 1) or (
        any(isbin) and method in ["split-reduce", "cohorts"] and is_duck_dask_array(array)
    )
    if factorize_early:
        by, final_groups, grp_shape = _factorize_multiple(
            by, expected_groups, by_is_dask=by_is_dask
        )
        expected_groups = (pd.RangeIndex(np.prod(grp_shape)),)

    assert len(by) == 1
    by = by[0]
    expected_groups = expected_groups[0]

    if axis is None:
        axis = tuple(array.ndim + np.arange(-by.ndim, 0))
    else:
        axis = np.core.numeric.normalize_axis_tuple(axis, array.ndim)  # type: ignore

    if method in ["blockwise", "cohorts", "split-reduce"] and len(axis) != by.ndim:
        raise NotImplementedError(
            "Must reduce along all dimensions of `by` when method != 'map-reduce'."
            f"Received method={method!r}"
        )

    # TODO: make sure expected_groups is unique
    if len(axis) == 1 and by.ndim > 1 and expected_groups is None:
        if not by_is_dask:
            expected_groups = _get_expected_groups(by, sort)
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

    assert len(axis) <= by.ndim
    if len(axis) < by.ndim:
        by = _move_reduce_dims_to_end(by, -array.ndim + np.array(axis) + by.ndim)
        array = _move_reduce_dims_to_end(array, axis)
        axis = tuple(array.ndim + np.arange(-len(axis), 0))

    has_dask = is_duck_dask_array(array) or is_duck_dask_array(by)

    # When axis is a subset of possible values; then npg will
    # apply it to groups that don't exist along a particular axis (for e.g.)
    # since these count as a group that is absent. thoo!
    # fill_value applies to all-NaN groups as well as labels in expected_groups that are not found.
    #     The only way to do this consistently is mask out using min_count
    #     Consider np.sum([np.nan]) = np.nan, np.nansum([np.nan]) = 0
    if min_count is None:
        if len(axis) < by.ndim or fill_value is not None:
            min_count = 1

    # TODO: set in xarray?
    if min_count is not None and func in ["nansum", "nanprod"] and fill_value is None:
        # nansum, nanprod have fill_value=0, 1
        # overwrite than when min_count is set
        fill_value = np.nan

    kwargs = dict(axis=axis, fill_value=fill_value, engine=engine, sort=sort)
    agg = _initialize_aggregation(func, array.dtype, fill_value, min_count, finalize_kwargs)

    if not has_dask:
        results = _reduce_blockwise(
            array, by, agg, expected_groups=expected_groups, reindex=reindex, **kwargs
        )
        groups = (results["groups"],)
        result = results[agg.name]

    else:
        if agg.chunk is None:
            raise NotImplementedError(f"{func} not implemented for dask arrays")

        # we always need some fill_value (see above) so choose the default if needed
        if kwargs["fill_value"] is None:
            kwargs["fill_value"] = agg.fill_value[agg.name]

        partial_agg = partial(dask_groupby_agg, agg=agg, split_out=split_out, **kwargs)

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

                # First deep copy becasue we might be doping blockwise,
                # which sets agg.finalize=None, then map-reduce (GH102)
                agg = copy.deepcopy(agg)

                # get final result for these groups
                r, *g = partial_agg(
                    array_subset,
                    by[np.ix_(*indexer)],
                    expected_groups=pd.Index(cohort),
                    # reindex to expected_groups at the blockwise step.
                    # this approach avoids replacing non-cohort members with
                    # np.nan or some other sentinel value, and preserves dtypes
                    reindex=True,
                    # sort controls the final output order so apply that at the end
                    sort=False,
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
            if method == "blockwise" and by.ndim == 1:
                array = rechunk_for_blockwise(array, axis=-1, labels=by)

            result, *groups = partial_agg(
                array,
                by,
                expected_groups=None if method == "blockwise" else expected_groups,
                reindex=reindex,
                method=method,
                sort=sort,
            )

        if sort and method != "map-reduce":
            assert len(groups) == 1
            sorted_idx = np.argsort(groups[0])
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
    return (result, *groups)
