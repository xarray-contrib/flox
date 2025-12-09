"""Dask-specific functions for groupby operations."""

from __future__ import annotations

import itertools
import operator
from collections.abc import Callable, Sequence
from functools import partial
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import toolz as tlz

if TYPE_CHECKING:
    from typing import Literal

    from .aggregations import Aggregation
    from .core import T_Axes, T_Engine, T_Method
    from .lib import ArrayLayer
    from .reindex import ReindexArrayType, ReindexStrategy
    from .types import DaskArray, Graph, IntermediateDict, T_By

    T_ScanMethod = Literal["blelloch", "blockwise"]

from .aggregations import Scan, scan_binary_op
from .core import (
    DUMMY_AXIS,
    _get_chunk_reduction,
    _reduce_blockwise,
    _unique,
    chunk_argreduce,
    chunk_reduce,
)
from .lib import _is_arg_reduction, _is_first_last_reduction, _issorted, identity
from .reindex import (
    ReindexArrayType,
    ReindexStrategy,
    reindex_,
)
from .scan import _finalize_scan, _zip, chunk_scan, grouped_reduce
from .types import FinalResultsDict, IntermediateDict
from .xrutils import is_duck_dask_array, notnull


def listify_groups(x: IntermediateDict):
    return list(np.atleast_1d(x["groups"].squeeze()))


def _find_unique_groups(x_chunk) -> np.ndarray:
    from dask.base import flatten
    from dask.utils import deepmap

    unique_groups = _unique(np.asarray(tuple(flatten(deepmap(listify_groups, x_chunk)))))
    unique_groups = unique_groups[notnull(unique_groups)]

    if len(unique_groups) == 0:
        unique_groups = np.array([np.nan])
    return unique_groups


def _conc2(x_chunk, key1, key2=slice(None), axis=None) -> np.ndarray:
    """copied from dask.array.reductions.mean_combine"""
    from dask.array.core import _concatenate2
    from dask.utils import deepmap

    mapped = deepmap(lambda x: x[key1][key2], x_chunk)
    return _concatenate2(mapped, axes=axis)


def reindex_intermediates(
    x: IntermediateDict, agg: Aggregation, unique_groups, array_type
) -> IntermediateDict:
    new_shape = x["groups"].shape[:-1] + (len(unique_groups),)
    newx: IntermediateDict = {"groups": np.broadcast_to(unique_groups, new_shape)}
    newx["intermediates"] = tuple(
        reindex_(
            v,
            from_=np.atleast_1d(x["groups"].squeeze()),
            to=pd.Index(unique_groups),
            fill_value=f,
            array_type=array_type,
        )
        for v, f in zip(x["intermediates"], agg.fill_value["intermediate"])
    )
    return newx


def _simple_combine(
    x_chunk,
    agg: Aggregation,
    axis,
    keepdims: bool,
    reindex: ReindexStrategy,
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
    import warnings

    from dask.array.core import deepfirst
    from dask.utils import deepmap

    if not reindex.blockwise:
        # We didn't reindex at the blockwise step
        # So now reindex before combining by reducing along DUMMY_AXIS
        unique_groups = _find_unique_groups(x_chunk)
        x_chunk = deepmap(
            partial(
                reindex_intermediates,
                agg=agg,
                unique_groups=unique_groups,
                array_type=reindex.array_type,
            ),
            x_chunk,
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
            # can't just pass DUMMY_AXIS, because of sparse.COO
            result = result.squeeze(range(result.ndim)[DUMMY_AXIS])
        results["intermediates"].append(result)
    return results


def _extract_result(result_dict: FinalResultsDict, key) -> np.ndarray:
    from dask.array.core import deepfirst

    # deepfirst should be not be needed here but sometimes we receive a list of dict?
    return deepfirst(result_dict)[key]


def _expand_dims(results: IntermediateDict) -> IntermediateDict:
    results["intermediates"] = tuple(
        np.expand_dims(array, axis=DUMMY_AXIS) for array in results["intermediates"]
    )
    return results


def _aggregate(
    x_chunk,
    combine,
    agg: Aggregation,
    expected_groups,
    axis,
    keepdims: bool,
    fill_value,
    reindex: ReindexStrategy,
) -> FinalResultsDict:
    """Final aggregation step of tree reduction"""
    from .core import _finalize_results

    results = combine(x_chunk, agg, axis, keepdims, is_aggregate=True)
    return _finalize_results(results, agg, axis, expected_groups, reindex=reindex)


def _unify_chunks(array, by):
    from dask.array import from_array, unify_chunks

    inds = tuple(range(array.ndim))

    # Unifying chunks is necessary for argreductions.
    # We need to rechunk before zipping up with the index
    # let's always do it anyway
    if not is_duck_dask_array(by):
        # chunk numpy arrays like the input array
        # This removes an extra rechunk-merge layer that would be
        # added otherwise
        chunks = tuple(array.chunks[ax] if by.shape[ax] != 1 else (1,) for ax in range(-by.ndim, 0))

        by = from_array(by, chunks=chunks)
    _, (array, by) = unify_chunks(array, inds, by, inds[-by.ndim :])

    return array, by


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
            partial(
                reindex_intermediates, agg=agg, unique_groups=unique_groups, array_type=ReindexArrayType.AUTO
            ),
            x_chunk,
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
        array_idx = tuple(_conc2(x_chunk, key1="intermediates", key2=idx, axis=axis) for idx in (0, 1))

        # for a single element along axis, we don't want to run the argreduction twice
        # This happens when we are reducing along an axis with a single chunk.
        avoid_reduction = array_idx[0].shape[axis[0]] == 1
        if avoid_reduction:
            results: IntermediateDict = {
                "groups": groups,
                "intermediates": list(array_idx),
            }
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
                # No groups found in input data. Return to avoid a tree-reduce
                # step with no data.
                results["groups"] = groups
                results["intermediates"].append(array)
                continue
            reduced = chunk_reduce(
                array,
                groups,
                axis=axis,
                func=combine_,
                expected_groups=None,
                fill_value=(fv,),
                dtype=(dtype,),
                engine=engine,
                sort=sort,
                user_dtype=agg.dtype["user"],
            )
            # we had groups so this should've been set
            if results["groups"] is None:
                results["groups"] = reduced["groups"]
            results["intermediates"].append(reduced["intermediates"][0])

    # final pass and add keepdims=False.
    results["groups"] = results["groups"].squeeze() if not keepdims else results["groups"]

    return results


def dask_groupby_agg(
    array: DaskArray,
    by: T_By,
    *,
    agg: Aggregation,
    expected_groups: pd.RangeIndex | None,
    reindex: ReindexStrategy,
    axis: T_Axes = (),
    fill_value: Any = None,
    method: T_Method = "map-reduce",
    engine: T_Engine = "numpy",
    sort: bool = True,
    chunks_cohorts=None,
) -> tuple[DaskArray, tuple[pd.Index | np.ndarray | DaskArray]]:
    import dask.array
    from dask.array.core import slices_from_chunks
    from dask.highlevelgraph import HighLevelGraph

    from .dask_array_ops import _tree_reduce

    # I think _tree_reduce expects this
    assert isinstance(axis, Sequence)
    assert all(ax >= 0 for ax in axis)

    inds = tuple(range(array.ndim))
    name = f"groupby_{agg.name}"

    if expected_groups is None and reindex.blockwise:
        raise ValueError("reindex.blockwise must be False-y if expected_groups is not provided.")
    if method == "cohorts" and reindex.blockwise:
        raise ValueError("reindex.blockwise must be False-y if method is 'cohorts'.")

    by_input = by

    array, by = _unify_chunks(array, by)

    # tokenize here since by has already been hashed if its numpy
    token = dask.base.tokenize(array, by, agg, expected_groups, axis, method)

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
    #        reindex.blockwise=True, and we must know expected_groups
    #    b. "_grouped_combine": A more general solution where we tree-reduce the groupby reduction.
    #       This allows us to discover groups at compute time, support argreductions, lower intermediate
    #       memory usage (but method="cohorts" would also work to reduce memory in some cases)
    labels_are_unknown = is_duck_dask_array(by_input) and expected_groups is None
    do_grouped_combine = (
        _is_arg_reduction(agg)
        or labels_are_unknown
        or (_is_first_last_reduction(agg) and array.dtype.kind != "f")
    )
    do_simple_combine = not do_grouped_combine

    if method == "blockwise":
        #  use the "non dask" code path, but applied blockwise
        blockwise_method = partial(_reduce_blockwise, agg=agg, fill_value=fill_value, reindex=reindex)
    else:
        # choose `chunk_reduce` or `chunk_argreduce`
        blockwise_method = partial(
            _get_chunk_reduction(agg.reduction_type),
            func=agg.chunk,
            reindex=reindex.blockwise,
            fill_value=agg.fill_value["intermediate"],
            dtype=agg.dtype["intermediate"],
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
            expected_groups=expected_groups if reindex.blockwise else None,
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
        combine: Callable[..., IntermediateDict] = (
            partial(_simple_combine, reindex=reindex)
            if do_simple_combine
            else partial(_grouped_combine, engine=engine, sort=sort)
        )

        tree_reduce = partial(
            dask.array.reductions._tree_reduce,
            name=f"{name}-simple-reduce",
            dtype=array.dtype,
            axis=axis,
            keepdims=True,
            concatenate=False,
        )
        aggregate = partial(_aggregate, combine=combine, agg=agg, fill_value=fill_value, reindex=reindex)

        # Each chunk of `reduced`` is really a dict mapping
        # 1. reduction name to array
        # 2. "groups" to an array of group labels
        # Note: it does not make sense to interpret axis relative to
        # shape of intermediate results after the blockwise call
        if method == "map-reduce":
            reduced = tree_reduce(
                intermediate,
                combine=partial(combine, agg=agg),
                aggregate=partial(aggregate, expected_groups=expected_groups),
            )
            if labels_are_unknown:
                groups = _extract_unknown_groups(reduced, dtype=by.dtype)
                group_chunks = ((np.nan,),)
            else:
                assert expected_groups is not None
                groups = (expected_groups,)
                group_chunks = ((len(expected_groups),),)

        elif method == "cohorts":
            assert chunks_cohorts
            block_shape = array.blocks.shape[-len(axis) :]

            out_name = f"{name}-reduce-{method}-{token}"
            groups_ = []
            chunks_as_array = tuple(np.array(c) for c in array.chunks)
            dsk: Graph = {}
            for icohort, (blks, cohort) in enumerate(chunks_cohorts.items()):
                cohort_index = pd.Index(cohort)
                reindexer = (
                    partial(
                        reindex_intermediates,
                        agg=agg,
                        unique_groups=cohort_index,
                        array_type=reindex.array_type,
                    )
                    if do_simple_combine
                    else identity
                )
                subset = subset_to_blocks(intermediate, blks, block_shape, reindexer, chunks_as_array)
                dsk |= subset.layer  # type: ignore[operator]
                # now that we have reindexed, we can set reindex=True explicitlly
                new_reindex = ReindexStrategy(blockwise=do_simple_combine, array_type=reindex.array_type)
                _tree_reduce(
                    subset,
                    out_dsk=dsk,
                    name=out_name,
                    block_index=icohort,
                    axis=axis,
                    combine=partial(combine, agg=agg, reindex=new_reindex, keepdims=True),
                    aggregate=partial(
                        aggregate, expected_groups=cohort_index, reindex=new_reindex, keepdims=True
                    ),
                )
                # This is done because pandas promotes to 64-bit types when an Index is created
                # So we use the index to generate the return value for consistency with "map-reduce"
                # This is important on windows
                groups_.append(cohort_index.values)

            graph = HighLevelGraph.from_collections(out_name, dsk, dependencies=[intermediate])

            out_chunks = list(array.chunks)
            out_chunks[axis[-1]] = tuple(len(c) for c in chunks_cohorts.values())
            for ax in axis[:-1]:
                out_chunks[ax] = (1,)
            reduced = dask.array.Array(graph, out_name, out_chunks, meta=array._meta)

            groups = (np.concatenate(groups_),)
            group_chunks = (tuple(len(cohort) for cohort in groups_),)

    elif method == "blockwise":
        reduced = intermediate
        if reindex.blockwise:
            if TYPE_CHECKING:
                assert expected_groups is not None
            # TODO: we could have `expected_groups` be a dask array with appropriate chunks
            # for now, we have a numpy array that is interpreted as listing all group labels
            # that are present in every chunk
            groups = (expected_groups,)
            group_chunks = ((len(expected_groups),),)
        else:
            # TODO: use chunks_cohorts here; hard because chunks_cohorts does not include all-NaN blocks
            #       but the array after applying the blockwise op; does. We'd have to insert a subsetting op.
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

    # Adjust output for any new dimensions added, example for multiple quantiles
    new_dims_shape = tuple(dim.size for dim in agg.new_dims if not dim.is_scalar)
    new_inds = tuple(range(-len(new_dims_shape), 0))
    out_inds = new_inds + inds[: -len(axis)] + (inds[-1],)
    output_chunks = new_dims_shape + reduced.chunks[: -len(axis)] + group_chunks
    new_axes = dict(zip(new_inds, new_dims_shape))

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
        key=agg.name,
        name=f"{name}-{token}",
        concatenate=False,
        new_axes=new_axes,
        meta=reindex.get_dask_meta(array, dtype=agg.dtype["final"], fill_value=agg.fill_value[agg.name]),
    )

    return (result, groups)


def dask_groupby_scan(array, by, axes: T_Axes, agg: Scan, method: T_ScanMethod = "blelloch") -> DaskArray:
    """Grouped scan for dask arrays.

    Parameters
    ----------
    array : DaskArray
        Input array to scan.
    by : DaskArray
        Group labels array, must have same chunks as array along scan axis.
    axes : T_Axes
        Tuple of axes to scan along (must be single axis).
    agg : Scan
        Scan aggregation specification.
    method : {"blelloch", "blockwise"}, optional
        Scan method to use:
        - "blelloch": Blelloch parallel prefix scan algorithm, allows scanning
          across chunk boundaries using tree reduction. Default.
        - "blockwise": Each chunk is processed independently. Only valid when
          all members of each group are contained within a single chunk.

    Returns
    -------
    DaskArray
        Result of the grouped scan with same shape and chunks as input.
    """
    from dask.array import map_blocks
    from dask.array.reductions import cumreduction as scan
    from dask.base import tokenize

    if len(axes) > 1:
        raise NotImplementedError("Scans are only supported along a single axis.")
    (axis,) = axes

    array, by = _unify_chunks(array, by)

    # Include method in token to differentiate task graphs
    token = tokenize(array, by, agg, axes, method)

    # 1. zip together group indices & array
    zipped = map_blocks(
        _zip,
        by,
        array,
        dtype=array.dtype,
        meta=array._meta,
        name=f"groupby-scan-preprocess-{token}",
    )

    # 2. Run the scan
    if method == "blockwise":
        # Apply chunk_scan blockwise - each block independently
        scan_func = partial(chunk_scan, agg=agg, axis=axis, dtype=agg.dtype)
        scanned = map_blocks(
            scan_func,
            zipped,
            dtype=agg.dtype,
            meta=array._meta,
            name=f"groupby-scan-{token}",
        )
    else:
        # Use Blelloch parallel prefix scan algorithm
        scan_ = partial(chunk_scan, agg=agg)
        # dask tokenizing error workaround
        scan_.__name__ = scan_.func.__name__  # type: ignore[attr-defined]

        scanned = scan(
            func=scan_,
            binop=partial(scan_binary_op, agg=agg),
            ident=agg.identity,
            x=zipped,
            axis=axis,
            # TODO: support method="sequential" here.
            method="blelloch",
            preop=partial(grouped_reduce, agg=agg),
            dtype=agg.dtype,
        )

    # 3. Extract final result
    result = map_blocks(
        partial(_finalize_scan, dtype=agg.dtype),
        scanned,
        dtype=agg.dtype,
        name=f"groupby-scan-finalize-{token}",
    )

    assert result.chunks == array.chunks

    return result


def _normalize_indexes(ndim: int, flatblocks: Sequence[int], blkshape: tuple[int, ...]) -> tuple:
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
            if len(i) == blkshape[ax] and np.array_equal(i, np.arange(blkshape[ax])):
                normalized.append(slice(None))
            elif _issorted(i) and np.array_equal(i, np.arange(i[0], i[-1] + 1)):
                start = None if i[0] == 0 else i[0]
                stop = i[-1] + 1
                stop = None if stop == blkshape[ax] else stop
                normalized.append(slice(start, stop))
            else:
                normalized.append(list(i))
    full_normalized = (slice(None),) * (ndim - len(normalized)) + tuple(normalized)

    # has no iterables
    noiter = list(i if not hasattr(i, "__len__") else slice(None) for i in full_normalized)
    # has all iterables
    alliter = {ax: i for ax, i in enumerate(full_normalized) if hasattr(i, "__len__")}

    mesh = dict(zip(alliter.keys(), np.ix_(*alliter.values())))  # type: ignore[arg-type, var-annotated]

    full_tuple = tuple(i if ax not in mesh else mesh[ax] for ax, i in enumerate(noiter))

    return full_tuple


def subset_to_blocks(
    array: DaskArray,
    flatblocks: Sequence[int],
    blkshape: tuple[int, ...] | None = None,
    reindexer=identity,
    chunks_as_array: tuple[np.ndarray, ...] | None = None,
) -> ArrayLayer:
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
    from dask.base import tokenize

    from .lib import ArrayLayer

    if blkshape is None:
        blkshape = array.blocks.shape

    if chunks_as_array is None:
        chunks_as_array = tuple(np.array(c) for c in array.chunks)

    index = _normalize_indexes(array.ndim, flatblocks, blkshape)

    # These rest is copied from dask.array.core.py with slight modifications
    index = tuple(slice(k, k + 1) if isinstance(k, Integral) else k for k in index)

    name = "groupby-cohort-" + tokenize(array, index)
    new_keys = array._key_array[index]

    squeezed = tuple(np.squeeze(i) if isinstance(i, np.ndarray) else i for i in index)
    chunks = tuple(tuple(c[i].tolist()) for c, i in zip(chunks_as_array, squeezed))

    keys = itertools.product(*(range(len(c)) for c in chunks))
    layer: Graph = {(name,) + key: (reindexer, tuple(new_keys[key].tolist())) for key in keys}
    return ArrayLayer(layer=layer, chunks=chunks, name=name)


def _extract_unknown_groups(reduced, dtype) -> tuple[DaskArray]:
    import dask.array
    from dask.highlevelgraph import HighLevelGraph

    groups_token = f"group-{reduced.name}"
    first_block = reduced.ndim * (0,)
    layer: Graph = {(groups_token, 0): (operator.getitem, (reduced.name, *first_block), "groups")}
    groups: tuple[DaskArray] = (
        dask.array.Array(
            HighLevelGraph.from_collections(groups_token, layer, dependencies=[reduced]),
            groups_token,
            chunks=((np.nan,),),
            meta=np.array([], dtype=dtype),
        ),
    )

    return groups


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


__all__ = [
    "_collapse_blocks_along_axes",
    "_extract_unknown_groups",
    "_grouped_combine",
    "_normalize_indexes",
    "_unify_chunks",
    "dask_groupby_agg",
    "dask_groupby_scan",
    "subset_to_blocks",
]
