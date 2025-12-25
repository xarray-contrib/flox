"""Expression-based dask operations for flox.

This module provides expression-system compatible implementations of flox's
dask operations. It is used when dask's array query planning is enabled
(DASK_ARRAY__QUERY_PLANNING=True).

See plans/array-expr-migration.md for migration status and details.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import toolz as tlz

if TYPE_CHECKING:
    from .aggregations import Aggregation
    from .core import T_Axes, T_Engine, T_Method
    from .reindex import ReindexStrategy
    from .types import DaskArray, T_By


# Detection
def _expr_enabled() -> bool:
    """Check if dask array expression system is enabled."""
    try:
        from dask.array import ARRAY_EXPR_ENABLED

        return ARRAY_EXPR_ENABLED
    except ImportError:
        return False


EXPR_ENABLED = _expr_enabled()

if EXPR_ENABLED:
    from dask._task_spec import Alias, Task, TaskRef
    from dask.array._array_expr._collection import new_collection
    from dask.array._array_expr._expr import ArrayExpr

    class ExtractFromDictExpr(ArrayExpr):
        """Extract a key from dict-valued array blocks.

        Used to extract group labels when they are discovered at compute time
        (unknown groups case).
        """

        _parameters = ["array", "key", "_dtype"]

        @property
        def dtype(self):
            return self._dtype

        @property
        def chunks(self):
            return ((np.nan,),)

        @property
        def _meta(self):
            return np.array([], dtype=self.dtype)

        def _layer(self):
            arr = self.array
            arr_name = arr._name
            first_block = arr.ndim * (0,)
            out_key = (self._name, 0)
            in_key = (arr_name, *first_block)
            return {out_key: Task(out_key, operator.getitem, TaskRef(in_key), self.key)}

    class CollapseBlocksExpr(ArrayExpr):
        """Remap block keys to collapse multiple axes into one.

        Used for blockwise method when reducing along multiple axes.
        This is a virtual reshape - no computation, just key aliasing.
        """

        _parameters = ["array", "axes", "group_chunks"]

        @property
        def dtype(self):
            return self.array.dtype

        @property
        def chunks(self):
            arr_chunks = self.array.chunks
            axes = self.axes
            return arr_chunks[: -len(axes)] + ((1,) * (len(axes) - 1),) + self.group_chunks

        @property
        def _meta(self):
            return self.array._meta

        def _layer(self):
            import itertools

            arr = self.array
            arr_name = arr._name
            axes = self.axes
            nblocks = tuple(arr.numblocks[ax] for ax in axes)
            output_chunks = self.chunks

            ochunks = tuple(range(len(chunks_v)) for chunks_v in output_chunks)
            layer = {}
            for ochunk in itertools.product(*ochunks):
                inchunk = ochunk[: -len(axes)] + np.unravel_index(ochunk[-1], nblocks)
                out_key = (self._name, *ochunk)
                in_key = (arr_name, *inchunk)
                layer[out_key] = Alias(out_key, in_key)
            return layer

    class SubsetBlocksExpr(ArrayExpr):
        """Select a subset of blocks from an array and optionally apply a function.

        Used in cohorts method to select blocks belonging to a specific cohort.
        """

        _parameters = ["array", "flatblocks", "blkshape", "reindexer", "output_chunks"]

        @property
        def dtype(self):
            return self.array.dtype

        @property
        def chunks(self):
            return self.output_chunks

        @property
        def _meta(self):
            return self.array._meta

        def _layer(self):
            import itertools

            arr = self.array
            arr_name = arr._name
            flatblocks = self.flatblocks
            blkshape = self.blkshape
            reindexer = self.reindexer
            output_chunks = self.output_chunks

            index = _normalize_indexes(arr.ndim, flatblocks, blkshape)
            index = tuple(slice(k, k + 1) if isinstance(k, Integral) else k for k in index)

            # Build key array manually since we're working with expressions
            # We need to map from new block indices to old block indices
            numblocks = arr.numblocks
            old_keys = np.empty(numblocks, dtype=object)
            for idx in itertools.product(*(range(n) for n in numblocks)):
                old_keys[idx] = (arr_name,) + idx

            new_keys = old_keys[index]

            layer = {}
            for key in itertools.product(*(range(len(c)) for c in output_chunks)):
                old_key = new_keys[key]
                if isinstance(old_key, np.ndarray):
                    old_key = tuple(old_key.flat[0])
                out_key = (self._name,) + key
                layer[out_key] = Task(out_key, reindexer, TaskRef(old_key))
            return layer


# Import shared functions from dask.py - these operate on data, not graphs
from numbers import Integral

from .dask import (
    _aggregate,
    _expand_dims,
    _extract_result,
    _grouped_combine,
    _normalize_indexes,
    _simple_combine,
    _unify_chunks,
    _zip,
    reindex_intermediates,
)
from .core import (
    _get_chunk_reduction,
    _reduce_blockwise,
)
from .lib import _is_arg_reduction, _is_first_last_reduction, identity
from .reindex import ReindexStrategy
from .xrutils import is_duck_dask_array


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
    """Expression-based groupby aggregation for dask arrays.

    This is the expression-system equivalent of flox.dask.dask_groupby_agg().
    Uses dask's expression-aware blockwise and tree_reduce operations.
    """
    import dask
    import dask.array
    from dask.array.core import slices_from_chunks

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

    # preprocess the array
    if agg.preprocess and method != "blockwise":
        array = agg.preprocess(array, axis=axis)

    # Determine combine strategy
    labels_are_unknown = is_duck_dask_array(by_input) and expected_groups is None
    do_grouped_combine = (
        _is_arg_reduction(agg)
        or labels_are_unknown
        or (_is_first_last_reduction(agg) and array.dtype.kind != "f")
    )
    do_simple_combine = not do_grouped_combine

    if method == "blockwise":
        blockwise_method = partial(_reduce_blockwise, agg=agg, fill_value=fill_value, reindex=reindex)
    else:
        blockwise_method = partial(
            _get_chunk_reduction(agg.reduction_type),
            func=agg.chunk,
            reindex=reindex.blockwise,
            fill_value=agg.fill_value["intermediate"],
            dtype=agg.dtype["intermediate"],
            user_dtype=agg.dtype["user"],
        )
        if do_simple_combine:
            blockwise_method = tlz.compose(_expand_dims, blockwise_method)

    # Apply reduction on chunk - uses expression-aware blockwise
    intermediate = dask.array.blockwise(
        partial(
            blockwise_method,
            axis=axis,
            expected_groups=expected_groups if reindex.blockwise else None,
            engine=engine,
            sort=sort,
        ),
        inds,
        array,
        inds,
        by,
        inds[-by.ndim :],
        concatenate=False,
        dtype=array.dtype,
        meta=array._meta,
        align_arrays=False,
        name=f"{name}-chunk-{token}",
    )

    group_chunks: tuple[tuple[int | float, ...]]

    if method == "map-reduce":
        combine = (
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

        reduced = tree_reduce(
            intermediate,
            combine=partial(combine, agg=agg),
            aggregate=partial(aggregate, expected_groups=expected_groups),
        )

        if labels_are_unknown:
            # Use expression class for extracting groups
            groups_expr = ExtractFromDictExpr(reduced.expr, "groups", by.dtype)
            groups = (new_collection(groups_expr),)
            group_chunks = ((np.nan,),)
        else:
            assert expected_groups is not None
            groups = (expected_groups,)
            group_chunks = ((len(expected_groups),),)

    elif method == "cohorts":
        assert chunks_cohorts
        block_shape = intermediate.blocks.shape[-len(axis) :]
        chunks_as_array = tuple(np.array(c) for c in intermediate.chunks)

        combine = (
            partial(_simple_combine, reindex=reindex)
            if do_simple_combine
            else partial(_grouped_combine, engine=engine, sort=sort)
        )
        aggregate = partial(_aggregate, combine=combine, agg=agg, fill_value=fill_value, reindex=reindex)

        cohort_results = []
        groups_ = []

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

            # Compute output chunks for this subset
            index = _normalize_indexes(intermediate.ndim, blks, block_shape)
            index = tuple(slice(k, k + 1) if isinstance(k, Integral) else k for k in index)
            squeezed = tuple(np.squeeze(i) if isinstance(i, np.ndarray) else i for i in index)
            subset_chunks = tuple(tuple(c[i].tolist()) for c, i in zip(chunks_as_array, squeezed))

            # Create subset expression
            subset_expr = SubsetBlocksExpr(
                intermediate.expr, tuple(blks), block_shape, reindexer, subset_chunks
            )
            subset = new_collection(subset_expr)

            # Apply tree reduce using expression-aware API
            new_reindex = ReindexStrategy(blockwise=do_simple_combine, array_type=reindex.array_type)
            cohort_reduced = dask.array.reductions._tree_reduce(
                subset,
                name=f"{name}-cohort-{icohort}-{token}",
                dtype=array.dtype,
                axis=axis,
                keepdims=True,
                concatenate=False,
                combine=partial(combine, agg=agg, reindex=new_reindex, keepdims=True),
                aggregate=partial(
                    aggregate, expected_groups=cohort_index, reindex=new_reindex, keepdims=True
                ),
            )
            cohort_results.append(cohort_reduced)
            groups_.append(cohort_index.values)

        # Concatenate cohort results along the last axis
        reduced = dask.array.concatenate(cohort_results, axis=-1)
        groups = (np.concatenate(groups_),)
        group_chunks = (tuple(len(cohort) for cohort in groups_),)

    elif method == "blockwise":
        reduced = intermediate
        if reindex.blockwise:
            if TYPE_CHECKING:
                assert expected_groups is not None
            groups = (expected_groups,)
            group_chunks = ((len(expected_groups),),)
        else:
            slices = slices_from_chunks(tuple(array.chunks[ax] for ax in axis))
            from .core import _unique

            groups_in_block = tuple(_unique(by_input[slc]) for slc in slices)
            groups = (np.concatenate(groups_in_block),)
            ngroups_per_block = tuple(len(grp) for grp in groups_in_block)
            group_chunks = (ngroups_per_block,)
    else:
        raise ValueError(f"Unknown method={method}.")

    # Adjust output for any new dimensions added
    new_dims_shape = tuple(dim.size for dim in agg.new_dims if not dim.is_scalar)
    new_inds = tuple(range(-len(new_dims_shape), 0))
    out_inds = new_inds + inds[: -len(axis)] + (inds[-1],)
    output_chunks = new_dims_shape + reduced.chunks[: -len(axis)] + group_chunks
    new_axes = dict(zip(new_inds, new_dims_shape))

    if method == "blockwise" and len(axis) > 1:
        # Use expression class for collapsing blocks
        collapse_expr = CollapseBlocksExpr(reduced.expr, tuple(axis), group_chunks)
        reduced = new_collection(collapse_expr)

    # Extract result - uses expression-aware blockwise
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


def dask_groupby_scan(array, by, axes, agg, method="blelloch", reverse_result=False):
    """Expression-based grouped scan for dask arrays.

    Uses the expression-aware cumreduction from dask.array._array_expr.

    Parameters
    ----------
    reverse_result : bool, optional
        If True, reverse the result (used for bfill). This is handled internally
        to avoid expression optimization issues with slicing scan results.
    """
    import dask.array as da
    from dask.array import map_blocks
    # Use expression-aware cumreduction
    from dask.array._array_expr import cumreduction as scan
    from dask.base import tokenize

    from .aggregations import scan_binary_op
    from .scan import _finalize_scan, chunk_scan, grouped_reduce

    if len(axes) > 1:
        raise NotImplementedError("Scans are only supported along a single axis.")
    (axis,) = axes

    array, by = _unify_chunks(array, by)

    token = tokenize(array, by, agg, axes, method, reverse_result)

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
        scan_func = partial(chunk_scan, agg=agg, axis=axis, dtype=agg.dtype)
        scanned = map_blocks(
            scan_func,
            zipped,
            dtype=agg.dtype,
            meta=array._meta,
            name=f"groupby-scan-{token}",
        )
    else:
        scan_ = partial(chunk_scan, agg=agg)
        scan_.__name__ = scan_.func.__name__

        # Wrap grouped_reduce to accept and ignore dtype argument
        # (expression-based cumreduction passes dtype to preop)
        def preop_wrapper(x, axis=None, keepdims=None, dtype=None):
            return grouped_reduce(x, agg=agg, axis=axis, keepdims=keepdims)

        scanned = scan(
            func=scan_,
            binop=partial(scan_binary_op, agg=agg),
            ident=agg.identity,
            x=zipped,
            axis=axis,
            method="blelloch",
            preop=preop_wrapper,
            dtype=agg.dtype,
        )

    # 3. Extract final result
    # If reverse_result is True, we handle it specially to avoid expression
    # optimization issues (slicing scan results triggers problematic
    # optimizations that push slices into cumreduction).
    if reverse_result:
        # For reverse finalization (bfill), we:
        # 1. Finalize and reverse each block
        # 2. Concatenate blocks in reverse order
        def finalize_and_reverse_block(block, dtype):
            arr = _finalize_scan(block, dtype)
            return arr[..., ::-1]

        # Process each block, then concatenate in reverse order
        chunks = scanned.chunks[-1]
        blocks_finalized = []
        for i in range(len(chunks)):
            block = scanned.blocks[i] if scanned.ndim == 1 else scanned.blocks[..., i]
            finalized = map_blocks(
                partial(finalize_and_reverse_block, dtype=agg.dtype),
                block,
                dtype=agg.dtype,
                name=f"groupby-scan-finalize-block-{i}-{token}",
            )
            blocks_finalized.append(finalized)

        # Concatenate in reverse order
        result = da.concatenate(blocks_finalized[::-1], axis=-1)
        # After reversal, chunks are reversed too
        assert result.chunks[-1] == array.chunks[-1][::-1]
    else:
        result = map_blocks(
            partial(_finalize_scan, dtype=agg.dtype),
            scanned,
            dtype=agg.dtype,
            name=f"groupby-scan-finalize-{token}",
        )
        assert result.chunks == array.chunks

    return result


__all__ = [
    "EXPR_ENABLED",
    "dask_groupby_agg",
    "dask_groupby_scan",
    "ExtractFromDictExpr",
    "CollapseBlocksExpr",
    "SubsetBlocksExpr",
]
