"""dask-array expression integration for flox.

This module is intentionally narrow: it mirrors the existing ``flox.dask``
groupby construction, but uses dask-array's expression primitives when the
input is a standalone ``dask_array.Array``.
"""

from __future__ import annotations

import itertools
import operator
from collections.abc import Sequence
from functools import cached_property, partial
from numbers import Integral
from typing import TYPE_CHECKING, Any

import dask
import dask_array as da
import numpy as np
import pandas as pd
import toolz as tlz
from dask._task_spec import Alias, Task, TaskRef
from dask_array._collection import Array
from dask_array._expr import ArrayExpr
from dask_array._new_collection import new_collection
from dask_array.reductions._reduction import _tree_reduce

from .core import _get_chunk_reduction, _reduce_blockwise
from .dask import (
    _aggregate,
    _expand_dims,
    _extract_result,
    _grouped_combine,
    _normalize_indexes,
    _simple_combine,
    reindex_intermediates,
)
from .lib import _is_arg_reduction, _is_first_last_reduction, identity
from .reindex import ReindexStrategy
from .xrutils import is_duck_dask_array

if TYPE_CHECKING:
    from .aggregations import Aggregation
    from .core import T_Axes, T_Engine, T_Method


def is_dask_array(x: Any) -> bool:
    return isinstance(x, Array)


def contains_dask_array(*args: Any) -> bool:
    return any(is_dask_array(arg) for arg in args)


class _DirectChunkManager:
    @property
    def array_api(self):
        return da

    def from_array(self, data, chunks, **kwargs):
        return da.from_array(data, chunks=chunks, **kwargs)

    def map_blocks(self, func, *args, **kwargs):
        return da.map_blocks(func, *args, **kwargs)

    def blockwise(self, func, out_ind, *args, **kwargs):
        return da.blockwise(func, out_ind, *args, **kwargs)

    def unify_chunks(self, *args, **kwargs):
        return da.unify_chunks(*args, **kwargs)


_DIRECT_CHUNKMANAGER = _DirectChunkManager()


def get_chunkmanager(*args: Any):
    chunked_args = tuple(arg for arg in args if is_dask_array(arg))
    if not chunked_args:
        return _DIRECT_CHUNKMANAGER

    try:
        from xarray.namedarray.parallelcompat import get_chunked_array_type
    except ImportError:
        return _DIRECT_CHUNKMANAGER

    try:
        return get_chunked_array_type(*chunked_args)
    except TypeError:
        return _DIRECT_CHUNKMANAGER


class FloxExtractGroups(ArrayExpr):
    _parameters = ["array", "key", "_dtype"]

    @cached_property
    def dtype(self):
        return np.dtype(self.operand("_dtype"))

    @cached_property
    def chunks(self):
        return ((np.nan,),)

    @cached_property
    def _meta(self):
        return np.array([], dtype=self.dtype)

    def _layer(self):
        first_block = self.array.ndim * (0,)
        in_key = (self.array.name, *first_block)
        out_key = (self._name, 0)
        return {out_key: Task(out_key, operator.getitem, TaskRef(in_key), self.key)}


class FloxCollapseBlocks(ArrayExpr):
    _parameters = ["array", "axis", "group_chunks"]

    @cached_property
    def dtype(self):
        return self.array.dtype

    @cached_property
    def chunks(self):
        axis = self.axis
        return self.array.chunks[: -len(axis)] + ((1,) * (len(axis) - 1),) + self.group_chunks

    @cached_property
    def _meta(self):
        return self.array._meta

    def _layer(self):
        axis = self.axis
        nblocks = tuple(self.array.numblocks[ax] for ax in axis)
        layer = {}
        for out_block in itertools.product(*(range(len(chunks)) for chunks in self.chunks)):
            in_block = out_block[: -len(axis)] + np.unravel_index(out_block[-1], nblocks)
            out_key = (self._name, *out_block)
            in_key = (self.array.name, *in_block)
            layer[out_key] = Alias(out_key, in_key)
        return layer


class FloxSubsetBlocks(ArrayExpr):
    _parameters = ["array", "flatblocks", "blkshape", "reindexer", "output_chunks"]

    @cached_property
    def dtype(self):
        return self.array.dtype

    @cached_property
    def chunks(self):
        return self.output_chunks

    @cached_property
    def _meta(self):
        return self.array._meta

    def _layer(self):
        index = _normalize_indexes(self.array.ndim, self.flatblocks, self.blkshape)
        index = tuple(slice(k, k + 1) if isinstance(k, Integral) else k for k in index)

        old_keys = np.empty(self.array.numblocks, dtype=object)
        for block in itertools.product(*(range(n) for n in self.array.numblocks)):
            old_keys[block] = (self.array.name, *block)

        selected_keys = old_keys[index]
        layer = {}
        for out_block in itertools.product(*(range(len(chunks)) for chunks in self.output_chunks)):
            in_key = selected_keys[out_block]
            if isinstance(in_key, np.ndarray):
                in_key = tuple(in_key.flat[0])
            out_key = (self._name, *out_block)
            layer[out_key] = Task(out_key, self.reindexer, TaskRef(in_key))
        return layer


def _unify_chunks(array, by, chunkmanager):
    inds = tuple(range(array.ndim))

    if not is_dask_array(by):
        if is_duck_dask_array(by):
            raise TypeError("Cannot mix dask_array.Array with other Dask-backed array types.")
        chunks = tuple(array.chunks[ax] if by.shape[ax] != 1 else (1,) for ax in range(-by.ndim, 0))
        by = chunkmanager.from_array(by, chunks=chunks)

    _, (array, by) = chunkmanager.unify_chunks(array, inds, by, inds[-by.ndim :])
    return array, by


def _argreduce_preprocess(array, axis, chunkmanager):
    assert len(axis) == 1
    axis = axis[0]

    idx = chunkmanager.array_api.arange(array.shape[axis], chunks=array.chunks[axis], dtype=np.intp)
    idx = idx[tuple(slice(None) if i == axis else np.newaxis for i in range(array.ndim))]

    def _zip_index(array_, idx_):
        return (array_, idx_)

    return chunkmanager.map_blocks(
        _zip_index,
        array,
        idx,
        dtype=array.dtype,
        meta=array._meta,
        name="groupby-argreduce-preprocess",
    )


def dask_groupby_agg(
    array: Array,
    by: Any,
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
) -> tuple[Array, tuple[pd.Index | np.ndarray | Array]]:
    from dask.array.core import slices_from_chunks

    assert isinstance(axis, Sequence)
    assert all(ax >= 0 for ax in axis)
    if not is_dask_array(array):
        raise TypeError("Cannot mix dask_array.Array with other Dask-backed array types.")

    chunkmanager = get_chunkmanager(array, by)
    array_api = chunkmanager.array_api

    inds = tuple(range(array.ndim))
    name = f"groupby_{agg.name}"

    if expected_groups is None and reindex.blockwise:
        raise ValueError("reindex.blockwise must be False-y if expected_groups is not provided.")
    if method == "cohorts" and reindex.blockwise:
        raise ValueError("reindex.blockwise must be False-y if method is 'cohorts'.")

    by_input = by
    array = new_collection(array.expr.simplify())
    if is_dask_array(by):
        by = new_collection(by.expr.simplify())
    array, by = _unify_chunks(array, by, chunkmanager)

    token = dask.base.tokenize(array, by, agg, expected_groups, axis, method)

    if agg.preprocess and method != "blockwise":
        if _is_arg_reduction(agg):
            array = _argreduce_preprocess(array, axis=axis, chunkmanager=chunkmanager)
        else:
            array = agg.preprocess(array, axis=axis)

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

    intermediate = chunkmanager.blockwise(
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

    if method in ["map-reduce", "cohorts"]:
        combine = (
            partial(_simple_combine, reindex=reindex)
            if do_simple_combine
            else partial(_grouped_combine, engine=engine, sort=sort)
        )
        aggregate = partial(_aggregate, combine=combine, agg=agg, fill_value=fill_value, reindex=reindex)

        if method == "map-reduce":
            reduced = _tree_reduce(
                intermediate.expr,
                aggregate=partial(aggregate, expected_groups=expected_groups),
                axis=axis,
                keepdims=True,
                dtype=array.dtype,
                combine=partial(combine, agg=agg),
                name=f"{name}-simple-reduce",
                concatenate=False,
            )
            if labels_are_unknown:
                groups = (new_collection(FloxExtractGroups(reduced.expr, "groups", by.dtype)),)
                group_chunks = ((np.nan,),)
            else:
                assert expected_groups is not None
                groups = (expected_groups,)
                group_chunks = ((len(expected_groups),),)

        else:
            assert chunks_cohorts
            block_shape = intermediate.blocks.shape[-len(axis) :]
            chunks_as_array = tuple(np.array(c) for c in intermediate.chunks)

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

                index = _normalize_indexes(intermediate.ndim, blks, block_shape)
                index = tuple(slice(k, k + 1) if isinstance(k, Integral) else k for k in index)
                squeezed = tuple(np.squeeze(i) if isinstance(i, np.ndarray) else i for i in index)
                subset_chunks = tuple(tuple(c[i].tolist()) for c, i in zip(chunks_as_array, squeezed))

                subset = new_collection(
                    FloxSubsetBlocks(intermediate.expr, tuple(blks), block_shape, reindexer, subset_chunks)
                )
                new_reindex = ReindexStrategy(blockwise=do_simple_combine, array_type=reindex.array_type)
                cohort_results.append(
                    _tree_reduce(
                        subset.expr,
                        aggregate=partial(
                            aggregate,
                            expected_groups=cohort_index,
                            reindex=new_reindex,
                            keepdims=True,
                        ),
                        axis=axis,
                        keepdims=True,
                        dtype=array.dtype,
                        combine=partial(combine, agg=agg, reindex=new_reindex, keepdims=True),
                        name=f"{name}-cohort-{icohort}-{token}",
                        concatenate=False,
                    )
                )
                groups_.append(cohort_index.values)

            reduced = array_api.concatenate(cohort_results, axis=-1)
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
            group_chunks = (tuple(len(grp) for grp in groups_in_block),)
    else:
        raise ValueError(f"Unknown method={method}.")

    new_dims_shape = tuple(dim.size for dim in agg.new_dims if not dim.is_scalar)
    new_inds = tuple(range(-len(new_dims_shape), 0))
    out_inds = new_inds + inds[: -len(axis)] + (inds[-1],)
    output_chunks = new_dims_shape + reduced.chunks[: -len(axis)] + group_chunks
    new_axes = dict(zip(new_inds, new_dims_shape))

    if method == "blockwise" and len(axis) > 1:
        reduced = new_collection(FloxCollapseBlocks(reduced.expr, tuple(axis), group_chunks))

    result = chunkmanager.blockwise(
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

    return result, groups


__all__ = [
    "FloxCollapseBlocks",
    "FloxExtractGroups",
    "FloxSubsetBlocks",
    "contains_dask_array",
    "dask_groupby_agg",
    "get_chunkmanager",
    "is_dask_array",
]
