"""Cubed-specific functions for groupby operations.

This module provides Cubed-specific implementations for groupby operations.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .aggregations import Aggregation
    from .core import T_Axes, T_Engine, T_Method
    from .types import CubedArray, T_By

from .core import (
    _finalize_results,
    _get_chunk_reduction,
    _is_arg_reduction,
    _reduce_blockwise,
)
from .reindex import ReindexStrategy
from .xrutils import is_chunked_array


def cubed_groupby_agg(
    array: CubedArray,
    by: T_By,
    agg: Aggregation,
    expected_groups: pd.Index | None,
    reindex: ReindexStrategy,
    axis: T_Axes = (),
    fill_value: Any = None,
    method: T_Method = "map-reduce",
    engine: T_Engine = "numpy",
    sort: bool = True,
    chunks_cohorts=None,
) -> tuple[CubedArray, tuple[pd.Index | np.ndarray | CubedArray]]:
    import cubed
    import cubed.core.groupby

    # I think _tree_reduce expects this
    assert isinstance(axis, Sequence)
    assert all(ax >= 0 for ax in axis)

    if method == "blockwise":
        assert by.ndim == 1
        assert expected_groups is not None

        def _reduction_func(a, by, axis, start_group, num_groups):
            # adjust group labels to start from 0 for each chunk
            by_for_chunk = by - start_group
            expected_groups_for_chunk = pd.RangeIndex(num_groups)

            axis = (axis,)  # convert integral axis to tuple

            blockwise_method = partial(
                _reduce_blockwise,
                agg=agg,
                axis=axis,
                expected_groups=expected_groups_for_chunk,
                fill_value=fill_value,
                engine=engine,
                sort=sort,
                reindex=reindex,
            )
            out = blockwise_method(a, by_for_chunk)
            return out[agg.name]

        num_groups = len(expected_groups)
        result = cubed.core.groupby.groupby_blockwise(
            array, by, axis=axis, func=_reduction_func, num_groups=num_groups
        )
        groups = (expected_groups,)
        return (result, groups)

    else:
        inds = tuple(range(array.ndim))

        by_input = by

        # Unifying chunks is necessary for argreductions.
        # We need to rechunk before zipping up with the index
        # let's always do it anyway
        if not is_chunked_array(by):
            # chunk numpy arrays like the input array
            chunks = tuple(array.chunks[ax] if by.shape[ax] != 1 else (1,) for ax in range(-by.ndim, 0))

            by = cubed.from_array(by, chunks=chunks, spec=array.spec)
        _, (array, by) = cubed.core.unify_chunks(array, inds, by, inds[-by.ndim :])

        # Cubed's groupby_reduction handles the generation of "intermediates", and the
        # "map-reduce" combination step, so we don't have to do that here.
        # Only the equivalent of "_simple_combine" is supported, there is no
        # support for "_grouped_combine".
        labels_are_unknown = is_chunked_array(by_input) and expected_groups is None
        do_simple_combine = not _is_arg_reduction(agg) and not labels_are_unknown

        assert do_simple_combine
        assert method == "map-reduce"
        assert expected_groups is not None
        assert reindex.blockwise is True
        assert len(axis) == 1  # one axis/grouping

        def _groupby_func(a, by, axis, intermediate_dtype, num_groups):
            blockwise_method = partial(
                _get_chunk_reduction(agg.reduction_type),
                func=agg.chunk,
                fill_value=agg.fill_value["intermediate"],
                dtype=agg.dtype["intermediate"],
                reindex=reindex,
                user_dtype=agg.dtype["user"],
                axis=axis,
                expected_groups=expected_groups,
                engine=engine,
                sort=sort,
            )
            out = blockwise_method(a, by)
            # Convert dict to one that cubed understands, dropping groups since they are
            # known, and the same for every block.
            return {f"f{idx}": intermediate for idx, intermediate in enumerate(out["intermediates"])}

        def _groupby_combine(a, axis, dummy_axis, dtype, keepdims):
            # this is similar to _simple_combine, except the dummy axis and concatenation is handled by cubed
            # only combine over the dummy axis, to preserve grouping along 'axis'
            dtype = dict(dtype)
            out = {}
            for idx, combine in enumerate(agg.simple_combine):
                field = f"f{idx}"
                out[field] = combine(a[field], axis=dummy_axis, keepdims=keepdims)
            return out

        def _groupby_aggregate(a, **kwargs):
            # Convert cubed dict to one that _finalize_results works with
            results = {"groups": expected_groups, "intermediates": a.values()}
            out = _finalize_results(results, agg, axis, expected_groups, reindex)
            return out[agg.name]

        # convert list of dtypes to a structured dtype for cubed
        intermediate_dtype = [(f"f{i}", dtype) for i, dtype in enumerate(agg.dtype["intermediate"])]
        dtype = agg.dtype["final"]
        num_groups = len(expected_groups)

        result = cubed.core.groupby.groupby_reduction(
            array,
            by,
            func=_groupby_func,
            combine_func=_groupby_combine,
            aggregate_func=_groupby_aggregate,
            axis=axis,
            intermediate_dtype=intermediate_dtype,
            dtype=dtype,
            num_groups=num_groups,
        )

        groups = (expected_groups,)

        return (result, groups)


__all__ = [
    "cubed_groupby_agg",
]
