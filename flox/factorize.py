"""Factorization functions for groupby operations.

This module provides functions for factorizing groupby labels.
"""

from __future__ import annotations

import itertools
import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import pandas as pd

from .types import FactorizeKwargs, FactorProps
from .xrutils import is_duck_dask_array, isnull

if TYPE_CHECKING:
    from .core import T_Axes, T_By, T_Bys, T_ExpectIndexOptTuple


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


def _factorize_single(by, expect, *, sort: bool, reindex: bool) -> tuple[pd.Index, np.ndarray]:
    flat = by.reshape(-1)
    if isinstance(expect, pd.RangeIndex):
        # idx is a view of the original `by` array
        # copy here so we don't have a race condition with the
        # group_idx[nanmask] = nan_sentinel assignment later
        # this is important in shared-memory parallelism with dask
        # TODO: figure out how to avoid this
        idx = flat.copy()
        found_groups = cast(pd.Index, expect)
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
        found_groups = cast(pd.Index, expect)
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
            idx, groups = pd.factorize(flat, sort=sort)
        found_groups = cast(pd.Index, groups)

    return (found_groups, idx.reshape(by.shape))


def _ravel_factorized(*factorized: np.ndarray, grp_shape: tuple[int, ...]) -> np.ndarray:
    group_idx = np.ravel_multi_index(factorized, grp_shape, mode="wrap")
    # NaNs; as well as values outside the bins are coded by -1
    # Restore these after the raveling
    nan_by_mask = reduce(np.logical_or, [(f == -1) for f in factorized])
    group_idx[nan_by_mask] = -1
    return group_idx


@overload
def factorize_(
    by: T_Bys,
    axes: T_Axes,
    *,
    fastpath: Literal[True],
    expected_groups: T_ExpectIndexOptTuple | None = None,
    reindex: bool = False,
    sort: bool = True,
) -> tuple[np.ndarray, tuple[pd.Index, ...], tuple[int, ...], int, int, None]: ...


@overload
def factorize_(
    by: T_Bys,
    axes: T_Axes,
    *,
    expected_groups: T_ExpectIndexOptTuple | None = None,
    reindex: bool = False,
    sort: bool = True,
    fastpath: Literal[False] = False,
) -> tuple[np.ndarray, tuple[pd.Index, ...], tuple[int, ...], int, int, FactorProps]: ...


@overload
def factorize_(
    by: T_Bys,
    axes: T_Axes,
    *,
    expected_groups: T_ExpectIndexOptTuple | None = None,
    reindex: bool = False,
    sort: bool = True,
    fastpath: bool = False,
) -> tuple[np.ndarray, tuple[pd.Index, ...], tuple[int, ...], int, int, FactorProps | None]: ...


def factorize_(
    by: T_Bys,
    axes: T_Axes,
    *,
    expected_groups: T_ExpectIndexOptTuple | None = None,
    reindex: bool = False,
    sort: bool = True,
    fastpath: bool = False,
) -> tuple[np.ndarray, tuple[pd.Index, ...], tuple[int, ...], int, int, FactorProps | None]:
    """
    Returns an array of integer codes for groups (and associated data)
    by wrapping pd.cut and pd.factorize (depending on isbin).
    This method handles reindex and sort so that we don't spend time reindexing / sorting
    a possibly large results array. Instead we set up the appropriate integer codes (group_idx)
    so that the results come out in the appropriate order.
    """
    if expected_groups is None:
        expected_groups = (None,) * len(by)

    if len(by) > 2:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(partial(_factorize_single, sort=sort, reindex=reindex), groupvar, expect)
                for groupvar, expect in zip(by, expected_groups)
            ]
            results = tuple(f.result() for f in futures)
    else:
        results = tuple(
            _factorize_single(groupvar, expect, sort=sort, reindex=reindex)
            for groupvar, expect in zip(by, expected_groups)
        )
    found_groups = tuple(r[0] for r in results)
    factorized = [r[1] for r in results]

    grp_shape = tuple(len(grp) for grp in found_groups)
    ngroups = math.prod(grp_shape)
    if len(by) > 1:
        group_idx = _ravel_factorized(*factorized, grp_shape=grp_shape)
    else:
        (group_idx,) = factorized

    if fastpath:
        return group_idx, found_groups, grp_shape, ngroups, ngroups, None

    if len(axes) == 1 and by[0].ndim > 1:
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


def _lazy_factorize_wrapper(*by: T_By, **kwargs) -> np.ndarray:
    group_idx, *_ = factorize_(by, **kwargs)
    return group_idx


def _factorize_multiple(
    by: T_Bys,
    expected_groups: T_ExpectIndexOptTuple,
    any_by_dask: bool,
    sort: bool = True,
) -> tuple[tuple[np.ndarray], tuple[pd.Index, ...], tuple[int, ...]]:
    kwargs: FactorizeKwargs = dict(
        axes=(),  # always (), we offset later if necessary.
        fastpath=True,
        # This is the only way it makes sense I think.
        # reindex controls what's actually allocated in chunk_reduce
        # At this point, we care about an accurate conversion to codes.
        reindex=True,
        sort=sort,
    )
    if any_by_dask:
        import dask.array

        from . import dask_array_ops  # noqa

        # unifying chunks will make sure all arrays in `by` are dask arrays
        # with compatible chunks, even if there was originally a numpy array
        inds = tuple(range(by[0].ndim))
        for by_, expect in zip(by, expected_groups):
            if expect is None and is_duck_dask_array(by_):
                raise ValueError("Please provide expected_groups when grouping by a dask array.")

        found_groups = tuple(
            pd.Index(pd.unique(by_.reshape(-1))) if expect is None else expect
            for by_, expect in zip(by, expected_groups)
        )
        grp_shape = tuple(map(len, found_groups))

        chunks, by_chunked = dask.array.unify_chunks(*itertools.chain(*zip(by, (inds,) * len(by))))
        group_idxs = [
            dask.array.map_blocks(
                _lazy_factorize_wrapper,
                by_,
                expected_groups=(expect_,),
                meta=np.array((), dtype=np.int64),
                **kwargs,
            )
            for by_, expect_ in zip(by_chunked, expected_groups)
        ]
        # This could be avoied but we'd use `np.where`
        # instead `_ravel_factorized` instead i.e. a copy.
        group_idx = dask.array.map_blocks(
            _ravel_factorized, *group_idxs, grp_shape=grp_shape, chunks=tuple(chunks.values()), dtype=np.int64
        )

    else:
        kwargs["by"] = by
        group_idx, found_groups, grp_shape, *_ = factorize_(**kwargs, expected_groups=expected_groups)

    return (group_idx,), found_groups, grp_shape


__all__ = [
    "FactorizeKwargs",
    "FactorProps",
    "_factorize_multiple",
    "_factorize_single",
    "_lazy_factorize_wrapper",
    "_ravel_factorized",
    "factorize_",
    "offset_labels",
]
