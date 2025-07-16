"""Reindexing functions for groupby operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .core import T_Axis

from . import xrdtypes
from .lib import sparse_array_type
from .utils import ReindexArrayType
from .xrutils import isnull


def reindex_numpy(array, from_: pd.Index, to: pd.Index, fill_value, dtype, axis: int):
    idx = from_.get_indexer(to)
    indexer = [slice(None, None)] * array.ndim
    indexer[axis] = idx
    reindexed = array[tuple(indexer)]
    if (idx == -1).any():
        if fill_value is None:
            raise ValueError("Filling is required. fill_value cannot be None.")
        indexer[axis] = idx == -1
        reindexed = reindexed.astype(dtype, copy=False)
        reindexed[tuple(indexer)] = fill_value
    return reindexed


def reindex_pydata_sparse_coo(array, from_: pd.Index, to: pd.Index, fill_value, dtype, axis: int):
    import sparse

    assert axis == -1

    # Are there any elements in `to` that are not in `from_`.
    if isinstance(to, pd.RangeIndex) and len(to) > len(from_):
        # 1. pandas optimizes set difference between two RangeIndexes only
        # 2. We want to avoid realizing a very large numpy array in to memory.
        #    This happens in the `else` clause.
        #    There are potentially other tricks we can play, but this is a simple
        #    and effective one. If a user is reindexing to sparse, then len(to) is
        #    almost guaranteed to be > len(from_). If len(to) <= len(from_), then realizing
        #    another array of the same shape should be fine.
        needs_reindex = True
    else:
        needs_reindex = (from_.get_indexer(to) == -1).any()

    if needs_reindex and fill_value is None:
        raise ValueError("Filling is required. fill_value cannot be None.")

    idx = to.get_indexer(from_)
    mask = idx != -1  # indices along last axis to keep
    if mask.all():
        mask = slice(None)
    shape = array.shape

    if isinstance(array, sparse.COO):
        subset = array[..., mask]
        data = subset.data
        coords = subset.coords
        if subset.nnz > 0:
            coords[-1, :] = idx[mask][coords[-1, :]]
        if fill_value is None:
            # no reindexing is actually needed (dense case)
            # preserve the fill_value
            fill_value = array.fill_value
    else:
        ranges = np.broadcast_arrays(
            *np.ix_(*(tuple(np.arange(size) for size in shape[:axis]) + (idx[mask],)))
        )
        coords = np.stack(ranges, axis=0).reshape(array.ndim, -1)
        data = array[..., mask].reshape(-1)

    reindexed = sparse.COO(
        coords=coords,
        data=data.astype(dtype, copy=False),
        shape=(*array.shape[:axis], to.size),
        fill_value=fill_value,
    )

    return reindexed


def reindex_(
    array: np.ndarray,
    from_,
    to,
    *,
    array_type: ReindexArrayType = ReindexArrayType.AUTO,
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
        shape = array.shape[:-1] + (len(to),)
        if array_type in (ReindexArrayType.AUTO, ReindexArrayType.NUMPY):
            reindexed = np.full(shape, fill_value, dtype=array.dtype)
        else:
            raise NotImplementedError
        return reindexed

    from_ = pd.Index(from_)
    # short-circuit for trivial case
    if from_.equals(to) and array_type.is_same_type(array):
        return array

    if from_.dtype.kind == "O" and isinstance(from_[0], tuple):
        raise NotImplementedError(
            "Currently does not support reindexing with object arrays of tuples. "
            "These occur when grouping by multi-indexed variables in xarray."
        )
    if fill_value is xrdtypes.NA or isnull(fill_value):
        new_dtype, fill_value = xrdtypes.maybe_promote(array.dtype)
    else:
        new_dtype = array.dtype

    if array_type is ReindexArrayType.AUTO:
        if isinstance(array, sparse_array_type):
            array_type = ReindexArrayType.SPARSE_COO
        else:
            # TODO: generalize here
            # Right now, we effectively assume NEP-18 I think
            array_type = ReindexArrayType.NUMPY

    if array_type is ReindexArrayType.NUMPY:
        reindexed = reindex_numpy(array, from_, to, fill_value, new_dtype, axis)
    elif array_type is ReindexArrayType.SPARSE_COO:
        reindexed = reindex_pydata_sparse_coo(array, from_, to, fill_value, new_dtype, axis)
    return reindexed


__all__ = [
    "reindex_",
    "reindex_numpy",
    "reindex_pydata_sparse_coo",
]
