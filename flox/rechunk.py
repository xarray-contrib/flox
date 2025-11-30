"""Rechunking functions for groupby operations.

This module provides functions for rechunking arrays to optimize groupby operations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy_groupies as npg
import pandas as pd

from .aggregations import _atleast_1d
from .cache import memoize
from .factorize import factorize_
from .options import OPTIONS

if TYPE_CHECKING:
    from .core import T_Axis, T_MethodOpt
    from .types import DaskArray

logger = logging.getLogger("flox")


@memoize
def _get_optimal_chunks_for_groups(chunks, labels):
    chunkidx = np.cumsum(chunks) - 1
    # what are the groups at chunk boundaries
    labels_at_chunk_bounds = pd.unique(labels[chunkidx])
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
        f = f.item()  # noqa
        l = l.item()  # noqa
        if Δf < Δl and f > newchunkidx[-1]:
            newchunkidx.append(f)
        else:
            newchunkidx.append(l + 1)
    if newchunkidx[-1] != chunkidx[-1] + 1:
        newchunkidx.append(chunkidx[-1] + 1)
    newchunks = np.diff(newchunkidx)

    assert sum(newchunks) == sum(chunks)
    return tuple(newchunks)


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
    labels : np.ndarray
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

        if (not ignore_old_chunks and idx in oldbreaks) or (counter >= chunksize and not next_break_is_close):
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


def rechunk_for_blockwise(
    array: DaskArray, axis: T_Axis, labels: np.ndarray, *, force: bool = True
) -> tuple[T_MethodOpt, DaskArray]:
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

    chunks = array.chunks[axis]
    if len(chunks) == 1:
        return "blockwise", array

    # import dask
    # from dask.utils import parse_bytes
    # factor = parse_bytes(dask.config.get("array.chunk-size")) / (
    #     math.prod(array.chunksize) * array.dtype.itemsize
    # )
    # if factor > BLOCKWISE_DEFAULT_ARRAY_CHUNK_SIZE_FACTOR:
    #     new_constant_chunks = math.ceil(factor) * max(chunks)
    #     q, r = divmod(array.shape[axis], new_constant_chunks)
    #     new_input_chunks = (new_constant_chunks,) * q + (r,)
    # else:
    new_input_chunks = chunks

    # FIXME: this should be unnecessary?
    labels = factorize_((labels,), axes=())[0]
    newchunks = _get_optimal_chunks_for_groups(new_input_chunks, labels)
    if newchunks == chunks:
        return "blockwise", array

    Δn = abs(len(newchunks) - len(new_input_chunks))
    if pass_num_chunks_threshold := (
        Δn / len(new_input_chunks) < OPTIONS["rechunk_blockwise_num_chunks_threshold"]
    ):
        logger.debug("blockwise rechunk passes num chunks threshold")
    if pass_chunk_size_threshold := (
        # we just pick the max because number of chunks may have changed.
        (abs(max(newchunks) - max(new_input_chunks)) / max(new_input_chunks))
        < OPTIONS["rechunk_blockwise_chunk_size_threshold"]
    ):
        logger.debug("blockwise rechunk passes chunk size change threshold")

    if force or (pass_num_chunks_threshold and pass_chunk_size_threshold):
        logger.debug("Rechunking to enable blockwise.")
        return "blockwise", array.rechunk({axis: newchunks})
    else:
        logger.debug("Didn't meet thresholds to do automatic rechunking for blockwise reductions.")
        return None, array


__all__ = [
    "rechunk_for_blockwise",
    "rechunk_for_cohorts",
]
