"""Functions for finding group cohorts in chunked arrays."""

from __future__ import annotations

import itertools
import logging
import math
import operator
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import TYPE_CHECKING, Literal

import numpy as np
import toolz as tlz
from scipy.sparse import csc_array, csr_array

if TYPE_CHECKING:
    import pandas as pd

    from .core import T_Method

logger = logging.getLogger("flox")


def slices_from_chunks(chunks):
    """slightly modified from dask.array.core.slices_from_chunks to be lazy"""
    cumdims = [tlz.accumulate(operator.add, bds, 0) for bds in chunks]
    slices = (
        (slice(s, s + dim) for s, dim in zip(starts, shapes)) for starts, shapes in zip(cumdims, chunks)
    )
    return product(*slices)


def _compute_label_chunk_bitmask(labels, chunks, nlabels):
    def make_bitmask(rows, cols):
        data = np.broadcast_to(np.array(1, dtype=np.uint8), rows.shape)
        return csc_array((data, (rows, cols)), dtype=bool, shape=(nchunks, nlabels))

    assert isinstance(labels, np.ndarray)
    shape = tuple(sum(c) for c in chunks)
    nchunks = math.prod(len(c) for c in chunks)
    approx_chunk_size = math.prod(c[0] for c in chunks)

    # Shortcut for 1D with size-1 chunks
    if shape == (nchunks,):
        rows_array = np.arange(nchunks)
        cols_array = labels
        mask = labels >= 0
        return make_bitmask(rows_array[mask], cols_array[mask])

    labels = np.broadcast_to(labels, shape[-labels.ndim :])
    cols = []
    ilabels = np.arange(nlabels)

    def chunk_unique(labels, slicer, nlabels, label_is_present=None):
        if label_is_present is None:
            label_is_present = np.empty((nlabels + 1,), dtype=bool)
        label_is_present[:] = False
        subset = labels[slicer]
        # This is a quite fast way to find unique integers, when we know how many there are
        # inspired by a similar idea in numpy_groupies for first, last
        # instead of explicitly finding uniques, repeatedly write True to the same location
        label_is_present[subset.reshape(-1)] = True
        # skip the -1 sentinel by slicing
        # Faster than np.argwhere by a lot
        uniques = ilabels[label_is_present[:-1]]
        return uniques

    # TODO: refine this heuristic.
    # The general idea is that with the threadpool, we repeatedly allocate memory
    # for `label_is_present`. We trade that off against the parallelism across number of chunks.
    # For large enough number of chunks (relative to number of labels), it makes sense to
    # suffer the extra allocation in exchange for parallelism.
    THRESHOLD = 2
    if nlabels < THRESHOLD * approx_chunk_size:
        logger.debug(
            "Using threadpool since num_labels %s < %d * chunksize %s",
            nlabels,
            THRESHOLD,
            approx_chunk_size,
        )
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(chunk_unique, labels, slicer, nlabels)
                for slicer in slices_from_chunks(chunks)
            ]
            cols = tuple(f.result() for f in futures)

    else:
        logger.debug(
            "Using serial loop since num_labels %s > %d * chunksize %s",
            nlabels,
            THRESHOLD,
            approx_chunk_size,
        )
        cols = []
        # Add one to handle the -1 sentinel value
        label_is_present = np.empty((nlabels + 1,), dtype=bool)
        for region in slices_from_chunks(chunks):
            uniques = chunk_unique(labels, region, nlabels, label_is_present)
            cols.append(uniques)
    rows_array = np.repeat(np.arange(nchunks), tuple(len(col) for col in cols))
    cols_array = np.concatenate(cols)

    return make_bitmask(rows_array, cols_array)


# @memoize
def find_group_cohorts(
    labels, chunks, expected_groups: pd.RangeIndex | None = None, merge: bool = False
) -> tuple[T_Method, dict]:
    """
    Finds groups labels that occur together aka "cohorts"

    If available, results are cached in a 1MB cache managed by `cachey`.
    This allows us to be quick when repeatedly calling groupby_reduce
    for arrays with the same chunking (e.g. an xarray Dataset).

    Parameters
    ----------
    labels : np.ndarray
        mD Array of integer group codes, factorized so that -1
        represents NaNs.
    chunks : tuple
        chunks of the array being reduced
    expected_groups: pd.RangeIndex (optional)
        Used to extract the largest label expected
    merge: bool (optional)
        Whether to merge cohorts or not. Set to True if a user
        specifies "cohorts" but other methods are preferable.

    Returns
    -------
    preferred_method: {"blockwise", cohorts", "map-reduce"}
    cohorts: dict_values
        Iterable of cohorts
    """
    # To do this, we must have values in memory so casting to numpy should be safe
    labels = np.asarray(labels)

    shape = tuple(sum(c) for c in chunks)
    nchunks = math.prod(len(c) for c in chunks)

    # assumes that `labels` are factorized
    if expected_groups is None:
        nlabels = labels.max() + 1
    else:
        nlabels = expected_groups[-1] + 1

    # 1. Single chunk, blockwise always
    if nchunks == 1:
        return "blockwise", {(0,): list(range(nlabels))}

    labels = np.broadcast_to(labels, shape[-labels.ndim :])
    bitmask = _compute_label_chunk_bitmask(labels, chunks, nlabels)

    CHUNK_AXIS, LABEL_AXIS = 0, 1
    chunks_per_label = bitmask.sum(axis=CHUNK_AXIS)

    # can happen when `expected_groups` is passed but not all labels are present
    # (binning, resampling)
    present_labels = np.arange(bitmask.shape[LABEL_AXIS])
    present_labels_mask = chunks_per_label != 0
    if not present_labels_mask.all():
        present_labels = present_labels[present_labels_mask]
        bitmask = bitmask[..., present_labels_mask]
        chunks_per_label = chunks_per_label[present_labels_mask]

    label_chunks = {
        present_labels[idx].item(): bitmask.indices[slice(bitmask.indptr[idx], bitmask.indptr[idx + 1])]
        for idx in range(bitmask.shape[LABEL_AXIS])
    }

    # Invert the label_chunks mapping so we know which labels occur together.
    def invert(x) -> tuple[np.ndarray, ...]:
        arr = label_chunks[x]
        return tuple(arr.tolist())

    chunks_cohorts = tlz.groupby(invert, label_chunks.keys())

    # 2. Every group is contained to one block, use blockwise here.
    if bitmask.shape[CHUNK_AXIS] == 1 or (chunks_per_label == 1).all():
        logger.debug("find_group_cohorts: blockwise is preferred.")
        return "blockwise", chunks_cohorts

    # 3. Perfectly chunked so there is only a single cohort
    if len(chunks_cohorts) == 1:
        logger.debug("Only found a single cohort. 'map-reduce' is preferred.")
        return "map-reduce", chunks_cohorts if merge else {}

    # 4. Our dataset has chunksize one along the axis,
    single_chunks = all(all(a == 1 for a in ac) for ac in chunks)
    # 5. Every chunk only has a single group, but that group might extend across multiple chunks
    one_group_per_chunk = (bitmask.sum(axis=LABEL_AXIS) == 1).all()
    # 6. Existing cohorts don't overlap, great for time grouping with perfect chunking
    no_overlapping_cohorts = (np.bincount(np.concatenate(tuple(chunks_cohorts.keys()))) == 1).all()
    if one_group_per_chunk or single_chunks or no_overlapping_cohorts:
        logger.debug("find_group_cohorts: cohorts is preferred, chunking is perfect.")
        return "cohorts", chunks_cohorts

    # We'll use containment to measure degree of overlap between labels.
    # Containment C = |Q & S| / |Q|
    #  - |X| is the cardinality of set X
    #  - Q is the query set being tested
    #  - S is the existing set
    # The bitmask matrix S allows us to calculate this pretty efficiently using a dot product.
    #       S.T @ S / chunks_per_label
    #
    # We treat the sparsity(C) = (nnz/size) as a summary measure of the net overlap.
    # 1. For high enough sparsity, there is a lot of overlap and we should use "map-reduce".
    # 2. When labels are uniformly distributed amongst all chunks
    #    (and number of labels < chunk size), sparsity is 1.
    # 3. Time grouping cohorts (e.g. dayofyear) appear as lines in this matrix.
    # 4. When there are no overlaps at all between labels, containment is a block diagonal matrix
    #    (approximately).
    #
    # However computing S.T @ S can still be the slowest step, especially if S
    # is not particularly sparse. Empirically the sparsity( S.T @ S ) > min(1, 2 x sparsity(S)).
    # So we use sparsity(S) as a shortcut.
    MAX_SPARSITY_FOR_COHORTS = 0.4  # arbitrary
    sparsity = bitmask.nnz / math.prod(bitmask.shape)
    preferred_method: Literal["map-reduce"] | Literal["cohorts"]
    logger.debug(
        "sparsity of bitmask is {}, threshold is {}".format(  # noqa
            sparsity, MAX_SPARSITY_FOR_COHORTS
        )
    )
    # 7. Groups seem fairly randomly distributed, use "map-reduce".
    if sparsity > MAX_SPARSITY_FOR_COHORTS:
        if not merge:
            logger.debug(
                "find_group_cohorts: bitmask sparsity={}, merge=False, choosing 'map-reduce'".format(  # noqa
                    sparsity
                )
            )
            return "map-reduce", {}
        preferred_method = "map-reduce"
    else:
        preferred_method = "cohorts"

    # Note: While A.T @ A is a symmetric matrix, the division by chunks_per_label
    #       makes it non-symmetric.
    asfloat = bitmask.astype(float)
    containment = csr_array(asfloat.T @ asfloat / chunks_per_label)

    logger.debug(
        "sparsity of containment matrix is {}".format(  # noqa
            containment.nnz / math.prod(containment.shape)
        )
    )

    # Next we for-loop over groups and merge those that are quite similar.
    # Use a threshold on containment to always force some merging.
    # Note that we do not use the filtered containment matrix for estimating "sparsity"
    # because it is a bit hard to reason about.
    MIN_CONTAINMENT = 0.75  # arbitrary
    mask = containment.data < MIN_CONTAINMENT

    # Now we also know "exact cohorts" -- cohorts whose constituent groups
    # occur in exactly the same chunks. We only need examine one member of each group.
    # Skip the others by first looping over the exact cohorts, and zero out those rows.
    repeated = np.concatenate([v[1:] for v in chunks_cohorts.values()]).astype(int)
    repeated_idx = np.searchsorted(present_labels, repeated)
    for i in repeated_idx:
        mask[containment.indptr[i] : containment.indptr[i + 1]] = True
    containment.data[mask] = 0
    containment.eliminate_zeros()

    # Figure out all the labels we need to loop over later
    n_overlapping_labels = containment.astype(bool).sum(axis=1)
    order = np.argsort(n_overlapping_labels, kind="stable")[::-1]
    # Order is such that we iterate over labels, beginning with those with most overlaps
    # Also filter out any "exact" cohorts
    order = order[n_overlapping_labels[order] > 0]

    logger.debug("find_group_cohorts: merging cohorts")
    merged_cohorts = {}
    merged_keys = set()
    for rowidx in order:
        if present_labels[rowidx] in merged_keys:
            continue
        cohidx = containment.indices[slice(containment.indptr[rowidx], containment.indptr[rowidx + 1])]
        cohort_ = present_labels[cohidx]
        cohort = [elem.item() for elem in cohort_ if elem not in merged_keys]
        if not cohort:
            continue
        merged_keys.update(cohort)
        allchunks = (label_chunks[member].tolist() for member in cohort)
        chunk = tuple(set(itertools.chain(*allchunks)))
        merged_cohorts[chunk] = cohort

    actual_ngroups = np.concatenate(tuple(merged_cohorts.values())).size
    expected_ngroups = present_labels.size
    assert len(merged_keys) == actual_ngroups
    assert expected_ngroups == actual_ngroups, (expected_ngroups, actual_ngroups)

    # sort by first label in cohort
    # This will help when sort=True (default)
    # and we have to resort the dask array
    as_sorted = dict(sorted(merged_cohorts.items(), key=lambda kv: kv[1][0]))
    return preferred_method, as_sorted
