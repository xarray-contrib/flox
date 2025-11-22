"""Dask-specific functions for groupby operations.

This module provides Dask-specific implementations for groupby operations.
Functions are re-exported from flox.core for backward compatibility,
with the intent to gradually move implementations here.
"""

from __future__ import annotations

# Re-export dask-specific functions from core for backward compatibility
from .core import (
    _collapse_blocks_along_axes,
    _extract_unknown_groups,
    _grouped_combine,
    _normalize_indexes,
    _simple_combine,
    _unify_chunks,
    dask_groupby_agg,
    dask_groupby_scan,
    subset_to_blocks,
)

__all__ = [
    "_collapse_blocks_along_axes",
    "_extract_unknown_groups",
    "_grouped_combine",
    "_normalize_indexes",
    "_simple_combine",
    "_unify_chunks",
    "dask_groupby_agg",
    "dask_groupby_scan",
    "subset_to_blocks",
]
