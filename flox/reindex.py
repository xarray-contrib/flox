"""Reindexing functions for groupby operations.

This module provides functions for reindexing arrays during groupby operations.
Re-exported from flox.core for backward compatibility.
"""

from __future__ import annotations

# Re-export from core for backward compatibility
from .core import (
    reindex_,
    reindex_numpy,
    reindex_pydata_sparse_coo,
)

__all__ = [
    "reindex_",
    "reindex_numpy",
    "reindex_pydata_sparse_coo",
]
