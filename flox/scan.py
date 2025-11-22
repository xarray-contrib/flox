"""Scan operations for groupby reductions.

This module provides scan operations (cumulative reductions) for grouped data.
Re-exported from flox.core for backward compatibility.
"""

from __future__ import annotations

# Re-export from core for backward compatibility
from .core import (
    _finalize_scan,
    _zip,
    chunk_scan,
    groupby_scan,
    grouped_reduce,
)

__all__ = [
    "_finalize_scan",
    "_zip",
    "chunk_scan",
    "grouped_reduce",
    "groupby_scan",
]
