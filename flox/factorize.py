"""Factorization functions for groupby operations.

This module provides functions for factorizing groupby labels.
Re-exported from flox.core for backward compatibility.
"""

from __future__ import annotations

# Re-export from core for backward compatibility
from .core import (
    FactorizeKwargs,
    FactorProps,
    _factorize_multiple,
    _factorize_single,
    _lazy_factorize_wrapper,
    _ravel_factorized,
    factorize_,
    offset_labels,
)

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
