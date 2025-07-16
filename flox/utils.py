"""Utility classes and functions for flox."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np


class ReindexArrayType(Enum):
    """
    Enum describing which array type to reindex to.

    These are enumerated, rather than accepting a constructor,
    because we might want to optimize for specific array types,
    and because they don't necessarily have the same signature.

    For example, scipy.sparse.COO only supports a fill_value of 0.
    """

    AUTO = auto()
    NUMPY = auto()
    SPARSE_COO = auto()
    # Sadly, scipy.sparse.coo_array only supports fill_value = 0
    # SCIPY_SPARSE_COO = auto()
    # SPARSE_GCXS = auto()

    def is_same_type(self, other) -> bool:
        match self:
            case ReindexArrayType.AUTO:
                return True
            case ReindexArrayType.NUMPY:
                return isinstance(other, np.ndarray)
            case ReindexArrayType.SPARSE_COO:
                import sparse

                return isinstance(other, sparse.COO)


__all__ = ["ReindexArrayType"]
