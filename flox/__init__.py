#!/usr/bin/env python
# flake8: noqa
"""Top-level module for flox ."""

from . import cache
from .aggregations import Aggregation, Scan, is_supported_aggregation
from .core import (
    groupby_reduce,
    rechunk_for_blockwise,
    rechunk_for_cohorts,
)
from .options import set_options
from .reindex import ReindexArrayType, ReindexStrategy
from .scan import groupby_scan


def _get_version():
    __version__ = "999"
    try:
        from ._version import __version__
    except ImportError:
        pass
    return __version__


__version__ = _get_version()

__all__ = [
    "Aggregation",
    "Scan",
    "groupby_reduce",
    "groupby_scan",
    "rechunk_for_blockwise",
    "rechunk_for_cohorts",
    "set_options",
    "ReindexStrategy",
    "ReindexArrayType",
    "is_supported_aggregation",
]
