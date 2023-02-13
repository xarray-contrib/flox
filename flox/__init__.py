#!/usr/bin/env python
# flake8: noqa
"""Top-level module for flox ."""
from . import cache
from .aggregations import Aggregation  # noqa
from .core import groupby_reduce, rechunk_for_blockwise, rechunk_for_cohorts  # noqa


def _get_version():
    __version__ = "999"
    try:
        from ._version import __version__
    except ImportError:
        pass
    return __version__


__version__ = _get_version()
