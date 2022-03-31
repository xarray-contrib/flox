#!/usr/bin/env python
# flake8: noqa
"""Top-level module for flox ."""
from .aggregations import Aggregation  # noqa
from .core import groupby_reduce, rechunk_for_blockwise, rechunk_for_cohorts  # noqa

try:
    from importlib.metadata import version as _version
except ImportError:
    # if the fallback library is missing, we are doomed.
    from importlib_metadata import version as _version  # type: ignore[no-redef]

try:
    __version__ = _version("flox")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
