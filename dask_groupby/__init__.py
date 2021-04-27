#!/usr/bin/env python
# flake8: noqa
"""Top-level module for dask_groupby ."""
from pkg_resources import DistributionNotFound, get_distribution

from .core import groupby_reduce, xarray_reduce  # noqa

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    _version__ = "unknown"
