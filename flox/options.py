"""
Started from xarray options.py; vendored from cf-xarray
"""

import copy
from collections.abc import MutableMapping
from typing import Any

OPTIONS: MutableMapping[str, Any] = {
    # Thresholds below which we will automatically rechunk to blockwise if it makes sense
    # 1. Fractional change in number of chunks after rechunking
    "rechunk_blockwise_num_chunks_threshold": 0.25,
    # 2. Fractional change in max chunk size after rechunking
    "rechunk_blockwise_chunk_size_threshold": 1.5,
    # 3. If input arrays have chunk size smaller than `dask.array.chunk-size`,
    #    then adjust chunks to meet that size first.
    # "rechunk.blockwise.chunk_size_factor": 1.5,
}


class set_options:  # numpydoc ignore=PR01,PR02
    """
    Set options for cf-xarray in a controlled context.

    Parameters
    ----------
    rechunk_blockwise_num_chunks_threshold : float
        Rechunk if fractional change in number of chunks after rechunking
        is less than this amount.
    rechunk_blockwise_chunk_size_threshold: float
        Rechunk if fractional change in max chunk size after rechunking
        is less than this threshold.

    Examples
    --------

    You can use ``set_options`` either as a context manager:

    >>> import flox
    >>> with flox.set_options(rechunk_blockwise_num_chunks_threshold=1):
    ...     pass

    Or to set global options:

    >>> flox.set_options(rechunk_blockwise_num_chunks_threshold=1):
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k in kwargs:
            if k not in OPTIONS:
                raise ValueError(f"argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}")
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        options_dict = copy.deepcopy(options_dict)
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)
