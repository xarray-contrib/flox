# The functions defined here were copied based on the source code
# defined in xarray


from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import dask.array

    dask_array_type = dask.array.Array
except ImportError:
    dask_array_type = ()


def is_duck_array(value: Any) -> bool:
    """Checks if value is a duck array."""
    if isinstance(value, np.ndarray):
        return True
    return (
        hasattr(value, "ndim")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "__array_function__")
        and hasattr(value, "__array_ufunc__")
    )


def is_dask_collection(x):
    try:
        import dask

        return dask.is_dask_collection(x)

    except ImportError:
        return False


def is_duck_dask_array(x):
    return is_duck_array(x) and is_dask_collection(x)


class ReprObject:
    """Object that prints as the given value, for use with sentinel values."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def __repr__(self) -> str:
        return self._value

    def __eq__(self, other) -> bool:
        if isinstance(other, ReprObject):
            return self._value == other._value
        return False

    def __hash__(self) -> int:
        return hash((type(self), self._value))

    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return normalize_token((type(self), self._value))


def is_scalar(value: Any, include_0d: bool = True) -> bool:
    """Whether to treat a value as a scalar.

    Any non-iterable, string, or 0-D array
    """
    NON_NUMPY_SUPPORTED_ARRAY_TYPES = (dask_array_type, pd.Index)

    if include_0d:
        include_0d = getattr(value, "ndim", None) == 0
    return (
        include_0d
        or isinstance(value, (str, bytes))
        or not (
            isinstance(value, (Iterable,) + NON_NUMPY_SUPPORTED_ARRAY_TYPES)
            or hasattr(value, "__array_function__")
        )
    )


def isnull(data):
    data = np.asarray(data)
    scalar_type = data.dtype.type
    if issubclass(scalar_type, (np.datetime64, np.timedelta64)):
        # datetime types use NaT for null
        # note: must check timedelta64 before integers, because currently
        # timedelta64 inherits from np.integer
        return np.isnat(data)
    elif issubclass(scalar_type, np.inexact):
        # float types use NaN for null
        return np.isnan(data)
    elif issubclass(scalar_type, (np.bool_, np.integer, np.character, np.void)):
        # these types cannot represent missing values
        return np.zeros_like(data, dtype=bool)
    else:
        # at this point, array should have dtype=object
        if isinstance(data, (np.ndarray, dask_array_type)):
            return pd.isnull(data)
        else:
            # Not reachable yet, but intended for use with other duck array
            # types. For full consistency with pandas, we should accept None as
            # a null value as well as NaN, but it isn't clear how to do this
            # with duck typing.
            return data != data
