# The functions defined here were copied based on the source code
# defined in xarray


from typing import Any

import numpy as np


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
