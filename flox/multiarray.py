from collections.abc import Callable
from typing import Self

import numpy as np

MULTIARRAY_HANDLED_FUNCTIONS: dict[Callable, Callable] = {}


class MultiArray:
    arrays: tuple[np.ndarray, ...]

    def __init__(self, arrays):
        self.arrays = arrays
        assert all(arrays[0].shape == a.shape for a in arrays), "Expect all arrays to have the same shape"

    def astype(self, dt, **kwargs) -> Self:
        return type(self)(tuple(array.astype(dt, **kwargs) for array in self.arrays))

    def reshape(self, shape, **kwargs) -> Self:
        return type(self)(tuple(array.reshape(shape, **kwargs) for array in self.arrays))

    def squeeze(self, axis=None) -> Self:
        return type(self)(tuple(array.squeeze(axis) for array in self.arrays))

    def __setitem__(self, key, value) -> None:
        assert len(value) == len(self.arrays)
        for array, val in zip(self.arrays, value):
            array[key] = val

    def __array_function__(self, func, types, args, kwargs):
        if func not in MULTIARRAY_HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects
        # if not all(issubclass(t, MyArray) for t in types): # I can't see this being relevant at all for this code, but maybe it's safer to leave it in?
        # return NotImplemented
        return MULTIARRAY_HANDLED_FUNCTIONS[func](*args, **kwargs)

    # Shape is needed, seems likely that the other two might be
    # Making some strong assumptions here that all the arrays are the same shape, and I don't really like this
    @property
    def dtype(self) -> np.dtype:
        return self.arrays[0].dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arrays[0].shape

    @property
    def ndim(self) -> int:
        return self.arrays[0].ndim

    def __getitem__(self, key) -> Self:
        return type(self)([array[key] for array in self.arrays])


def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""

    def decorator(func):
        MULTIARRAY_HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.expand_dims)
def expand_dims(multiarray, axis) -> MultiArray:
    return MultiArray(tuple(np.expand_dims(a, axis) for a in multiarray.arrays))


@implements(np.concatenate)
def concatenate(multiarrays, axis) -> MultiArray:
    n_arrays = len(multiarrays[0].arrays)
    for ma in multiarrays[1:]:
        assert len(ma.arrays) == n_arrays
    return MultiArray(
        tuple(np.concatenate(tuple(ma.arrays[i] for ma in multiarrays), axis) for i in range(n_arrays))
    )


@implements(np.transpose)
def transpose(multiarray, axes) -> MultiArray:
    return MultiArray(tuple(np.transpose(a, axes) for a in multiarray.arrays))


@implements(np.squeeze)
def squeeze(multiarray, axis) -> MultiArray:
    return MultiArray(tuple(np.squeeze(a, axis) for a in multiarray.arrays))


@implements(np.full)
def full(shape, fill_values, *args, **kwargs) -> MultiArray:
    """All arguments except fill_value are shared by each array in the MultiArray.
    Iterate over fill_values to create arrays
    """
    return MultiArray(tuple(np.full(shape, fv, *args, **kwargs) for fv in fill_values))
