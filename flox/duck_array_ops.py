import numpy as np


def get_array_namespace(x):
    if hasattr(x, "__array_namespace__"):
        return x.__array_namespace__()
    else:
        return np


def reshape(array, shape):
    xp = get_array_namespace(array)
    return xp.reshape(array, shape)


def asarray(obj, like):
    xp = get_array_namespace(like)
    return xp.asarray(obj)
