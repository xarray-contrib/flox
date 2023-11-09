import numpy as np
import numpy_groupies as npg


def tdigest_chunk(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None, **kwargs):
    from crick import TDigest

    def _(arr):
        digest = TDigest()
        # we receive object arrays from numpy_groupies
        digest.update(arr.astype(array.dtype, copy=False))
        return digest

    result = npg.aggregate_numpy.aggregate(
        group_idx, array, func=_, size=size, fill_value=fill_value, axis=axis, dtype=object
    )
    return result


def tdigest_combine(digests, axis=-1, keepdims=True):
    from crick import TDigest

    def _(arr):
        t = TDigest()
        t.merge(*arr)
        return np.array([t], dtype=object)

    if not isinstance(axis, tuple):
        axis = (axis,)

    # If reducing along multiple axes, we can just keep combining ;)
    result = digests
    for ax in axis:
        result = np.apply_along_axis(_, ax, result)

    return result


def tdigest_aggregate(digests, q, axis=-1, keepdims=True):
    for idx in np.ndindex(digests.shape):
        digests[idx] = digests[idx].quantile(q)
    return digests
