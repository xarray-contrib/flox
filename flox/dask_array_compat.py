import builtins
import math
from functools import partial
from itertools import product
from numbers import Integral

from dask import config
from dask.blockwise import lol_tuples
from tlz import partition_all


def _tree_reduce(
    dsk,
    *,
    chunks,
    aggregate,
    axis,
    name: str,
    dep_name: str,
    block_index: int,
    split_every=None,
    combine=None,
):
    """Perform the tree reduction step of a reduction.

    Lower level, users should use ``reduction`` or ``arg_reduction`` directly.
    """
    # Normalize split_every
    split_every = split_every or config.get("split_every", 4)
    if isinstance(split_every, dict):
        split_every = {k: split_every.get(k, 2) for k in axis}
    elif isinstance(split_every, Integral):
        n = builtins.max(int(split_every ** (1 / (len(axis) or 1))), 2)
        split_every = dict.fromkeys(axis, n)
    else:
        raise ValueError("split_every must be a int or a dict")

    numblocks = tuple(len(c) for c in chunks)
    out_chunks = chunks

    # Reduce across intermediates
    depth = 1
    for i, n in enumerate(numblocks):
        if i in split_every and split_every[i] != 1:
            depth = int(builtins.max(depth, math.ceil(math.log(n, split_every[i]))))
    func = partial(combine or aggregate, axis=axis)

    agg_dep_name = dep_name
    for level in range(depth - 1):
        newname = name + f"-{block_index}-partial-{level}"
        dsk, out_chunks = partial_reduce(
            func,
            dsk,
            chunks=out_chunks,
            split_every=split_every,
            name=newname,
            dep_name=agg_dep_name,
            axis=axis,
        )
        agg_dep_name = newname
    func = partial(aggregate, axis=axis)
    return partial_reduce(
        func,
        dsk,
        chunks=out_chunks,
        split_every=split_every,
        name=name,
        dep_name=agg_dep_name,
        axis=axis,
        block_index=block_index,
    )


def partial_reduce(func, dsk, *, chunks, name, dep_name, split_every, axis, block_index=None):
    """Partial reduction across multiple axes.

    Parameters
    ----------
    func : function
    x : Array
    split_every : dict
        Maximum reduction block sizes in each dimension.

    Examples
    --------
    Reduce across axis 0 and 2, merging a maximum of 1 block in the 0th
    dimension, and 3 blocks in the 2nd dimension:

    >>> partial_reduce(np.min, x, {0: 1, 2: 3})  # doctest: +SKIP
    """
    numblocks = tuple(len(c) for c in chunks)
    ndim = len(numblocks)
    parts = [list(partition_all(split_every.get(i, 1), range(n))) for (i, n) in enumerate(numblocks)]
    keys = product(*map(range, map(len, parts)))
    out_chunks = [
        tuple(1 for p in partition_all(split_every[i], c)) if i in split_every else c
        for (i, c) in enumerate(chunks)
    ]
    for k, p in zip(keys, product(*parts)):
        free = {i: j[0] for (i, j) in enumerate(p) if len(j) == 1 and i not in split_every}
        dummy = dict(i for i in enumerate(p) if i[0] in split_every)
        g = lol_tuples((dep_name,), range(ndim), free, dummy)
        assert dep_name != name
        if block_index is not None:
            k = list(k)
            k[axis[-1]] = block_index
            k = tuple(k)
        dsk[(name,) + k] = (func, g)
    return dsk, out_chunks
