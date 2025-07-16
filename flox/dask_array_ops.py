import builtins
import math
from functools import lru_cache, partial
from itertools import product
from numbers import Integral

import dask
import pandas as pd
from dask import config
from dask.base import normalize_token
from dask.blockwise import lol_tuples
from packaging.version import Version
from toolz import partition_all

from .lib import ArrayLayer
from .types import Graph

if Version(dask.__version__) <= Version("2025.03.1"):
    # workaround for https://github.com/dask/dask/issues/11862
    @normalize_token.register(pd.RangeIndex)
    def normalize_range_index(x):
        return normalize_token(type(x)), x.start, x.stop, x.step, x.dtype, x.name


# _tree_reduce and partial_reduce are copied from dask.array.reductions
# They have been modified to work purely with graphs, and without creating new Array layers
# in the graph. The `block_index` kwarg is new and avoids a concatenation by simply setting the right
# key initially
def _tree_reduce(
    x: ArrayLayer,
    *,
    name: str,
    out_dsk: Graph,
    aggregate,
    axis: tuple[int, ...],
    block_index: int,
    split_every=None,
    combine=None,
):
    # Normalize split_every
    split_every = split_every or config.get("split_every", 4)
    if isinstance(split_every, dict):
        split_every = {k: split_every.get(k, 2) for k in axis}
    elif isinstance(split_every, Integral):
        n = builtins.max(int(split_every ** (1 / (len(axis) or 1))), 2)
        split_every = dict.fromkeys(axis, n)
    else:
        raise ValueError("split_every must be a int or a dict")

    numblocks = tuple(len(c) for c in x.chunks)
    out_chunks = x.chunks

    # Reduce across intermediates
    depth = 1
    for i, n in enumerate(numblocks):
        if i in split_every and split_every[i] != 1:
            depth = int(builtins.max(depth, math.ceil(math.log(n, split_every[i]))))
    func = partial(combine or aggregate, axis=axis)

    agg_dep_name = x.name
    for level in range(depth - 1):
        newname = name + f"-{block_index}-partial-{level}"
        out_dsk, out_chunks = partial_reduce(
            func,
            out_dsk,
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
        out_dsk,
        chunks=out_chunks,
        split_every=split_every,
        name=name,
        dep_name=agg_dep_name,
        axis=axis,
        block_index=block_index,
    )


def partial_reduce(
    func,
    dsk,
    *,
    chunks: tuple[tuple[int, ...], ...],
    name: str,
    dep_name: str,
    split_every: dict[int, int],
    axis: tuple[int, ...],
    block_index: int | None = None,
):
    ndim = len(chunks)
    keys, parts, out_chunks = get_parts(tuple(split_every.items()), chunks)
    for k, p in zip(keys, product(*parts)):
        free = {i: j[0] for (i, j) in enumerate(p) if len(j) == 1 and i not in split_every}
        dummy = dict(i for i in enumerate(p) if i[0] in split_every)
        g = lol_tuples((dep_name,), range(ndim), free, dummy)
        assert dep_name != name
        if block_index is not None:
            k = (*k[:-1], block_index)
        dsk[(name,) + k] = (func, g)
    return dsk, out_chunks


@lru_cache
def get_parts(split_every_items, chunks):
    numblocks = tuple(len(c) for c in chunks)
    split_every = dict(split_every_items)

    parts = [list(partition_all(split_every.get(i, 1), range(n))) for (i, n) in enumerate(numblocks)]
    keys = tuple(product(*map(range, map(len, parts))))
    out_chunks = tuple(
        tuple(1 for p in partition_all(split_every[i], c)) if i in split_every else c
        for (i, c) in enumerate(chunks)
    )
    return keys, parts, out_chunks
