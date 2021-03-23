import itertools
from functools import partial
from operator import getitem
from typing import Iterable, Mapping, Union

import dask.array
import numpy as np
import numpy_groupies as npg
import pandas as pd
from dask.highlevelgraph import HighLevelGraph

DaskArray = dask.array.Array

# how to aggregate results after first round of reduction
# e.g. we "sum" the "count" to aggregate counts across blocks
_agg_reduction = {"count": "sum"}


def chunk_reduce(array, to_group, func, fill_value=0, size=None):
    """
    Wrapper for numpy_groupies aggregate that supports nD ``array`` and
    mD ``to_group``.
    """

    # if indices = [2, 2, 2], npg assumes groups are (0, 1, 2);
    # and will return a result that is bigger than necessary
    # avoid by factorizing again so indices=[2,2,2] is changed to
    # indices=[0,0,0]

    # factorize can handle strings etc unlike digitize
    group_idx, groups = pd.factorize(to_group.ravel())

    # TODO: deal with NaNs in to_group
    # assert (axis == np.sort(axis)).all()

    newshape = (np.prod(array.shape[: to_group.ndim]),) + array.shape[to_group.ndim :]
    array = array.reshape(newshape)

    # mask = np.logical_not(np.isnan(to_group))
    # group_idx = digitized[mask].astype(int)

    results = {"groups": groups}
    for reduction in func:
        results[reduction] = npg.aggregate_numpy.aggregate(
            group_idx,
            array,
            axis=0,
            func=reduction,
        )
    return results


def _npg_aggregate(x_chunk, func, axis, keepdims):
    """ Aggregation step of tree reduction. """
    from dask.array.core import _concatenate2
    from dask.utils import deepmap

    # print(x_chunk)
    if isinstance(x_chunk, dict):
        x_chunk = [x_chunk]

    def _conc(key):
        return np.concatenate([xx[key] for xx in x_chunk])

    def _conc2(key):
        """ copied from dask.array.reductions.mean_combine"""

        # some magic
        mapped = deepmap(lambda x: x[key], x_chunk)
        return _concatenate2(mapped, axes=(0,))

    groups = _conc2("groups")
    # print(groups)

    results = {}
    for reduction in func:
        _reduction = _agg_reduction.get(reduction, reduction)
        x = _conc2(reduction)
        # print("x", x)
        _results = chunk_reduce(x, groups, func=(_reduction,))
        results.update({reduction: _results[_reduction]})

    results["groups"] = _results["groups"]
    return results


def groupby_agg(
    array: DaskArray,
    to_group: DaskArray,
    func: Union[str, Iterable[str]],
    expected_groups: Iterable = None,
):
    # TODO: Make gufunc compatible by expecting aggregated dimensions at the end
    #       instead of at the beginning
    inds = tuple(range(array.ndim))
    axis = tuple(range(to_group.ndim))
    assert array.shape[: to_group.ndim] == to_group.shape

    group_chunks = (len(expected_groups),) if expected_groups is not None else (np.nan,)
    output_chunks = tuple(array.chunks[i] for i in range(array.ndim) if i not in axis)
    output_chunks = (group_chunks,) + (output_chunks)

    # apply reduction on chunk
    applied = dask.array.blockwise(
        partial(chunk_reduce, func=func),
        inds,
        array,
        inds,
        to_group,
        inds[: to_group.ndim],
        concatenate=False,
        dtype=array.dtype,
        meta=array._meta,
        name="groupby-chunk-reduce",
    )

    # reduced is a dict mapping reduction name to array and "groups" to an array of group labels
    reduced = dask.array.reductions._tree_reduce(
        applied,
        aggregate=partial(_npg_aggregate, func=func),
        name="groupby-tree-reduce",
        dtype=array.dtype,
        axis=axis,
        keepdims=True,
        concatenate=False,
    )

    ochunks = tuple(range(len(chunks_v)) for chunks_v in output_chunks)
    ichunks = (range(1),) * len(axis) + tuple(
        range(len(chunks_v)) for chunks_v in array.chunks[len(axis) :]
    )
    # TODO: write as dict comprehension
    # TODO: check that inchunk is right
    # extract results from the dict
    layer = {}
    for reduction in func:
        for ochunk, inchunk in zip(itertools.product(*ochunks), itertools.product(*ichunks)):
            layer[(reduction, *ochunk)] = (getitem, (reduced.name, *inchunk), reduction)

    # we've used keepdims=True, so _tree_reduce preserves some dummy dimensions
    first_block = len(ochunks) * (0,)  # TODO: this may not be right
    layer[("groups", 0)] = (getitem, (reduced.name, *first_block), "groups")
    print(layer)

    # TODO: is this the best way to create multiple output arrays from a single graph?
    # it looks like the approach in dask.array.apply_gufunc
    result = {}
    if expected_groups is None:
        result["groups"] = DaskArray(
            HighLevelGraph.from_collections("groups", layer, dependencies=[reduced]),
            "groups",
            chunks=(group_chunks,),
            dtype=object,  # TODO: change appropriately
        )
    else:
        result["groups"] = np.array(expected_groups)

    for reduction in func:
        result[reduction] = DaskArray(
            HighLevelGraph.from_collections(reduction, layer, dependencies=[reduced]),
            reduction,
            chunks=output_chunks,
            meta=array._meta,  # TODO: change appropriately
        )
    return result


def groupby_reduce(
    array: DaskArray,
    to_group: DaskArray,
    func: Union[str, Iterable[str]],
    expected_groups: Iterable = None,
) -> Mapping[str, DaskArray]:
    rewrite_func = {"mean": (("sum", "count"))}

    if isinstance(func, str):
        func = [func]

    reductions = tuple(
        itertools.chain(*[rewrite_func.get(reduction, [reduction]) for reduction in func])
    )

    intermediate = groupby_agg(array, to_group, reductions, expected_groups)

    result = {"groups": intermediate["groups"]}
    for reduction in func:
        if reduction == "mean":
            result["mean"] = intermediate["sum"] / intermediate["count"]

    return result
