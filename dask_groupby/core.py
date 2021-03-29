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
fill_values = {"count": 0}


def reindex_(array, from_, to, fill_value=0, axis=-1):

    assert axis in (0, -1)
    idx = np.array([np.argwhere(np.array(from_) == bb)[0, 0] if bb in from_ else -1 for bb in to])
    indexer = [slice(None, None)] * array.ndim
    indexer[axis] = idx
    reindexed = array[tuple(indexer)]
    if axis == 0:
        loc = (idx == -1, ...)
    else:
        loc = (..., idx == -1)
    reindexed[loc] = fill_value
    return reindexed


def offset_labels(a):
    """
    Offset group labels by dimension.
    Copied from https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    """
    N = a.max() + 1
    offset = a + np.arange(np.prod(a.shape[:-1])).reshape((*a.shape[:-1], -1)) * N
    # print("N =", N, "offset = ", offset)
    size = np.prod(a.shape[:-1]) * N
    return offset, N, size


def chunk_reduce(array, to_group, func, axis=None, expected_groups=None):
    """
    Wrapper for numpy_groupies aggregate that supports nD ``array`` and
    mD ``to_group``.

    Core groupby reduction using numpy_groupies. Uses ``pandas.factorize`` to factorize
    ``to_group``. Offsets the groups if not reducing along all dimensions of ``to_group``.
    Always ravels ``to_group`` to 1D, flattens appropriate dimensions of array.

    When dask arrays are passed to groupby_reduce, this function is called on every
    block.

    Parameters
    ----------
    array: numpy.ndarray
        Array of values to reduced
    to_group: numpy.ndarray
        Array to group by.
    func: str or Tuple[str]
        Name of reduction, passed to numpy_groupies. Supports multiple reductions.
    axis: (optional) int or Tuple[int]
        If None, reduce along all dimensions of to_group.
        Else reduce along specified axes

    Returns
    -------
    dict
    """

    if isinstance(func, str):
        func = (func,)

    if isinstance(axis, Iterable) and len(axis) == 1:
        axis = axis[0]

    if to_group.ndim == 1:
        assert axis in (0, -1, array.ndim - 1, None)
        axis = None

    if np.isscalar(axis) and axis != -1 and axis != to_group.ndim:
        to_group = np.swapaxes(to_group, axis, -1)
        array = np.swapaxes(array, axis, -1)

    # if indices=[2,2,2], npg assumes groups are (0, 1, 2);
    # and will return a result that is bigger than necessary
    # avoid by factorizing again so indices=[2,2,2] is changed to
    # indices=[0,0,0]. This is necessary when combining block results
    # factorize can handle strings etc unlike digitize
    group_idx, groups = pd.factorize(to_group.ravel())
    final_shape = to_group.shape[:-1]
    size = None

    if axis is not None:
        # Not reducing along all dimensions of to_group
        # offset the group ids
        group_idx, N, size = offset_labels(group_idx.reshape(to_group.shape))
        group_idx = group_idx.ravel()

    # TODO: deal with NaNs in to_group
    # assert (axis == np.sort(axis)).all()

    # always reshape to 1D along group dimensions
    newshape = array.shape[: array.ndim - to_group.ndim] + (np.prod(array.shape[-to_group.ndim :]),)
    array = array.reshape(newshape)

    # mask = np.logical_not(np.isnan(to_group))
    # group_idx = digitized[mask].astype(int)

    # print(array)
    if expected_groups:
        results = {"groups": expected_groups}
    else:
        sortidx = np.argsort(groups)
        results = {"groups": groups[sortidx]}

    for reduction in func:
        result = npg.aggregate_numpy.aggregate(
            group_idx,
            array,
            axis=-1,
            func=reduction,
            size=size,
        )
        if axis is not None:
            result = result.reshape(*final_shape, N)
        if expected_groups:
            results[reduction] = reindex_(result, groups, expected_groups)
        else:
            results[reduction] = result[..., sortidx]

    return results


def _npg_aggregate(x_chunk, func, expected_groups, axis, keepdims):
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
        return _concatenate2(mapped, axes=(-1,))

    groups = _conc2("groups")
    # print(groups)

    results = {}
    for reduction in func:
        _reduction = _agg_reduction.get(reduction, reduction)
        x = _conc2(reduction)
        # print("x", x)
        _results = chunk_reduce(
            x, groups, func=(_reduction,), axis=axis, expected_groups=expected_groups
        )
        results.update({reduction: _results[_reduction]})

    results["groups"] = _results["groups"]
    return results


def groupby_agg(
    array: DaskArray,
    to_group: DaskArray,
    func: Union[str, Iterable[str]],
    expected_groups: Iterable = None,
    axis=None,
):
    inds = tuple(range(array.ndim))

    # set axis for _tree_reduce
    if isinstance(axis, Iterable):
        reduced_ndim = len(axis)
    elif axis is None:
        reduced_ndim = to_group.ndim
    else:
        reduced_ndim = 1
    axis = tuple(array.ndim - np.arange(reduced_ndim) - 1)

    assert array.shape[-to_group.ndim :] == to_group.shape

    # apply reduction on chunk
    applied = dask.array.blockwise(
        partial(chunk_reduce, func=func, axis=axis, expected_groups=expected_groups),
        inds,
        array,
        inds,
        to_group,
        inds[-to_group.ndim :],
        concatenate=False,
        dtype=array.dtype,
        meta=array._meta,
        name="groupby-chunk-reduce",
    )

    # reduced is really a dict mapping reduction name to array
    # and "groups" to an array of group labels
    reduced = dask.array.reductions._tree_reduce(
        applied,
        aggregate=partial(_npg_aggregate, func=func, expected_groups=expected_groups),
        name="groupby-tree-reduce",
        dtype=array.dtype,
        axis=axis,
        keepdims=True,
        concatenate=False,
    )

    group_chunks = (len(expected_groups),) if expected_groups is not None else (np.nan,)
    output_chunks = reduced.chunks[:-1] + (group_chunks,)

    # extract results from the dict
    ochunks = tuple(range(len(chunks_v)) for chunks_v in output_chunks)
    layer = {}
    for reduction in func:
        for ochunk in itertools.product(*ochunks):
            layer[(reduction, *ochunk)] = (getitem, (reduced.name, *ochunk), reduction)

    # we've used keepdims=True, so _tree_reduce preserves some dummy dimensions
    first_block = len(ochunks) * (0,)  # TODO: this may not be right
    layer[("groups", 0)] = (getitem, (reduced.name, *first_block), "groups")
    print(layer)

    result = {}
    if expected_groups is None:
        result["groups"] = DaskArray(
            HighLevelGraph.from_collections("groups", layer, dependencies=[reduced]),
            "groups",
            chunks=(group_chunks,),
            dtype=object,  # TODO: change appropriately
        )
    else:
        result["groups"] = np.sort(expected_groups)  # TODO: check

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
    axis=None,
) -> Mapping[str, DaskArray]:
    rewrite_func = {"mean": (("sum", "count"))}

    if not isinstance(array, DaskArray) and not isinstance(to_group, DaskArray):
        return chunk_reduce(array, to_group, func, axis, expected_groups)

    if isinstance(func, str):
        func = [func]

    if np.isscalar(axis) and to_group.ndim > 1 and expected_groups is None:
        # When we reduce along all axes, it guarantees that we will see all
        # groups in the final combine stage, so everything works.
        # This is not necessarily true when reducing along a subset of axes
        # (of to_group)
        # TODO: depends on chunking of to_group?
        # we could relax this if there is only one chunk along all
        # to_group dim != axis?
        raise NotImplementedError(
            "Please provide ``expected_groups`` when not reducing along all axes."
        )
    if np.isscalar(axis) and axis != -1 and axis != to_group.ndim:
        to_group = np.swapaxes(to_group, axis, -1)
        array = np.swapaxes(array, axis, -1)
        axis = -1

    reductions = tuple(
        itertools.chain(*[rewrite_func.get(reduction, [reduction]) for reduction in func])
    )

    intermediate = groupby_agg(array, to_group, reductions, expected_groups, axis=axis)

    result = {"groups": intermediate["groups"]}
    for reduction in func:
        if reduction == "mean":
            result["mean"] = intermediate["sum"] / intermediate["count"]
        else:
            result[reduction] = intermediate[reduction]

    return result
