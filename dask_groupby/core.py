import itertools
from functools import partial
from operator import getitem
from typing import Iterable, Mapping, Union

import dask.array
import numpy as np
import numpy_groupies as npg
import pandas as pd
from dask.highlevelgraph import HighLevelGraph
from icecream import ic

# how to aggregate results after first round of reduction
# e.g. we "sum" the "count" to aggregate counts across blocks
_agg_reduction = {"count": "sum"}

# These are used to reindex to expected_groups.
# They should make sense when aggregated together with results from other blocks
fill_values = {"count": 0, "sum": 0}


def _collapse_and_reshape(arr: np.ndarray, axis: Iterable[int]):
    axis = tuple(axis)
    order = tuple(ax for ax in np.arange(arr.ndim) if ax not in axis) + axis
    # ic(order)
    arr = arr.transpose(order)
    newshape = arr.shape[: -len(axis)] + (np.prod(arr.shape[-len(axis) :]),)
    # ic(arr.shape, axis, newshape)
    return arr.reshape(newshape)


def reindex_(array: np.ndarray, from_, to, fill_value=0, axis=-1):

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


def offset_labels(a: np.ndarray):
    """
    Offset group labels by dimension.
    Copied from https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    """
    N = a.max() + 1
    offset = a + np.arange(np.prod(a.shape[:-1])).reshape((*a.shape[:-1], -1)) * N
    # -1 indicates NaNs. preserve these otherwise we aggregate in the wrong groups!
    offset[a == -1] = -1
    # print("N =", N, "offset = ", offset)
    size = np.prod(a.shape[:-1]) * N
    return offset, N, size


def chunk_reduce(
    array: np.ndarray,
    to_group: np.ndarray,
    func: Union[str, Iterable[str]],
    expected_groups: Iterable = None,
    axis: Union[int, Iterable[int]] = None,
):
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
    func: str or Iterable[str]
        Name of reduction, passed to numpy_groupies. Supports multiple reductions.
    axis: (optional) int or Iterable[int]
        If None, reduce along all dimensions of array.
        Else reduce along specified axes.

    Returns
    -------
    dict
    """

    if isinstance(func, str):
        func = (func,)

    # ic(array, to_group, axis)
    n = len(axis) if isinstance(axis, Iterable) else to_group.ndim
    final_array_shape = array.shape[:-n] + (1,) * (n - 1)
    final_groups_shape = (1,) * (n - 1)
    # ic(array.shape, to_group.shape, axis, final_array_shape, final_groups_shape)

    if isinstance(axis, Iterable):
        if len(axis) == 1:
            axis = next(iter(axis))
        elif np.all(np.sort(axis) == np.arange(array.ndim)):
            # None indicates reduction along all axes of to_group
            axis = None

    # when axis is a tuple
    # collapse and move reduction dimensions to the end
    # TODO: move this down to chunk_reduce
    if isinstance(axis, Iterable) and len(axis) < array.ndim:
        to_group = _collapse_and_reshape(to_group, -array.ndim + np.array(axis) + to_group.ndim)
        array = _collapse_and_reshape(array, axis)
        axis = -1
        # ic(array.shape, to_group.shape, axis)

    if to_group.ndim == 1:
        # TODO: This assertion doesn't work with dask reducing across all dimensions
        # when to_group.ndim == array.ndim
        # the intermediates are 1D but axis=range(array.ndim)
        # assert axis in (0, -1, array.ndim - 1, None)
        axis = -1

    # if indices=[2,2,2], npg assumes groups are (0, 1, 2);
    # and will return a result that is bigger than necessary
    # avoid by factorizing again so indices=[2,2,2] is changed to
    # indices=[0,0,0]. This is necessary when combining block results
    # factorize can handle strings etc unlike digitize
    group_idx, groups = pd.factorize(to_group.ravel())
    size = None

    offset_group = False
    if np.isscalar(axis) and to_group.ndim > 1:
        ic(f"offsetting because axis={axis}")
        offset_group = True
        ic(array, to_group)
        # Not reducing along all dimensions of to_group
        # offset the group ids
        group_idx, N, size = offset_labels(group_idx.reshape(to_group.shape))
        group_idx = group_idx.ravel()

    # always reshape to 1D along group dimensions
    newshape = array.shape[: array.ndim - to_group.ndim] + (np.prod(array.shape[-to_group.ndim :]),)
    array = array.reshape(newshape)
    # ic(array.shape, group_idx.shape, axis)

    # pd.factorize uses -1 to indicate NaNs
    assert group_idx.ndim == 1
    mask = np.logical_not(group_idx == -1)
    empty = np.all(~mask) or np.prod(to_group.shape) == 0
    if empty:
        print("empty!")

    if expected_groups is not None:
        results = {"groups": np.array(expected_groups)}
    else:
        if empty:
            results = {"groups": np.array([np.nan])}
        else:
            sortidx = np.argsort(groups)
            results = {"groups": groups[sortidx]}

    final_array_shape += results["groups"].shape
    final_groups_shape += results["groups"].shape

    ic(expected_groups, results["groups"])

    for reduction in func:
        if empty:
            result = np.full(shape=final_array_shape, fill_value=fill_values[reduction])
            ic("empty", result.shape)
        else:
            result = npg.aggregate_numpy.aggregate(
                group_idx[..., mask],
                array[..., mask],
                axis=-1,
                func=reduction,
                size=size,
            )
            # ic(result.shape, final_array_shape)
            if offset_group:
                result = result.reshape(*final_array_shape[:-1], N)
            if expected_groups is not None:
                result = reindex_(
                    result, groups, expected_groups, fill_value=fill_values[reduction]
                )
            else:
                result = result[..., sortidx]
            result = result.reshape(final_array_shape)
        results[reduction] = result
    results["groups"] = np.broadcast_to(results["groups"], final_groups_shape)
    # ic(result, results["groups"])
    # ic(result.shape, final_array_shape, results["groups"].shape)

    return results


def _squeeze_results(results, func, axis):
    # at the end we squeeze out extra dims
    assert isinstance(axis, Iterable)
    groups = results["groups"]
    results["groups"] = np.squeeze(
        groups, axis=tuple(ax for ax in range(groups.ndim - 1) if groups.shape[ax] == 1)
    )
    squeeze_ax = tuple(ax for ax in sorted(axis)[:-1] if results[func[0]].shape[ax] == 1)
    for reduction in func:
        result = results[reduction]
        ic(axis, result.shape, np.squeeze(result).shape)
        results[reduction] = np.squeeze(result, axis=squeeze_ax) if squeeze_ax else result

    return results


def _npg_aggregate(x_chunk, func, expected_groups, axis, keepdims):
    """ Final aggregation step of tree reduction"""

    results = _npg_combine(x_chunk, func, expected_groups, axis, keepdims)

    ic("agg")

    ic("should squeeze here")

    return _squeeze_results(results, func, axis)


def _npg_combine(x_chunk, func, expected_groups, axis, keepdims):
    """ Combine intermediates step of tree reduction. """
    from dask.array.core import _concatenate2, concatenate3
    from dask.utils import deepmap

    if not isinstance(x_chunk, list):
        x_chunk = [x_chunk]

    def _conc2(key):
        """ copied from dask.array.reductions.mean_combine"""
        # some magic
        mapped = deepmap(lambda x: x[key], x_chunk)
        # ic(mapped)
        return _concatenate2(mapped, axes=(-1, -2))

    groups = _conc2("groups")
    # print(groups)

    results = {}
    for reduction in func:
        _reduction = _agg_reduction.get(reduction, reduction)
        x = _conc2(reduction)
        _results = chunk_reduce(
            x, groups, func=(_reduction,), axis=axis, expected_groups=expected_groups
        )
        results.update({reduction: _results[_reduction]})

    results["groups"] = _results["groups"]
    return results


def groupby_agg(
    array: dask.array.Array,
    to_group: dask.array.Array,
    func: Union[str, Iterable[str]],
    expected_groups: Iterable = None,
    axis=None,
):
    inds = tuple(range(array.ndim))

    # set axis for _tree_reduce
    if axis is None:
        axis = tuple(array.ndim - range(to_group.ndim) - 1)
    if not isinstance(axis, Iterable):
        if axis is None:
            reduced_ndim = to_group.ndim
        else:
            reduced_ndim = 1
        axis = tuple(array.ndim - np.arange(reduced_ndim) - 1)

    # I think _tree_reduce expects this
    assert all(ax >= 0 for ax in axis)

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
    # Note: it does not make sense to interpret axis relative to
    # shape of intermediate results after the blockwise call
    reduced = dask.array.reductions._tree_reduce(
        applied,
        aggregate=partial(_npg_aggregate, func=func, expected_groups=expected_groups),
        combine=partial(_npg_combine, func=func, expected_groups=expected_groups),
        name="groupby-tree-reduce",
        dtype=array.dtype,
        axis=axis,
        keepdims=True,
        concatenate=False,
    )
    # print(reduced.__dask_graph__().keys())

    group_chunks = (len(expected_groups),) if expected_groups is not None else (np.nan,)
    output_chunks = reduced.chunks[: -len(axis)] + (group_chunks,)

    # extract results from the dict
    ochunks = tuple(range(len(chunks_v)) for chunks_v in output_chunks)
    layer = {}
    for reduction in func:
        for ochunk in itertools.product(*ochunks):
            inchunk = ochunk[:-1] + (0,) * len(axis)
            layer[(reduction, *ochunk)] = (getitem, (reduced.name, *inchunk), reduction)

    # we've used keepdims=True, so _tree_reduce preserves some dummy dimensions
    first_block = len(ochunks) * (0,)  # TODO: this may not be right
    layer[("groups", *first_block)] = (getitem, (reduced.name, *first_block), "groups")
    print(layer)

    result = {}
    if expected_groups is None:
        result["groups"] = dask.array.Array(
            HighLevelGraph.from_collections("groups", layer, dependencies=[reduced]),
            "groups",
            chunks=(group_chunks,),
            dtype=to_group.dtype,  # TODO: change appropriately
        )
    else:
        result["groups"] = np.sort(expected_groups)  # TODO: check

    dtype = {"count": int}
    for reduction in func:
        result[reduction] = dask.array.Array(
            HighLevelGraph.from_collections(reduction, layer, dependencies=[reduced]),
            reduction,
            chunks=output_chunks,
            dtype=dtype.get(reduction, array.dtype),  # TODO: change appropriately
        )
    return result


def groupby_reduce(
    array: Union[np.ndarray, dask.array.Array],
    to_group: Union[np.ndarray, dask.array.Array],
    func: Union[str, Iterable[str]],
    expected_groups: Iterable = None,
    axis=None,
) -> Mapping[str, Union[dask.array.Array, np.ndarray]]:
    """
    Dask-aware GroupBy reductions

    Parameters
    ----------
    array: numpy.ndarray, dask.array.Array
        Array to be reduced, nD
    to_group: numpy.ndarray, dask.array.Array
        Array of labels to group over. Must be aligned with `array` so that
            ``array.shape[-to_group.ndim :] == to_group.shape``
    func: str or Iterable[str]
        Single function name or an Iterable of function names
    expected_groups: (optional) Iterable
        Expected unique labels.
    axis: (optional) None or int or Iterable[int]
        If None, reduce across all dimensions of to_group
        Else, reduce across corresponding axes of array

    Returns
    -------
    dict[str, [np.ndarray, dask.array.Array]]
        Keys include ``"groups"`` and ``func``.
    """

    assert array.shape[-to_group.ndim :] == to_group.shape

    if axis is None:
        if array.ndim == to_group.ndim:
            axis = np.arange(array.ndim)
        else:
            axis = array.ndim + np.arange(-to_group.ndim, 0)
            print(axis)

    if np.isscalar(axis) and axis != -1 and axis != array.ndim - 1:
        ic("swapping axes")
        to_group = np.swapaxes(to_group, axis, -1)
        array = np.swapaxes(array, axis, -1)
        axis = array.ndim - 1

    if np.isscalar(axis):
        axis = (axis,)

    rewrite_func = {"mean": (("sum", "count"))}

    if isinstance(func, str):
        func = (func,)

    if expected_groups is None and isinstance(to_group, np.ndarray):
        expected_groups = np.unique(to_group)
        expected_groups = expected_groups[~np.isnan(expected_groups)]

    # TODO: make sure expected_groups is unique
    if len(axis) == 1 and to_group.ndim > 1 and expected_groups is None:
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

    if not isinstance(array, dask.array.Array) and not isinstance(to_group, dask.array.Array):
        results = chunk_reduce(array, to_group, func=func, axis=axis, expected_groups=expected_groups)  # type: ignore
        return _squeeze_results(results, func, axis)

    # import IPython; IPython.core.debugger.set_trace()
    # when axis is a tuple
    # collapse and move reduction dimensions to the end
    if expected_groups is None and isinstance(axis, Iterable) and len(axis) <= array.ndim:
        raise NotImplementedError
        to_group = _collapse_and_reshape(to_group, -array.ndim + np.array(axis) + to_group.ndim)
        array = _collapse_and_reshape(array, axis)
        axis = array.ndim - 1

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

    # TODO: deal with NaNs and fill_values here if there are labels in
    # expected_groups that are missing
    return result
