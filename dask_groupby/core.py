import itertools
import operator
from functools import partial
from operator import getitem
from typing import Any, Callable, Iterable, Mapping, Tuple, Union

import dask.array
import numpy as np
import numpy_groupies as npg
import pandas as pd
from dask.highlevelgraph import HighLevelGraph
from icecream import ic

from . import aggregations
from .aggregations import Aggregation


def _get_chunk_reduction(reduction_type):
    if reduction_type == "reduce":
        return chunk_reduce
    elif reduction_type == "argreduce":
        return chunk_argreduce
    else:
        raise ValueError(f"Unknown reduction type: {reduction_type}")


def _move_reduce_dims_to_end(arr, axis):
    axis = tuple(axis)
    order = tuple(ax for ax in np.arange(arr.ndim) if ax not in axis) + axis
    # ic(order)
    arr = arr.transpose(order)
    return arr


def _collapse_axis(arr: np.ndarray, axis: Iterable[int]):
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


def chunk_argreduce(
    array_plus_idx: Tuple[np.ndarray, np.ndarray],
    to_group: np.ndarray,
    func: Tuple[str, Callable],
    expected_groups: Iterable = None,
    axis: Union[int, Iterable[int]] = None,
    fill_value: Mapping[str, Any] = None,
):
    array, idx = array_plus_idx
    fill_value = {k: v[idx] for idx, (k, v) in enumerate(fill_value.items())}

    results = chunk_reduce(array, to_group, func, expected_groups, axis, fill_value)

    # glorious
    newidx = np.broadcast_to(idx, array.shape)[
        np.unravel_index(results["intermediates"][1], array.shape)
    ]
    results["intermediates"][1] = newidx
    return results


def chunk_reduce(
    array: np.ndarray,
    to_group: np.ndarray,
    func: Tuple[str, Callable],
    expected_groups: Iterable = None,
    axis: Union[int, Iterable[int]] = None,
    fill_value: Mapping[str, Any] = None,
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

    if fill_value is None:
        fill_value = {f: None for f in func}

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
    if isinstance(axis, Iterable) and len(axis) < to_group.ndim:
        ic("collapsing and reshaping")
        to_group = _collapse_axis(to_group, -array.ndim + np.array(axis) + to_group.ndim)
        array = _collapse_axis(array, axis)
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
        ic("offsetting because", axis, to_group.ndim)
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

    # ic(expected_groups, results["groups"])

    results["intermediates"] = []
    for reduction in func:
        assert isinstance(reduction, (str, Callable))
        if empty:
            result = np.full(shape=final_array_shape, fill_value=fill_value[reduction])
            ic("empty", result.shape)
        else:
            result = npg.aggregate_numpy.aggregate(
                group_idx[..., mask],
                array[..., mask],
                axis=-1,
                func=reduction,
                size=size,
            )
            if offset_group:
                result = result.reshape(*final_array_shape[:-1], N)
            if expected_groups is not None:
                result = reindex_(result, groups, expected_groups, fill_value=fill_value[reduction])
            else:
                result = result[..., sortidx]
            result = result.reshape(final_array_shape)
        results["intermediates"].append(result)
    results["groups"] = np.broadcast_to(results["groups"], final_groups_shape)
    # ic(results, results["groups"])
    # ic(result.shape, final_array_shape, results["groups"].shape)

    return results


def _squeeze_results(results, axis):
    # at the end we squeeze out extra dims
    assert isinstance(axis, Iterable)
    groups = results["groups"]
    newresults = {"groups": None, "intermediates": []}
    newresults["groups"] = np.squeeze(
        groups, axis=tuple(ax for ax in range(groups.ndim - 1) if groups.shape[ax] == 1)
    )
    for v in results["intermediates"]:
        squeeze_ax = tuple(ax for ax in sorted(axis)[:-1] if v.shape[ax] == 1)
        ic(axis, squeeze_ax, v.shape, np.squeeze(v, axis=squeeze_ax).shape)
        newresults["intermediates"].append(np.squeeze(v, axis=squeeze_ax) if squeeze_ax else v)

    return newresults


def _npg_aggregate(x_chunk, func, expected_groups, axis, keepdims, group_ndim):
    """ Final aggregation step of tree reduction"""
    results = _npg_combine(x_chunk, func, expected_groups, axis, keepdims, group_ndim)
    return _squeeze_results(results, axis)


def _npg_combine(x_chunk, agg, expected_groups, axis, keepdims, group_ndim):
    """ Combine intermediates step of tree reduction. """
    from dask.array.core import _concatenate2, concatenate3
    from dask.utils import deepmap

    if not isinstance(x_chunk, list):
        x_chunk = [x_chunk]

    def _conc2(key1, key2=None, axis=None):
        """ copied from dask.array.reductions.mean_combine"""
        # some magic
        if key2 is not None:
            mapped = deepmap(lambda x: x[key1][key2], x_chunk)
        else:
            mapped = deepmap(lambda x: x[key1], x_chunk)
        return _concatenate2(mapped, axes=axis)

    if group_ndim == 1:
        group_conc_axis = (0,)
    else:
        group_conc_axis = sorted(group_ndim - ax - 1 for ax in axis)
    groups = _conc2("groups", axis=group_conc_axis)
    # print(groups)

    if agg.reduction_type == "argreduce":
        # We need to send the intermediate array values & indexes at the same time
        x = tuple(
            _conc2(key1="intermediates", key2=idx, axis=axis) for idx in range(len(agg.combine))
        )
        results = chunk_argreduce(
            x,
            groups,
            func=agg.combine,
            axis=axis,
            expected_groups=expected_groups,
            # set the fill_value for the original reduction
            fill_value={r: agg.fill_value for r in agg.chunk},  # TODO: fix
        )

    elif agg.reduction_type == "reduce":
        # Here we reduce the intermediates individually
        results = {"groups": None, "intermediates": []}
        for idx, combine in enumerate(agg.combine):
            x = _conc2(key1="intermediates", key2=idx, axis=axis)
            # ic(combine, x)
            _results = chunk_reduce(
                x,
                groups,
                func=combine,
                axis=axis,
                expected_groups=expected_groups,
                # set the fill_value for the original reduction
                fill_value={combine: agg.fill_value},
            )
            results["intermediates"].append(*_results["intermediates"])
            results["groups"] = _results["groups"]
    return results


def groupby_agg(
    array: dask.array.Array,
    to_group: dask.array.Array,
    func: Aggregation,
    expected_groups: Iterable = None,
    axis=None,
):

    # I think _tree_reduce expects this
    assert isinstance(axis, Iterable)
    assert all(ax >= 0 for ax in axis)

    # TODO: cleanup
    assert len(func) == 1
    func = func[0]

    inds = tuple(range(array.ndim))

    # preprocess the array
    if func.preprocess:
        array = func.preprocess(array, axis=axis)

    # apply reduction on chunk
    applied = dask.array.blockwise(
        partial(
            _get_chunk_reduction(func.reduction_type),
            func=func.chunk,
            axis=axis,
            expected_groups=expected_groups,
            fill_value={r: func.fill_value for r in func.chunk},
        ),
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
        aggregate=partial(
            _npg_aggregate, func=func, expected_groups=expected_groups, group_ndim=to_group.ndim
        ),
        combine=partial(
            _npg_combine, agg=func, expected_groups=expected_groups, group_ndim=to_group.ndim
        ),
        name="groupby-tree-reduce",
        dtype=array.dtype,
        axis=axis,
        keepdims=True,
        concatenate=False,
    )

    group_chunks = (len(expected_groups),) if expected_groups is not None else (np.nan,)
    output_chunks = reduced.chunks[: -len(axis)] + (group_chunks,)

    def _getitem(d, key1, key2):
        return d[key1][key2]

    # extract results from the dict
    result = {"groups": None, "intermediates": []}
    layer = {}
    ochunks = tuple(range(len(chunks_v)) for chunks_v in output_chunks)
    if expected_groups is None:
        # we've used keepdims=True, so _tree_reduce preserves some dummy dimensions
        first_block = len(ochunks) * (0,)  # TODO: this may not be right
        layer[("groups", *first_block)] = (getitem, (reduced.name, *first_block), "groups")
        result["groups"] = dask.array.Array(
            HighLevelGraph.from_collections("groups", layer, dependencies=[reduced]),
            "groups",
            chunks=(group_chunks,),
            dtype=to_group.dtype,
        )
    else:
        result["groups"] = np.sort(expected_groups)  # TODO: check

    for idx, chunk in enumerate(func.chunk):
        layer = {}
        name = f"{func.name}_{chunk}"
        for ochunk in itertools.product(*ochunks):
            inchunk = ochunk[:-1] + (0,) * len(axis)
            layer[(name, *ochunk)] = (
                _getitem,
                (reduced.name, *inchunk),
                "intermediates",
                idx,
            )
        result["intermediates"].append(
            dask.array.Array(
                HighLevelGraph.from_collections(name, layer, dependencies=[reduced]),
                name,
                chunks=output_chunks,
                dtype=func.dtype if func.dtype else array.dtype,
            )
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
        Negative integers are normalized using array.ndim

    Returns
    -------
    dict[str, [np.ndarray, dask.array.Array]]
        Keys include ``"groups"`` and ``func``.
    """

    assert array.shape[-to_group.ndim :] == to_group.shape

    if axis is None:
        axis = array.ndim + np.arange(-to_group.ndim, 0)
    else:
        axis = np.core.numeric.normalize_axis_tuple(axis, array.ndim)

    # TODO: isscalar check
    if isinstance(func, str):
        func = (func,)

    if expected_groups is None and isinstance(to_group, np.ndarray):
        expected_groups = np.unique(to_group)
        if "U" not in expected_groups.dtype.kind:
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

    if isinstance(axis, Iterable) and len(axis) < to_group.ndim:
        to_group = _move_reduce_dims_to_end(to_group, -array.ndim + np.array(axis) + to_group.ndim)
        array = _move_reduce_dims_to_end(array, axis)
        axis = array.ndim + np.arange(-len(axis), 0)

    reductions = tuple(
        itertools.chain(*[[getattr(aggregations, reduction, reduction)] for reduction in func])
    )

    if not isinstance(array, dask.array.Array) and not isinstance(to_group, dask.array.Array):
        results = chunk_reduce(
            array,
            to_group,
            func=func,
            axis=axis,
            expected_groups=expected_groups,
            fill_value={r.name: r.fill_value for r in reductions},
        )  # type: ignore
        squeezed = _squeeze_results(results, axis)
        squeezed[func[0]] = squeezed.pop("intermediates")[0]
        return squeezed

    # Needed since we need not have equal number of groups per block
    if expected_groups is None and len(axis) > 1:
        to_group = _collapse_axis(to_group, -array.ndim + np.array(axis) + to_group.ndim)
        array = _collapse_axis(array, axis)
        axis = (array.ndim - 1,)

    intermediate = groupby_agg(array, to_group, reductions, expected_groups, axis=axis)

    # finalize step
    result = {"groups": intermediate["groups"]}
    for reduction in reductions:
        result[reduction.name] = reduction.finalize(*intermediate["intermediates"])

    return result
