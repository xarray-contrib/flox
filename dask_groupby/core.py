import copy
import itertools
import operator
from functools import partial
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import dask.array
import numpy as np
import numpy_groupies as npg
import pandas as pd
import xarray as xr
from dask.array.core import normalize_chunks
from dask.highlevelgraph import HighLevelGraph

from . import aggregations
from .aggregations import Aggregation, _get_fill_value

IntermediateDict = Dict[Union[str, Callable], Any]
FinalResultsDict = Dict[str, Union[dask.array.Array, np.ndarray]]


def _maybe_sub_inf(array):
    if np.issubdtype(array.dtype, np.floating):
        finfo = np.finfo(array.dtype)
        array[array == finfo.max] = np.inf
        array[array == finfo.min] = -np.inf
    return array


def _get_chunk_reduction(reduction_type: str) -> Callable:
    if reduction_type == "reduce":
        return chunk_reduce
    elif reduction_type == "argreduce":
        return chunk_argreduce
    else:
        raise ValueError(f"Unknown reduction type: {reduction_type}")


def _move_reduce_dims_to_end(arr: np.ndarray, axis: Sequence) -> np.ndarray:
    """ Transpose `arr` by moving `axis` to the end."""
    axis = tuple(axis)
    order = tuple(ax for ax in np.arange(arr.ndim) if ax not in axis) + axis
    arr = arr.transpose(order)
    return arr


def _collapse_axis(arr: np.ndarray, naxis: int) -> np.ndarray:
    """ Reshape so that the last `naxis` axes are collapsed to one axis."""
    newshape = arr.shape[:-naxis] + (np.prod(arr.shape[-naxis:]),)
    return arr.reshape(newshape)


def reindex_(array: np.ndarray, from_, to, fill_value=0, axis: int = -1) -> np.ndarray:

    assert axis in (0, -1)
    idx = np.array(
        [np.argwhere(np.array(from_) == label)[0, 0] if label in from_ else -1 for label in to]
    )
    indexer = [slice(None, None)] * array.ndim
    indexer[axis] = idx  # type: ignore
    reindexed = array[tuple(indexer)]
    if any(idx == -1):
        if axis == 0:
            loc = (idx == -1, ...)
        else:
            loc = (..., idx == -1)
        reindexed[loc] = fill_value
    return reindexed


def offset_labels(labels: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Offset group labels by dimension. This is used for "group_over" functionality, where
    we reduce over a subset of the dimensions of to_group. It assumes that the reductions
    dimensions have been flattened in the last dimension
    Copied from https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
    """
    ngroups = labels.max() + 1
    offset = (
        labels + np.arange(np.prod(labels.shape[:-1])).reshape((*labels.shape[:-1], -1)) * ngroups
    )
    # -1 indicates NaNs. preserve these otherwise we aggregate in the wrong groups!
    offset[labels == -1] = -1
    # print("N =", N, "offset = ", offset)
    size = np.prod(labels.shape[:-1]) * ngroups
    return offset, ngroups, size


def chunk_argreduce(
    array_plus_idx: Tuple[np.ndarray, ...],
    to_group: np.ndarray,
    func: Sequence[str],
    expected_groups: Optional[Union[Sequence, np.ndarray]],
    axis: Union[int, Sequence[int]],
    fill_value: Mapping[Union[str, Callable], Any],
) -> IntermediateDict:
    """
    Per-chunk arg reduction.

    Expects a tuple of (array, index along reduction axis). Inspired by
    dask.array.reductions.argtopk
    """
    array, idx = array_plus_idx

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
    func: Union[str, Callable, Sequence[str], Sequence[Callable]],
    expected_groups: Union[Sequence, np.ndarray] = None,
    axis: Union[int, Sequence[int]] = None,
    fill_value: Mapping[Union[str, Callable], Any] = None,
) -> IntermediateDict:
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
    func: str or Callable or Sequence[str] or Sequence[Callable]
        Name of reduction or function, passed to numpy_groupies.
        Supports multiple reductions.
    axis: (optional) int or Sequence[int]
        If None, reduce along all dimensions of array.
        Else reduce along specified axes.

    Returns
    -------
    dict
    """

    if isinstance(func, str) or callable(func):
        func = (func,)  # type: ignore

    func: Union[Sequence[str], Sequence[Callable]]

    if fill_value is None:
        fill_value = {f: None for f in func}

    # ic(array, to_group, axis)
    nax = len(axis) if isinstance(axis, Sequence) else to_group.ndim
    final_array_shape = array.shape[:-nax] + (1,) * (nax - 1)
    final_groups_shape = (1,) * (nax - 1)
    # ic(array.shape, to_group.shape, axis, final_array_shape, final_groups_shape)

    if isinstance(axis, Sequence) and len(axis) == 1:
        axis = next(iter(axis))

    # when axis is a tuple
    # collapse and move reduction dimensions to the end
    if isinstance(axis, Sequence) and len(axis) < to_group.ndim:
        # ic("collapsing and reshaping")
        to_group = _collapse_axis(to_group, len(axis))
        array = _collapse_axis(array, len(axis))
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
        # Not reducing along all dimensions of to_group
        offset_group = True
        group_idx, N, size = offset_labels(group_idx.reshape(to_group.shape))
        group_idx = group_idx.ravel()

    # always reshape to 1D along group dimensions
    newshape = array.shape[: array.ndim - to_group.ndim] + (np.prod(array.shape[-to_group.ndim :]),)
    array = array.reshape(newshape)

    assert group_idx.ndim == 1
    mask = np.logical_not(group_idx == -1)
    empty = np.all(~mask) or np.prod(to_group.shape) == 0
    # numpy_groupies cannot deal with group_idx = -1
    # so we'll add use (ngroups+1) as the sentinel
    # note we cannot simply remove the NaN locations;
    # that would mess up argmax, argmin
    # we could set na_sentinel in pd.factorize, but we don't know
    # what to set it to yet.
    group_idx[group_idx == -1] = group_idx.max() + 1

    results: IntermediateDict = {"groups": [], "intermediates": []}
    if expected_groups is not None:
        results["groups"] = np.array(expected_groups)
    else:
        if empty:
            results["groups"] = np.array([np.nan])
        else:
            sortidx = np.argsort(groups)
            results["groups"] = groups[sortidx]

    final_array_shape += results["groups"].shape
    final_groups_shape += results["groups"].shape

    for reduction in func:
        if empty:
            result = np.full(shape=final_array_shape, fill_value=fill_value[reduction])
            # ic("empty", result.shape)
        else:
            result = npg.aggregate_numpy.aggregate(
                group_idx,
                array,
                axis=-1,
                func=reduction,
                size=size,
            )
            if np.any(~mask):
                # remove NaN group label which should be last
                result = result[..., :-1]
            if offset_group:
                result = result.reshape(*final_array_shape[:-1], N)
            if expected_groups is not None:
                result = reindex_(result, groups, expected_groups, fill_value=fill_value[reduction])
            else:
                result = result[..., sortidx]
            result = result.reshape(final_array_shape)
        results["intermediates"].append(result)
    results["groups"] = np.broadcast_to(results["groups"], final_groups_shape)
    return results


def _squeeze_results(results: IntermediateDict, axis: Sequence) -> IntermediateDict:
    # at the end we squeeze out extra dims
    groups = results["groups"]
    newresults: IntermediateDict = {"groups": [], "intermediates": []}
    newresults["groups"] = np.squeeze(
        groups, axis=tuple(ax for ax in range(groups.ndim - 1) if groups.shape[ax] == 1)
    )
    for v in results["intermediates"]:
        squeeze_ax = tuple(ax for ax in sorted(axis)[:-1] if v.shape[ax] == 1)
        # ic(axis, squeeze_ax, v.shape, np.squeeze(v, axis=squeeze_ax).shape)
        newresults["intermediates"].append(np.squeeze(v, axis=squeeze_ax) if squeeze_ax else v)
    return newresults


def _split_groups(array, j, slicer):
    """ Slices out chunks when split_out > 1"""
    results = {"groups": array["groups"][..., slicer]}
    results["intermediates"] = [v[..., slicer] for v in array["intermediates"]]
    return results


def _npg_aggregate(
    x_chunk,
    agg: Aggregation,
    expected_groups: Union[Sequence, np.ndarray],
    axis: Sequence,
    keepdims,
    group_ndim: int,
) -> FinalResultsDict:
    """ Final aggregation step of tree reduction"""
    results = _npg_combine(x_chunk, agg, expected_groups, axis, keepdims, group_ndim)
    squeezed = _squeeze_results(results, axis)

    # finalize step
    result: Dict[str, Union[dask.array.Array, np.ndarray]] = {"groups": squeezed["groups"]}
    result[agg.name] = agg.finalize(*squeezed["intermediates"])

    if agg.name in ["max", "min"]:
        # Work aroung npg bug where we get finfo.max, finfo.min
        # instead of np.inf, -np.inf
        result[agg.name] = _maybe_sub_inf(result[agg.name])
    return result


def _npg_combine(
    x_chunk,
    agg: Aggregation,
    expected_groups: Union[Sequence, np.ndarray],
    axis: Sequence,
    keepdims,
    group_ndim: int,
) -> IntermediateDict:
    """ Combine intermediates step of tree reduction. """
    from dask.array.core import _concatenate2
    from dask.utils import deepmap

    if not isinstance(x_chunk, list):
        x_chunk = [x_chunk]

    def _conc2(key1, key2=None, axis=None) -> np.ndarray:
        """ copied from dask.array.reductions.mean_combine"""
        # some magic
        if key2 is not None:
            mapped = deepmap(lambda x: x[key1][key2], x_chunk)
        else:
            mapped = deepmap(lambda x: x[key1], x_chunk)
        return _concatenate2(mapped, axes=axis)

    group_conc_axis: Iterable[int]
    if group_ndim == 1:
        group_conc_axis = (0,)
    else:
        group_conc_axis = sorted(group_ndim - ax - 1 for ax in axis)
    groups = _conc2("groups", axis=group_conc_axis)

    if agg.reduction_type == "argreduce":
        # We need to send the intermediate array values & indexes at the same time
        array_plus_idx = tuple(_conc2(key1="intermediates", key2=idx, axis=axis) for idx in (0, 1))
        results = chunk_argreduce(
            array_plus_idx,
            groups,
            func=agg.combine,
            axis=axis,
            expected_groups=expected_groups,
            fill_value=agg.fill_value,
        )

    elif agg.reduction_type == "reduce":
        # Here we reduce the intermediates individually
        results = {"groups": None, "intermediates": []}
        for idx, combine in enumerate(agg.combine):
            array = _conc2(key1="intermediates", key2=idx, axis=axis)
            # ic(combine, x)
            _results = chunk_reduce(
                array,
                groups,
                func=combine,
                axis=axis,
                expected_groups=expected_groups,
                fill_value=agg.fill_value,
            )
            results["intermediates"].append(*_results["intermediates"])
            results["groups"] = _results["groups"]
    return results


def groupby_agg(
    array: dask.array.Array,
    to_group: dask.array.Array,
    agg: Aggregation,
    expected_groups: Optional[Union[Sequence, np.ndarray]],
    axis: Sequence = None,
    split_out: int = 1,
) -> FinalResultsDict:

    # I think _tree_reduce expects this
    assert isinstance(axis, Sequence)
    assert all(ax >= 0 for ax in axis)

    inds = tuple(range(array.ndim))
    name = f"groupby_{agg.name}"

    # preprocess the array
    if agg.preprocess:
        # This is necessary for argreductions.
        # We need to rechunk before zipping up with the index
        _, (array, to_group) = dask.array.unify_chunks(
            array, inds, to_group, inds[-to_group.ndim :]
        )
        align_arrays = False
        array = agg.preprocess(array, axis=axis)
    else:
        align_arrays = True

    # apply reduction on chunk
    applied = dask.array.blockwise(
        partial(
            _get_chunk_reduction(agg.reduction_type),
            func=agg.chunk,  # type: ignore
            axis=axis,
            expected_groups=expected_groups,
            fill_value=agg.fill_value,
        ),
        inds,
        array,
        inds,
        to_group,
        inds[-to_group.ndim :],
        concatenate=False,
        dtype=array.dtype,
        meta=array._meta,
        align_arrays=align_arrays,
        name=f"{name}-chunk",
    )

    if split_out > 1:
        if expected_groups is None:
            raise NotImplementedError
        chunk_tuples = tuple(itertools.product(*tuple(range(n) for n in applied.numblocks)))
        ngroups = len(expected_groups)
        group_chunks = normalize_chunks(np.ceil(ngroups / split_out), (ngroups,))[0]
        idx = tuple(np.cumsum((0,) + group_chunks))

        # split each block into `split_out` chunks
        dsk = {}
        for i in chunk_tuples:
            for j in range(split_out):
                dsk[("split", *i, j)] = (
                    _split_groups,
                    (applied.name, *i),
                    j,
                    slice(idx[j], idx[j + 1]),
                )

        # now construct an array that can be passed to _tree_reduce
        intergraph = HighLevelGraph.from_collections("split", dsk, dependencies=(applied,))
        intermediate = dask.array.Array(
            intergraph, name="split", chunks=applied.chunks + ((1,) * split_out,), meta=array._meta
        )
        # We don't want the reindexing step in this case.
        expected_ = None

    else:
        intermediate = applied
        group_chunks = (len(expected_groups),) if expected_groups is not None else (np.nan,)
        expected_ = expected_groups

    # reduced is really a dict mapping reduction name to array
    # and "groups" to an array of group labels
    # Note: it does not make sense to interpret axis relative to
    # shape of intermediate results after the blockwise call
    reduced = dask.array.reductions._tree_reduce(
        intermediate,
        aggregate=partial(
            _npg_aggregate, agg=agg, expected_groups=expected_, group_ndim=to_group.ndim
        ),
        combine=partial(_npg_combine, agg=agg, expected_groups=expected_, group_ndim=to_group.ndim),
        name=name,
        dtype=array.dtype,
        axis=axis,
        keepdims=True,
        concatenate=False,
    )

    # if split_out > 1:
    #     inds = tuple(range(reduced.ndim))
    #     reduced = dask.array.blockwise(
    #         lambda x: x[0],
    #         inds[:-2] + (inds[-1],),
    #         reduced,
    #         inds,
    #         name="squeeze",
    #         dtype=array.dtype,
    #     )

    output_chunks = reduced.chunks[: -(len(axis) + int(split_out > 1))] + (group_chunks,)
    print(output_chunks)

    def _getitem(d, key1, key2):
        return d[key1][key2]

    # extract results from the dict
    result: Dict = {}
    layer: Dict[Tuple, Tuple] = {}
    ochunks = tuple(range(len(chunks_v)) for chunks_v in output_chunks)
    if expected_groups is None:
        # we've used keepdims=True, so _tree_reduce preserves some dummy dimensions
        first_block = len(ochunks) * (0,)  # TODO: this may not be right
        layer[("groups", *first_block)] = (operator.getitem, (reduced.name, *first_block), "groups")
        result["groups"] = dask.array.Array(
            HighLevelGraph.from_collections("groups", layer, dependencies=[reduced]),
            "groups",
            chunks=(group_chunks,),
            dtype=to_group.dtype,
        )
    else:
        result["groups"] = np.sort(expected_groups)  # TODO: check

    layer: Dict[Tuple, Tuple] = {}  # type: ignore
    for ochunk in itertools.product(*ochunks):
        inchunk = ochunk[:-1] + (0,) * (len(axis)) + (ochunk[-1],) * int(split_out > 1)
        layer[(name, *ochunk)] = (
            operator.getitem,
            (reduced.name, *inchunk),
            agg.name,
        )
    result[agg.name] = dask.array.Array(
        HighLevelGraph.from_collections(name, layer, dependencies=[reduced]),
        name,
        chunks=output_chunks,
        dtype=agg.dtype if agg.dtype else array.dtype,
    )

    return result


def groupby_reduce(
    array: Union[np.ndarray, dask.array.Array],
    to_group: Union[np.ndarray, dask.array.Array],
    func: Union[str, Aggregation],
    expected_groups: Union[Sequence, np.ndarray] = None,
    axis=None,
    fill_value=None,
    split_out=1,
) -> FinalResultsDict:
    """
    GroupBy reductions using tree reductions for dask.array

    Parameters
    ----------
    array: numpy.ndarray, dask.array.Array
        Array to be reduced, nD
    to_group: numpy.ndarray, dask.array.Array
        Array of labels to group over. Must be aligned with `array` so that
            ``array.shape[-to_group.ndim :] == to_group.shape``
    func: str or Aggregation
        Single function name or an Aggregation instance
    expected_groups: (optional) Sequence
        Expected unique labels.
    axis: (optional) None or int or Sequence[int]
        If None, reduce across all dimensions of to_group
        Else, reduce across corresponding axes of array
        Negative integers are normalized using array.ndim
    fill_value: Any
        Value when a label in `expected_groups` is not present
    split_out: int, optional
        Number of chunks along group axis in output (last axis)

    Returns
    -------
    dict[str, [np.ndarray, dask.array.Array]]
        Keys include ``"groups"`` and ``func``.
    """

    assert array.shape[-to_group.ndim :] == to_group.shape

    if axis is None:
        axis = tuple(array.ndim + np.arange(-to_group.ndim, 0))
    else:
        axis = np.core.numeric.normalize_axis_tuple(axis, array.ndim)  # type: ignore

    if expected_groups is None and isinstance(to_group, np.ndarray):
        expected_groups = np.unique(to_group)
        if np.issubdtype(expected_groups.dtype, np.floating):  # type: ignore
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

    if isinstance(axis, Sequence) and len(axis) < to_group.ndim:
        to_group = _move_reduce_dims_to_end(to_group, -array.ndim + np.array(axis) + to_group.ndim)
        array = _move_reduce_dims_to_end(array, axis)
        axis = tuple(array.ndim + np.arange(-len(axis), 0))

    if not isinstance(func, Aggregation):
        try:
            # TODO: need better interface
            # we set dtype, fillvalue on reduction later. so deepcopy now
            reduction = copy.deepcopy(getattr(aggregations, func))
        except AttributeError:
            raise NotImplementedError(f"Reduction {func!r} not implemented yet")
    else:
        reduction = func

    # Replace sentinel fill values according to dtype
    if reduction.dtype is None:
        reduction.dtype = array.dtype
    reduction.fill_value = {
        k: _get_fill_value(array.dtype, v) for k, v in reduction.fill_value.items()
    }
    if not isinstance(array, dask.array.Array) and not isinstance(to_group, dask.array.Array):
        results = chunk_reduce(
            array,
            to_group,
            func=reduction.name,
            axis=axis,
            expected_groups=expected_groups,
            fill_value=reduction.fill_value,
        )  # type: ignore
        intermediate = _squeeze_results(results, axis)
        result: FinalResultsDict = {"groups": intermediate["groups"]}
        result[reduction.name] = intermediate["intermediates"][0]

        if reduction.name in ["argmin", "argmax"]:
            # TODO: Fix npg bug where argmax with nD array, 1D group_idx, axis=-1
            # will return wrong indices
            result[reduction.name] = np.unravel_index(result[reduction.name], array.shape)[-1]
    else:
        if func in ["first", "last"]:
            raise NotImplementedError("first, last not implemented for dask arrays")

        # Needed since we need not have equal number of groups per block
        if expected_groups is None and len(axis) > 1:
            to_group = _collapse_axis(to_group, len(axis))
            array = _collapse_axis(array, len(axis))
            axis = (array.ndim - 1,)

        # TODO: deal with mixed array kinds (numpy + dask; dask + numpy)
        result = groupby_agg(
            array, to_group, reduction, expected_groups, axis=axis, split_out=split_out
        )

    return result


def xarray_reduce(
    groupby: xr.core.groupby.GroupBy,
    func: Union[str, Aggregation],
    split_out=1,
):
    def wrapper(*args, **kwargs):
        result = groupby_reduce(*args, **kwargs)
        # TODO: how do we return groups here
        return tuple(result.values())[1]

    expected_groups = list(groupby.groups.keys())
    outdim = groupby._unique_coord.name
    groupdim = groupby._group_dim

    actual = xr.apply_ufunc(
        wrapper,
        groupby._obj,
        groupby._group,
        input_core_dims=[[groupdim], [groupdim]],
        dask="allowed",
        output_core_dims=[[outdim]],  # TODO: return groups
        dask_gufunc_kwargs=dict(output_sizes={outdim: len(expected_groups)}),
        kwargs={"func": func, "axis": -1, "split_out": split_out},
    )
    actual[outdim] = groupby._unique_coord

    return actual


def xarray_groupby_reduce(
    groupby: xr.core.groupby.GroupBy,
    func: Union[str, Aggregation],
    split_out=1,
):
    def wrapper(*args, **kwargs):
        result = groupby_reduce(*args, **kwargs)
        # TODO: how do we return groups here
        return tuple(result.values())

    expected_groups = list(groupby.groups.keys())
    outdim = groupby._unique_coord.name
    groupdim = groupby._group_dim
    indims = groupby._obj.dims
    result_dims = tuple(dim for dim in indims if dim != groupdim) + (outdim,)
    # input_dims = groupby._obj.dims

    groups, actual = xr.apply_ufunc(
        wrapper,
        groupby._obj,
        groupby._group,
        input_core_dims=[indims, [groupdim]],
        dask="allowed",
        output_core_dims=[[outdim], result_dims],  # TODO: return groups
        dask_gufunc_kwargs=dict(output_sizes={outdim: len(expected_groups)}),
        kwargs={"func": func, "axis": -1, "split_out": split_out},
    )
    actual[outdim] = groups

    return actual
