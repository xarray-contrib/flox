import itertools
from typing import TYPE_CHECKING, Dict, Iterable, Sequence, Tuple, Union

import dask
import numpy as np
import numpy_groupies as npg
import xarray as xr

from .aggregations import Aggregation, _atleast_1d
from .core import factorize_, groupby_reduce, reindex_

if TYPE_CHECKING:
    from xarray import DataArray, Dataset, GroupBy, Resample


def xarray_reduce(
    obj: Union["Dataset", "DataArray"],
    *by: Union["DataArray", Iterable[str], Iterable["DataArray"]],
    func: Union[str, Aggregation],
    expected_groups: Dict[str, Sequence] = None,
    bins=None,
    dim=None,
    split_out=1,
    fill_value=None,
    blockwise=False,
    keep_attrs: bool = True,
):

    # TODO: handle this _DummyGroup stuff when dispatching from xarray
    from xarray.core.groupby import _DummyGroup

    unindexed_dims = tuple(b.name for b in by if isinstance(b, _DummyGroup))
    by = tuple(b.name if isinstance(b, _DummyGroup) else b for b in by)

    by: Tuple["DataArray"] = tuple(obj[g] if isinstance(g, str) else g for g in by)  # type: ignore

    if len(by) > 1 and any(dask.is_dask_collection(by_) for by_ in by):
        raise ValueError("Grouping by multiple variables will call compute dask variables.")

    grouper_dims = set(itertools.chain(*tuple(g.dims for g in by)))

    if isinstance(obj, xr.DataArray):
        ds = obj._to_temp_dataset()
    else:
        ds = obj

    if dim is Ellipsis:
        dim = obj.dims

    # we broadcast all variables against each other along all dimensions in `by` variables
    # we avoid excluding `dim` because it need not be a dimension in any of the `by` variables!
    exclude_dims = set(ds.dims) - grouper_dims
    if dim is not None:
        exclude_dims -= set(dim)
    ds, *by = xr.broadcast(ds, *by, exclude=exclude_dims)

    if dim is None:
        dim = by[0].dims
    else:
        dim = _atleast_1d(dim)

    axis = tuple(range(-len(dim), 0))

    group_names = tuple(g.name for g in by)
    # ds = ds.drop_vars(tuple(g for g in group_names))

    if len(by) > 1:
        group_idx, expected_groups, group_shape, _, _, _ = factorize_(
            tuple(g.data for g in by), expected_groups, bins
        )
        to_group = xr.DataArray(group_idx, dims=dim, coords={d: by[0][d] for d in by[0].indexes})
    else:
        if expected_groups is None and isinstance(by[0].data, np.ndarray):
            expected_groups = (np.unique(by[0].data),)
        if expected_groups is None:
            raise NotImplementedError(
                "Please provided expected_groups if not grouping by a numpy-backed DataArray"
            )
        group_shape = (len(expected_groups[0]),)
        to_group = by[0]

    group_sizes = dict(zip(group_names, group_shape))

    def wrapper(*args, **kwargs):
        result, groups = groupby_reduce(*args, **kwargs)
        if len(by) > 1:
            # all groups need not be present. reindex here
            # TODO: add test
            reindexed = reindex_(
                result,
                from_=groups,
                to=np.arange(np.prod(group_shape)),
                fill_value=fill_value,
                axis=-1,
            )
            result = reindexed.reshape(result.shape[:-1] + group_shape)
        return result

    # These data variables do not have the core dimension,
    # take them out to prevent errors.
    # apply_ufunc can handle non-dim coordinate variables without core dimensions
    missing_dim = {}
    for k, v in ds.data_vars.items():
        is_missing_dim = not (all(d in v.dims for d in dim))
        if is_missing_dim:
            missing_dim[k] = v

    actual = xr.apply_ufunc(
        wrapper,
        ds.drop_vars(tuple(missing_dim)),
        to_group,
        input_core_dims=[dim, dim],
        # for xarray's test_groupby_duplicate_coordinate_labels
        exclude_dims=set(dim),
        output_core_dims=[group_names],
        dask="allowed",
        dask_gufunc_kwargs=dict(output_sizes=group_sizes),
        keep_attrs=keep_attrs,
        kwargs={
            "func": func,
            "axis": axis,
            "split_out": split_out,
            "fill_value": fill_value,
            "blockwise": blockwise,
        },
    )

    for name, expect in zip(group_names, expected_groups):
        if isinstance(actual, xr.Dataset) and name in actual:
            actual = actual.drop_vars(name)
        actual[name] = expect

    if missing_dim:
        actual = actual.update(missing_dim)

    if unindexed_dims:
        actual = actual.drop_vars(unindexed_dims)

    if isinstance(obj, xr.DataArray):
        return obj._from_temp_dataset(actual)
    else:
        return actual


def xarray_groupby_reduce(
    groupby: "GroupBy",
    func: Union[str, Aggregation],
    split_out=1,
    blockwise=False,
    keep_attrs: bool = True,
):
    """Apply on an existing Xarray groupby object for convenience."""

    def wrapper(*args, **kwargs):
        result, _ = groupby_reduce(*args, **kwargs)
        return result

    groups = list(groupby.groups.keys())
    outdim = groupby._unique_coord.name
    groupdim = groupby._group_dim

    actual = xr.apply_ufunc(
        wrapper,
        groupby._obj,
        groupby._group,
        input_core_dims=[[groupdim], [groupdim]],
        # for xarray's test_groupby_duplicate_coordinate_labels
        exclude_dims=set(groupdim),
        output_core_dims=[[outdim]],
        dask="allowed",
        dask_gufunc_kwargs=dict(output_sizes={outdim: len(groups)}),
        keep_attrs=keep_attrs,
        kwargs={
            "func": func,
            "axis": -1,
            "split_out": split_out,
            "expected_groups": groups,
            "blockwise": blockwise,
        },
    )
    actual[outdim] = groups

    return actual


def _get_optimal_chunks_for_groups(chunks, labels):
    chunkidx = np.cumsum(chunks) - 1
    # what are the groups at chunk boundaries
    labels_at_chunk_bounds = np.unique(labels[chunkidx])
    # what's the last index of all groups
    last_indexes = npg.aggregate_numpy.aggregate(labels, np.arange(len(labels)), func="last")
    # what's the last index of groups at the chunk boundaries.
    lastidx = last_indexes[labels_at_chunk_bounds]

    if len(chunkidx) == len(lastidx) and (chunkidx == lastidx).all():
        return chunks

    first_indexes = npg.aggregate_numpy.aggregate(labels, np.arange(len(labels)), func="first")
    firstidx = first_indexes[labels_at_chunk_bounds]

    newchunkidx = [0]
    for c, f, l in zip(chunkidx, firstidx, lastidx):
        Δf = abs(c - f)
        Δl = abs(c - l)
        if c == 0 or newchunkidx[-1] > l:
            continue
        if Δf < Δl and f > newchunkidx[-1]:
            newchunkidx.append(f)
        else:
            newchunkidx.append(l + 1)
    if newchunkidx[-1] != chunkidx[-1] + 1:
        newchunkidx.append(chunkidx[-1] + 1)
    newchunks = np.diff(newchunkidx)

    assert sum(newchunks) == sum(chunks)
    return tuple(newchunks)


def rechunk_to_group_boundaries(array, dim, labels):
    """
    Rechunks array so that group boundaries line up with chunk boundaries, allowing
    parallel group reductions.

    This only works when the groups are sequential (e.g. labels = [0,0,0,1,1,1,1,2,2]).
    Such patterns occur when using ``.resample``.
    """
    axis = array.get_axis_num(dim)
    chunks = array.chunks[axis]
    newchunks = _get_optimal_chunks_for_groups(chunks, labels.data)
    if newchunks == chunks:
        return array
    else:
        return array.chunk({dim: newchunks})


def resample_reduce(
    resampler: "Resample",
    func,
    keep_attrs: bool = True,
):

    obj = resampler._obj
    dim = resampler._group_dim

    # this creates a label DataArray since resample doesn't do that somehow
    tostack = []
    for idx, slicer in enumerate(resampler._group_indices):
        if slicer.stop is None:
            stop = resampler._obj.sizes[resampler._group_dim]
        else:
            stop = slicer.stop
        tostack.append(idx * np.ones((stop - slicer.start,), dtype=np.int32))
    by = xr.DataArray(np.hstack(tostack), dims=(dim,), name="__resample_dim__")

    if isinstance(obj, xr.Dataset):
        for var in obj:
            if obj[var].chunks is not None:
                obj[var] = rechunk_to_group_boundaries(obj[var], dim, by)
    else:
        if obj.chunks is not None:
            obj = rechunk_to_group_boundaries(obj, dim, by)

    result = xarray_reduce(
        obj,
        by,
        func=func,
        blockwise=True,
        expected_groups=(resampler._unique_coord.data,),
        keep_attrs=keep_attrs,
    ).rename({"__resample_dim__": dim})
    return result
