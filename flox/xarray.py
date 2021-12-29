import itertools
from typing import TYPE_CHECKING, Hashable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from .aggregations import Aggregation, _atleast_1d
from .core import (
    factorize_,
    groupby_reduce,
    rechunk_for_blockwise,
    rechunk_for_cohorts as rechunk_array_for_cohorts,
    reindex_,
)
from .xrutils import is_duck_dask_array, isnull

if TYPE_CHECKING:
    from xarray import DataArray, Dataset, Resample


def _get_input_core_dims(group_names, dim, ds, to_group):
    input_core_dims = [[], []]
    for g in group_names:
        if g in dim:
            continue
        if g in ds.dims:
            input_core_dims[0].extend([g])
        if g in to_group.dims:
            input_core_dims[1].extend([g])
    input_core_dims[0].extend(dim)
    input_core_dims[1].extend(dim)
    return input_core_dims


def _restore_dim_order(result, obj, by):
    def lookup_order(dimension):
        if dimension == by.name and by.ndim == 1:
            (dimension,) = by.dims
        if dimension in obj.dims:
            axis = obj.get_axis_num(dimension)
        else:
            axis = 1e6  # some arbitrarily high value
        return axis

    new_order = sorted(result.dims, key=lookup_order)
    return result.transpose(*new_order)


def xarray_reduce(
    obj: Union["Dataset", "DataArray"],
    *by: Union["DataArray", Iterable[str], Iterable["DataArray"]],
    func: Union[str, Aggregation],
    expected_groups=None,
    isbin: Union[bool, Sequence[bool]] = False,
    sort: bool = True,
    dim: Hashable = None,
    split_out: int = 1,
    fill_value=None,
    method: str = "map-reduce",
    engine: str = "flox",
    keep_attrs: bool = True,
    skipna: Optional[bool] = None,
    min_count: Optional[int] = None,
    **finalize_kwargs,
):
    """GroupBy reduce operations on xarray objects using numpy-groupies

    Parameters
    ----------
    obj : DataArray or Dataset
        Xarray object to reduce
    *by : DataArray or iterable of str or iterable of DataArray
        Variables with which to group by ``obj``
    func : str or Aggregation
        Reduction method
    expected_groups : str or sequence
        expected group labels corresponding to each `by` variable
    isbin : iterable of bool
        If True, corresponding entry in ``expected_groups`` are bin edges.
        If False, the entry in ``expected_groups`` is treated as a simple label.
    sort : (optional), bool
        Whether groups should be returned in sorted order. Only applies for dask
        reductions when ``method`` is not `"map-reduce"`. For ``"map-reduce", the groups
        are always sorted.
    dim : hashable
        dimension name along which to reduce. If None, reduces across all
        dimensions of `by`
    split_out : int, optional
        Number of output chunks along grouped dimension in output.
    fill_value :
        Value used for missing groups in the output i.e. when one of the labels
        in ``expected_groups`` is not actually present in ``by``.
    method : {"map-reduce", "blockwise", "cohorts", "split-reduce"}, optional
        Strategy for reduction of dask arrays only:
          * ``"map-reduce"``:
            First apply the reduction blockwise on ``array``, then
            combine a few newighbouring blocks, apply the reduction.
            Continue until finalizing. Usually, ``func`` will need
            to be an Aggregation instance for this method to work.
            Common aggregations are implemented.
          * ``"blockwise"``:
            Only reduce using blockwise and avoid aggregating blocks
            together. Useful for resampling-style reductions where group
            members are always together. If  `by` is 1D,  `array` is automatically
            rechunked so that chunk boundaries line up with group boundaries
            i.e. each block contains all members of any group present
            in that block. For nD `by`, you must make sure that all members of a group
            are present in a single block.
          * ``"cohorts"``:
            Finds group labels that tend to occur together ("cohorts"),
            indexes out cohorts and reduces that subset using "map-reduce",
            repeat for all cohorts. This works well for many time groupings
            where the group labels repeat at regular intervals like 'hour',
            'month', dayofyear' etc. Optimize chunking ``array`` for this
            method by first rechunking using ``rechunk_for_cohorts``
            (for 1D ``by`` only).
          * ``"split-reduce"``:
            Break out each group into its own array and then ``"map-reduce"``.
            This is implemented by having each group be its own cohort,
            and is identical to xarray's default strategy.

    engine : {"flox", numpy", "numba"}, optional
        Underlying algorithm to compute the groupby reduction on non-dask arrays
        and on each dask chunk at compute-time.
          * ``"flox"``:
            Use an internal implementation where the data is sorted so that
            all members of a group occur sequentially, and then numpy.ufunc.reduceat
            is to used for the reduction. This will fall back to ``numpy_groupies.aggregate_numpy``
            for a reduction that is not yet implemented.
          * ``"numpy"``:
            Use the vectorized implementations in ``numpy_groupies.aggregate_numpy``.
          * ``"numba"``:
            Use the implementations in ``numpy_groupies.aggregate_numba``.

    keep_attrs : bool, optional
        Preserve attrs?
    skipna : bool, optional
        If True, skip missing values (as marked by NaN). By default, only
        skips missing values for float dtypes; other dtypes either do not
        have a sentinel missing value (int) or ``skipna=True`` has not been
        implemented (object, datetime64 or timedelta64).
    min_count : int, default: None
        The required number of valid values to perform the operation. If
        fewer than min_count non-NA values are present the result will be
        NA. Only used if skipna is set to True or defaults to True for the
        array's dtype.
    finalize_kwargs : dict, optional
        kwargs passed to the finalize function, like ddof for var, std.

    Returns
    -------
    DataArray or Dataset
        Reduced object

    See Also
    --------
    flox.core.groupby_reduce

    Raises
    ------
    NotImplementedError
    ValueError

    Examples
    --------
    FIXME: Add docs.
    """

    if skipna is not None and isinstance(func, Aggregation):
        raise ValueError("skipna must be None when func is an Aggregation.")

    for b in by:
        if isinstance(b, xr.DataArray) and b.name is None:
            raise ValueError("Cannot group by unnamed DataArrays.")

    if isinstance(isbin, bool):
        isbin = (isbin,) * len(by)

    # eventually  drop the variables we are grouping by
    maybe_drop = [b for b in by if isinstance(b, str)]
    unindexed_dims = tuple(
        b
        for b, isbin_ in zip(by, isbin)
        if isinstance(b, str) and not isbin_ and b in obj.dims and b not in obj.indexes
    )

    by: Tuple["DataArray"] = tuple(obj[g] if isinstance(g, str) else g for g in by)  # type: ignore

    if len(by) > 1 and any(is_duck_dask_array(by_.data) for by_ in by):
        raise NotImplementedError("Grouping by multiple variables will compute dask variables.")

    grouper_dims = set(itertools.chain(*tuple(g.dims for g in by)))

    if isinstance(obj, xr.DataArray):
        ds = obj._to_temp_dataset()
    else:
        ds = obj

    ds = ds.drop_vars([var for var in maybe_drop if var in ds.variables])
    if dim is Ellipsis:
        dim = tuple(obj.dims)
        if by[0].name in ds.dims and not isbin[0]:
            dim = tuple(d for d in dim if d != by[0].name)

    # TODO: do this for specific reductions only
    bad_dtypes = tuple(
        k for k in ds.variables if k not in ds.dims and ds[k].dtype.kind in ("S", "U")
    )

    # broadcast all variables against each other along all dimensions in `by` variables
    # don't exclude `dim` because it need not be a dimension in any of the `by` variables!
    # in the case where dim is Ellipsis, and by.ndim < obj.ndim
    # then we also broadcast `by` to all `obj.dims`
    # TODO: avoid this broadcasting
    exclude_dims = set(ds.dims) - grouper_dims
    if dim is not None:
        exclude_dims -= set(dim)
    ds, *by = xr.broadcast(ds, *by, exclude=exclude_dims)

    if dim is None:
        dim = tuple(by[0].dims)
    else:
        dim = _atleast_1d(dim)

    if any(d not in grouper_dims and d not in obj.dims for d in dim):
        raise ValueError(f"Cannot reduce over absent dimensions {dim}.")

    dims_not_in_groupers = tuple(d for d in dim if d not in grouper_dims)
    if dims_not_in_groupers == dim and not any(isbin):
        # reducing along a dimension along which groups do not vary
        # This is really just a normal reduction.
        # This is not right when binning so we exclude.
        if skipna:
            dsfunc = func[3:]
        else:
            dsfunc = func
        # TODO: skipna needs test
        result = getattr(ds, dsfunc)(dim=dim, skipna=skipna)
        if isinstance(obj, xr.DataArray):
            return obj._from_temp_dataset(result)
        else:
            return result

    axis = tuple(range(-len(dim), 0))
    group_names = tuple(g.name if not binned else f"{g.name}_bins" for g, binned in zip(by, isbin))

    if len(by) > 1:
        group_idx, expected_groups, group_shape, _, _, _ = factorize_(
            tuple(g.data for g in by),
            axis,
            expected_groups,
            isbin,
        )
        to_group = xr.DataArray(group_idx, dims=dim, coords={d: by[0][d] for d in by[0].indexes})
    else:
        if expected_groups is None and isinstance(by[0].data, np.ndarray):
            uniques = np.unique(by[0].data)
            nans = isnull(uniques)
            if nans.any():
                uniques = uniques[~nans]
            expected_groups = (uniques,)
        if expected_groups is None:
            raise NotImplementedError(
                "Please provide expected_groups if not grouping by a numpy-backed DataArray"
            )
        if isinstance(expected_groups, np.ndarray):
            expected_groups = (expected_groups,)
        if isbin[0]:
            if isinstance(expected_groups[0], int):
                raise NotImplementedError(
                    "Does not support binning into an integer number of bins yet."
                )
                #    factorized, bins = pd.cut(by[0], bins=expected_groups[0], retbins=True)
                group_shape = (expected_groups[0],)
            else:
                group_shape = (len(expected_groups[0]) - 1,)
        else:
            group_shape = (len(expected_groups[0]),)
        to_group = by[0]

    group_sizes = dict(zip(group_names, group_shape))

    def wrapper(array, to_group, *, func, skipna, **kwargs):
        # Handle skipna here because I need to know dtype to make a good default choice.
        # We cannnot handle this easily for xarray Datasets in xarray_reduce
        if skipna and func in ["all", "any", "count"]:
            raise ValueError(f"skipna cannot be truthy for {func} reductions.")

        if skipna or (skipna is None and isinstance(func, str) and array.dtype.kind in "cfO"):
            if "nan" not in func and func not in ["all", "any", "count"]:
                func = f"nan{func}"

        result, groups = groupby_reduce(array, to_group, func=func, **kwargs)
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
        else:
            # TODO: migrate this to core.groupby_reduce
            # index out NaN or NaT groups; these should be last
            if np.any(isnull(groups)):
                result = result[..., :-1]
                groups = groups[:-1]

        return result

    # These data variables do not have any of the core dimension,
    # take them out to prevent errors.
    # apply_ufunc can handle non-dim coordinate variables without core dimensions
    missing_dim = {}
    if isinstance(obj, xr.Dataset):
        # broadcasting means the group dim gets added to ds, so we check the original obj
        for k, v in obj.data_vars.items():
            if k in bad_dtypes:
                continue
            is_missing_dim = not (any(d in v.dims for d in dim))
            if is_missing_dim:
                missing_dim[k] = v

    actual = xr.apply_ufunc(
        wrapper,
        ds.drop_vars(tuple(missing_dim) + bad_dtypes),
        to_group,
        input_core_dims=_get_input_core_dims(group_names, dim, ds, to_group),
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
            "method": method,
            "min_count": min_count,
            "skipna": skipna,
            "engine": engine,
            # The following mess exists because for multiple `by`s I factorize eagerly
            # here before passing it on; this means I have to handle the
            # "binning by single by variable" case explicitly where the factorization
            # happens later allowing `by` to  be a dask variable.
            # Another annoyance is that for resampling expected_groups is "disconnected"
            # from "by" so we need the isbin part of the condition
            "expected_groups": expected_groups[0] if len(by) == 1 and isbin[0] else None,
            "isbin": isbin[0] if len(by) == 1 else False,
            "finalize_kwargs": finalize_kwargs,
        },
    )

    # restore non-dim coord variables without the core dimension
    # TODO: shouldn't apply_ufunc handle this?
    for var in set(ds.variables) - set(ds.dims):
        if all(d not in ds[var].dims for d in dim):
            actual[var] = ds[var]

    for name, expect, isbin_ in zip(group_names, expected_groups, isbin):
        if isbin_:
            expect = [pd.Interval(left, right) for left, right in zip(expect[:-1], expect[1:])]
        if isinstance(actual, xr.Dataset) and name in actual:
            actual = actual.drop_vars(name)
        actual[name] = expect

    # if grouping by multi-indexed variable, then restore it
    for name, index in ds.indexes.items():
        if name in actual.indexes and isinstance(index, pd.MultiIndex):
            actual[name] = index

    if unindexed_dims:
        actual = actual.drop_vars(unindexed_dims)

    if len(by) == 1:
        for var in actual:
            if isinstance(obj, xr.DataArray):
                template = obj
            else:
                template = obj[var]
            actual[var] = _restore_dim_order(actual[var], template, by[0])

    if missing_dim:
        for k, v in missing_dim.items():
            missing_group_dims = {
                dim: size for dim, size in group_sizes.items() if dim not in v.dims
            }
            # The expand_dims is for backward compat with xarray's questionable behaviour
            if missing_group_dims:
                actual[k] = v.expand_dims(missing_group_dims)
            else:
                actual[k] = v

    if isinstance(obj, xr.DataArray):
        return obj._from_temp_dataset(actual)
    else:
        return actual


def rechunk_for_cohorts(
    obj: Union["DataArray", "Dataset"],
    dim: str,
    labels: "DataArray",
    force_new_chunk_at,
    chunksize: Optional[int] = None,
):
    """
    Rechunks array so that each new chunk contains groups that always occur together.

    Parameters
    ----------
    array: DataArray or Dataset
        array to rechunk
    dim: str
        Dimension to rechunk
    labels: DataArray
        1D Group labels to align chunks with. This routine works
        well when ``labels`` has repeating patterns: e.g.
        ``1, 2, 3, 1, 2, 3, 4, 1, 2, 3`` though there is no requirement
        that the pattern must contain sequences.
    force_new_chunk_at:
        label at which we always start a new chunk. For
        the example ``labels`` array, this would be `1``.
    chunksize: int, optional
        nominal chunk size. Chunk size is exceded when the label
        in ``force_new_chunk_at`` is less than ``chunksize//2`` elements away.
        If None, uses median chunksize along ``dim``.
    Returns
    -------
    dask.array.Array
        rechunked array
    """
    return _rechunk(
        rechunk_array_for_cohorts,
        obj,
        dim,
        labels,
        force_new_chunk_at=force_new_chunk_at,
        chunksize=chunksize,
    )


def rechunk_to_group_boundaries(obj: Union["DataArray", "Dataset"], dim: str, labels: "DataArray"):
    """
    Rechunks array so that group boundaries line up with chunk boundaries, allowing
    parallel group reductions.

    This only works when the groups are sequential (e.g. labels = [0,0,0,1,1,1,1,2,2]).
    Such patterns occur when using ``.resample``.
    """

    return _rechunk(rechunk_for_blockwise, obj, dim, labels)


def _rechunk(func, obj, dim, labels, **kwargs):
    """Common logic for rechunking xarray objects."""
    obj = obj.copy(deep=True)

    if isinstance(obj, xr.Dataset):
        for var in obj:
            if obj[var].chunks is not None:
                obj[var] = obj[var].copy(
                    data=func(
                        obj[var].data, axis=obj[var].get_axis_num(dim), labels=labels.data, **kwargs
                    )
                )
    else:
        if obj.chunks is not None:
            obj = obj.copy(
                data=func(obj.data, axis=obj.get_axis_num(dim), labels=labels.data, **kwargs)
            )

    return obj


def resample_reduce(
    resampler: "Resample",
    func: Union[str, Aggregation],
    keep_attrs: bool = True,
    **kwargs,
):

    obj = resampler._obj
    dim = resampler._group_dim

    # this creates a label DataArray since resample doesn't do that somehow
    tostack = []
    for idx, slicer in enumerate(resampler._group_indices):
        if slicer.stop is None:
            stop = resampler._obj.sizes[dim]
        else:
            stop = slicer.stop
        tostack.append(idx * np.ones((stop - slicer.start,), dtype=np.int32))
    by = xr.DataArray(np.hstack(tostack), dims=(dim,), name="__resample_dim__")

    result = (
        xarray_reduce(
            obj,
            by,
            func=func,
            method="blockwise",
            expected_groups=(resampler._unique_coord.data,),
            keep_attrs=keep_attrs,
            **kwargs,
        )
        .rename({"__resample_dim__": dim})
        .transpose(dim, ...)
    )
    return result
