from __future__ import annotations

from typing import TYPE_CHECKING, Hashable, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from packaging.version import Version
from xarray.core.duck_array_ops import _datetime_nanmin

from .aggregations import Aggregation, _atleast_1d
from .core import (
    _convert_expected_groups_to_index,
    _get_expected_groups,
    groupby_reduce,
    rechunk_for_blockwise as rechunk_array_for_blockwise,
    rechunk_for_cohorts as rechunk_array_for_cohorts,
)
from .xrutils import _contains_cftime_datetimes, _to_pytimedelta, datetime_to_numeric

if TYPE_CHECKING:
    from xarray import DataArray, Dataset, Resample


def _get_input_core_dims(group_names, dim, ds, grouper_dims):
    input_core_dims = [[], []]
    for g in group_names:
        if g in dim:
            continue
        if g in ds.dims:
            input_core_dims[0].extend([g])
        if g in grouper_dims:
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
    obj: Dataset | DataArray,
    *by: DataArray | Iterable[str] | Iterable[DataArray],
    func: str | Aggregation,
    expected_groups=None,
    isbin: bool | Sequence[bool] = False,
    sort: bool = True,
    dim: Hashable = None,
    split_out: int = 1,
    fill_value=None,
    method: str = "map-reduce",
    engine: str = "flox",
    keep_attrs: bool = True,
    skipna: bool | None = None,
    min_count: int | None = None,
    reindex: bool | None = None,
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
        reductions when ``method`` is not ``"map-reduce"``. For ``"map-reduce"``, the groups
        are always sorted.
    dim : hashable
        dimension name along which to reduce. If None, reduces across all
        dimensions of `by`
    split_out : int, optional
        Number of output chunks along grouped dimension in output.
    fill_value
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
    engine : {"flox", "numpy", "numba"}, optional
        Algorithm to compute the groupby reduction on non-dask arrays and on each dask chunk:
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
    reindex : bool, optional
        Whether to "reindex" the blockwise results to `expected_groups` (possibly automatically detected).
        If True, the intermediate result of the blockwise groupby-reduction has a value for all expected groups,
        and the final result is a simple reduction of those intermediates. In nearly all cases, this is a significant
        boost in computation speed. For cases like time grouping, this may result in large intermediates relative to the
        original block size. Avoid that by using method="cohorts". By default, it is turned off for arg reductions.
    **finalize_kwargs :
        kwargs passed to the finalize function, like ``ddof`` for var, std.

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
    if expected_groups is None:
        expected_groups = (None,) * len(by)
    if isinstance(expected_groups, (np.ndarray, list)):  # TODO: test for list
        if len(by) == 1:
            expected_groups = (expected_groups,)
        else:
            raise ValueError("Needs better message.")

    if not sort:
        raise NotImplementedError

    # eventually drop the variables we are grouping by
    maybe_drop = [b for b in by if isinstance(b, str)]
    unindexed_dims = tuple(
        b
        for b, isbin_ in zip(by, isbin)
        if isinstance(b, str) and not isbin_ and b in obj.dims and b not in obj.indexes
    )

    by: tuple[DataArray] = tuple(obj[g] if isinstance(g, str) else g for g in by)  # type: ignore

    grouper_dims = []
    for g in by:
        for d in g.dims:
            if d not in grouper_dims:
                grouper_dims.append(d)

    if isinstance(obj, xr.DataArray):
        ds = obj._to_temp_dataset()
    else:
        ds = obj

    ds = ds.drop_vars([var for var in maybe_drop if var in ds.variables])

    if dim is Ellipsis:
        dim = tuple(obj.dims)
        if by[0].name in ds.dims and not isbin[0]:
            dim = tuple(d for d in dim if d != by[0].name)
    elif dim is not None:
        dim = _atleast_1d(dim)
    else:
        dim = tuple()

    # broadcast all variables against each other along all dimensions in `by` variables
    # don't exclude `dim` because it need not be a dimension in any of the `by` variables!
    # in the case where dim is Ellipsis, and by.ndim < obj.ndim
    # then we also broadcast `by` to all `obj.dims`
    # TODO: avoid this broadcasting
    exclude_dims = tuple(d for d in ds.dims if d not in grouper_dims and d not in dim)
    ds, *by = xr.broadcast(ds, *by, exclude=exclude_dims)

    if not dim:
        dim = tuple(by[0].dims)

    if any(d not in grouper_dims and d not in obj.dims for d in dim):
        raise ValueError(f"Cannot reduce over absent dimensions {dim}.")

    dims_not_in_groupers = tuple(d for d in dim if d not in grouper_dims)
    if dims_not_in_groupers == dim and not any(isbin):
        # reducing along a dimension along which groups do not vary
        # This is really just a normal reduction.
        # This is not right when binning so we exclude.
        if skipna and isinstance(func, str):
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

    group_shape = [None] * len(by)
    expected_groups = list(expected_groups)

    # Set expected_groups and convert to index since we need coords, sizes
    # for output xarray objects
    for idx, (b, expect, isbin_) in enumerate(zip(by, expected_groups, isbin)):
        if isbin_ and isinstance(expect, int):
            raise NotImplementedError(
                "flox does not support binning into an integer number of bins yet."
            )
        if expect is None:
            if isbin_:
                raise ValueError(
                    f"Please provided bin edges for group variable {idx} "
                    f"named {group_names[idx]} in expected_groups."
                )
            expected_groups[idx] = _get_expected_groups(b.data, sort=sort, raise_if_dask=True)

    expected_groups = _convert_expected_groups_to_index(expected_groups, isbin, sort=sort)
    group_shape = tuple(len(e) for e in expected_groups)
    group_sizes = dict(zip(group_names, group_shape))

    def wrapper(array, *by, func, skipna, **kwargs):
        # Handle skipna here because I need to know dtype to make a good default choice.
        # We cannnot handle this easily for xarray Datasets in xarray_reduce
        if skipna and func in ["all", "any", "count"]:
            raise ValueError(f"skipna cannot be truthy for {func} reductions.")

        if skipna or (skipna is None and isinstance(func, str) and array.dtype.kind in "cfO"):
            if "nan" not in func and func not in ["all", "any", "count"]:
                func = f"nan{func}"

        requires_numeric = func not in ["count", "any", "all"]
        if requires_numeric:
            is_npdatetime = array.dtype.kind in "Mm"
            is_cftime = _contains_cftime_datetimes(array)
            if is_npdatetime:
                offset = _datetime_nanmin(array)
                # xarray always uses np.datetime64[ns] for np.datetime64 data
                dtype = "timedelta64[ns]"
                array = datetime_to_numeric(array, offset)
            elif _contains_cftime_datetimes(array):
                offset = min(array)
                array = datetime_to_numeric(array, offset, datetime_unit="us")

        result, *groups = groupby_reduce(array, *by, func=func, **kwargs)

        if requires_numeric:
            if is_npdatetime:
                return result.astype(dtype) + offset
            elif is_cftime:
                return _to_pytimedelta(result, unit="us") + offset

        return result

    # These data variables do not have any of the core dimension,
    # take them out to prevent errors.
    # apply_ufunc can handle non-dim coordinate variables without core dimensions
    missing_dim = {}
    if isinstance(obj, xr.Dataset):
        # broadcasting means the group dim gets added to ds, so we check the original obj
        for k, v in obj.data_vars.items():
            is_missing_dim = not (any(d in v.dims for d in dim))
            if is_missing_dim:
                missing_dim[k] = v

    input_core_dims = _get_input_core_dims(group_names, dim, ds, grouper_dims)
    input_core_dims += [input_core_dims[-1]] * (len(by) - 1)

    actual = xr.apply_ufunc(
        wrapper,
        ds.drop_vars(tuple(missing_dim)).transpose(..., *grouper_dims),
        *by,
        input_core_dims=input_core_dims,
        # for xarray's test_groupby_duplicate_coordinate_labels
        exclude_dims=set(dim),
        output_core_dims=[group_names],
        dask="allowed",
        dask_gufunc_kwargs=dict(output_sizes=group_sizes),
        keep_attrs=keep_attrs,
        kwargs={
            "func": func,
            "axis": axis,
            "sort": sort,
            "split_out": split_out,
            "fill_value": fill_value,
            "method": method,
            "min_count": min_count,
            "skipna": skipna,
            "engine": engine,
            "reindex": reindex,
            "expected_groups": tuple(expected_groups),
            "isbin": isbin,
            "finalize_kwargs": finalize_kwargs,
        },
    )

    # restore non-dim coord variables without the core dimension
    # TODO: shouldn't apply_ufunc handle this?
    for var in set(ds.variables) - set(ds.dims):
        if all(d not in ds[var].dims for d in dim):
            actual[var] = ds[var]

    for name, expect in zip(group_names, expected_groups):
        # Can't remove this till xarray handles IntervalIndex
        if isinstance(expect, pd.IntervalIndex):
            expect = expect.to_numpy()
        if isinstance(actual, xr.Dataset) and name in actual:
            actual = actual.drop_vars(name)
        # When grouping by MultiIndex, expect is an pd.Index wrapping
        # an object array of tuples
        if name in ds.indexes and isinstance(ds.indexes[name], pd.MultiIndex):
            levelnames = ds.indexes[name].names
            expect = pd.MultiIndex.from_tuples(expect.values, names=levelnames)
            actual[name] = expect
            if Version(xr.__version__) > Version("2022.03.0"):
                actual = actual.set_coords(levelnames)
        else:
            actual[name] = expect

    if unindexed_dims:
        actual = actual.drop_vars(unindexed_dims)

    if len(by) == 1:
        for var in actual:
            if isinstance(obj, xr.DataArray):
                template = obj
            else:
                template = obj[var]
            if actual[var].ndim > 1:
                actual[var] = _restore_dim_order(actual[var], template, by[0])

    if missing_dim:
        for k, v in missing_dim.items():
            missing_group_dims = {
                dim: size for dim, size in group_sizes.items() if dim not in v.dims
            }
            # The expand_dims is for backward compat with xarray's questionable behaviour
            if missing_group_dims:
                actual[k] = v.expand_dims(missing_group_dims).variable
            else:
                actual[k] = v.variable

    if isinstance(obj, xr.DataArray):
        return obj._from_temp_dataset(actual)
    else:
        return actual


def rechunk_for_cohorts(
    obj: DataArray | Dataset,
    dim: str,
    labels: DataArray,
    force_new_chunk_at,
    chunksize: int | None = None,
    ignore_old_chunks: bool = False,
    debug: bool = False,
):
    """
    Rechunks array so that each new chunk contains groups that always occur together.

    Parameters
    ----------
    obj : DataArray or Dataset
        array to rechunk
    dim : str
        Dimension to rechunk
    labels : DataArray
        1D Group labels to align chunks with. This routine works
        well when ``labels`` has repeating patterns: e.g.
        ``1, 2, 3, 1, 2, 3, 4, 1, 2, 3`` though there is no requirement
        that the pattern must contain sequences.
    force_new_chunk_at : Sequence
        Labels at which we always start a new chunk. For
        the example ``labels`` array, this would be `1`.
    chunksize : int, optional
        nominal chunk size. Chunk size is exceded when the label
        in ``force_new_chunk_at`` is less than ``chunksize//2`` elements away.
        If None, uses median chunksize along ``dim``.

    Returns
    -------
    DataArray or Dataset
        Xarray object with rechunked arrays.
    """
    return _rechunk(
        rechunk_array_for_cohorts,
        obj,
        dim,
        labels,
        force_new_chunk_at=force_new_chunk_at,
        chunksize=chunksize,
        ignore_old_chunks=ignore_old_chunks,
        debug=debug,
    )


def rechunk_for_blockwise(obj: DataArray | Dataset, dim: str, labels: DataArray):
    """
    Rechunks array so that group boundaries line up with chunk boundaries, allowing
    embarassingly parallel group reductions.

    This only works when the groups are sequential
    (e.g. labels = ``[0,0,0,1,1,1,1,2,2]``).
    Such patterns occur when using ``.resample``.

    Parameters
    ----------
    obj : DataArray or Dataset
        Array to rechunk
    dim : hashable
        Name of dimension to rechunk
    labels : DataArray
        Group labels

    Returns
    -------
    DataArray or Dataset
        Xarray object with rechunked arrays.
    """
    return _rechunk(rechunk_array_for_blockwise, obj, dim, labels)


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
    resampler: Resample,
    func: str | Aggregation,
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
            keep_attrs=keep_attrs,
            **kwargs,
        )
        .rename({"__resample_dim__": dim})
        .transpose(dim, ...)
    )
    result[dim] = resampler._unique_coord.data
    return result
