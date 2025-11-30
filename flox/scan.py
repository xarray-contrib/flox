"""Scan operations for groupby reductions.

This module provides scan operations (cumulative reductions) for grouped data.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from .aggregations import (
    AGGREGATIONS,
    AlignedArrays,
    Scan,
    ScanState,
    _atleast_1d,
    generic_aggregate,
)
from .factorize import _factorize_multiple
from .xrutils import is_duck_array, is_duck_dask_array, module_available

if module_available("numpy", minversion="2.0.0"):
    from numpy.lib.array_utils import normalize_axis_tuple
else:
    from numpy.core.numeric import normalize_axis_tuple  # type: ignore[no-redef]

if TYPE_CHECKING:
    from .core import (
        T_By,
        T_EngineOpt,
        T_ExpectedGroupsOpt,
        T_MethodOpt,
        T_Scan,
    )
    from .types import DaskArray


def _validate_expected_groups_for_scan(nby, expected_groups):
    """Validate expected_groups for scan operations."""
    if expected_groups is None:
        return (None,) * nby
    return expected_groups


def _convert_expected_groups_to_index_for_scan(expected_groups, isbin, sort):
    """Convert expected_groups to index for scan operations."""
    import pandas as pd

    result = []
    for expect, isbin_ in zip(expected_groups, isbin):
        if expect is None:
            result.append(None)
        elif isinstance(expect, pd.Index):
            result.append(expect)
        else:
            result.append(pd.Index(expect))
    return tuple(result)


def groupby_scan(
    array: np.ndarray | DaskArray,
    *by: T_By,
    func: T_Scan,
    expected_groups: T_ExpectedGroupsOpt = None,
    axis: int | tuple[int] = -1,
    dtype: np.typing.DTypeLike = None,
    method: T_MethodOpt = None,
    engine: T_EngineOpt = None,
) -> np.ndarray | DaskArray:
    """
    GroupBy reductions using parallel scans for dask.array

    Parameters
    ----------
    array : ndarray or DaskArray
        Array to be reduced, possibly nD
    *by : ndarray or DaskArray
        Array of labels to group over. Must be aligned with ``array`` so that
        ``array.shape[-by.ndim :] == by.shape`` or any disagreements in that
        equality check are for dimensions of size 1 in `by`.
    func : {"nancumsum", "ffill", "bfill"} or Scan
        Single function name or a Scan instance
    expected_groups : (optional) Sequence
        Expected unique labels.
    axis : None or int or Sequence[int], optional
        If None, reduce across all dimensions of by
        Else, reduce across corresponding axes of array
        Negative integers are normalized using array.ndim.
    fill_value : Any
        Value to assign when a label in ``expected_groups`` is not present.
    dtype : data-type , optional
        DType for the output. Can be anything that is accepted by ``np.dtype``.
    method : {"blockwise", "cohorts"}, optional
        Strategy for reduction of dask arrays only:
          * ``"blockwise"``:
            Only scan using blockwise and avoid aggregating blocks
            together. Useful for resampling-style groupby problems where group
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
    engine : {"flox", "numpy", "numba", "numbagg"}, optional
        Algorithm to compute the groupby reduction on non-dask arrays and on each dask chunk:
          * ``"numpy"``:
            Use the vectorized implementations in ``numpy_groupies.aggregate_numpy``.
            This is the default choice because it works for most array types.
          * ``"flox"``:
            Use an internal implementation where the data is sorted so that
            all members of a group occur sequentially, and then numpy.ufunc.reduceat
            is to used for the reduction. This will fall back to ``numpy_groupies.aggregate_numpy``
            for a reduction that is not yet implemented.
          * ``"numba"``:
            Use the implementations in ``numpy_groupies.aggregate_numba``.
          * ``"numbagg"``:
            Use the reductions supported by ``numbagg.grouped``. This will fall back to ``numpy_groupies.aggregate_numpy``
            for a reduction that is not yet implemented.

    Returns
    -------
    result
        Aggregated result

    See Also
    --------
    xarray.xarray_reduce
    """
    from . import xrdtypes

    axis = _atleast_1d(axis)
    if len(axis) > 1:
        raise NotImplementedError("Scans are only supported along a single dimension.")

    bys: tuple = tuple(np.asarray(b) if not is_duck_array(b) else b for b in by)
    nby = len(by)
    by_is_dask = tuple(is_duck_dask_array(b) for b in bys)
    any_by_dask = any(by_is_dask)

    axis_ = normalize_axis_tuple(axis, array.ndim)

    if engine is not None:
        raise NotImplementedError("Setting `engine` is not supported for scans yet.")
    if method is not None:
        raise NotImplementedError("Setting `method` is not supported for scans yet.")
    if engine is None:
        engine = "flox"
    assert engine == "flox"

    if not is_duck_array(array):
        array = np.asarray(array)

    if isinstance(func, str):
        agg = AGGREGATIONS[func]
    assert isinstance(agg, Scan)
    agg = copy.deepcopy(agg)

    if (agg == AGGREGATIONS["ffill"] or agg == AGGREGATIONS["bfill"]) and array.dtype.kind != "f":
        # nothing to do, no NaNs!
        return array

    if expected_groups is not None:
        raise NotImplementedError("Setting `expected_groups` and binning is not supported yet.")
    expected_groups = _validate_expected_groups_for_scan(nby, expected_groups)
    expected_groups = _convert_expected_groups_to_index_for_scan(
        expected_groups, isbin=(False,) * nby, sort=False
    )

    # Don't factorize early only when
    # grouping by dask arrays, and not having expected_groups
    factorize_early = not (
        # can't do it if we are grouping by dask array but don't have expected_groups
        any(is_dask and ex_ is None for is_dask, ex_ in zip(by_is_dask, expected_groups))
    )
    if factorize_early:
        bys, final_groups, grp_shape = _factorize_multiple(
            bys,
            expected_groups,
            any_by_dask=any_by_dask,
            sort=False,
        )
    else:
        raise NotImplementedError

    assert len(bys) == 1
    by_: np.ndarray
    (by_,) = bys
    has_dask = is_duck_dask_array(array) or is_duck_dask_array(by_)

    if array.dtype.kind in "Mm":
        cast_to = array.dtype
        array = array.view(np.int64)
    elif array.dtype.kind == "b":
        array = array.view(np.int8)
        cast_to = None
        if agg.preserves_dtype:
            cast_to = bool
    else:
        cast_to = None

    # TODO: move to aggregate_npg.py
    if agg.name in ["cumsum", "nancumsum"] and array.dtype.kind in ["i", "u"]:
        # https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
        # it defaults to the dtype of a, unless a
        # has an integer dtype with a precision less than that of the default platform integer.
        if array.dtype.kind == "i":
            agg.dtype = np.result_type(array.dtype, np.int_)
        elif array.dtype.kind == "u":
            agg.dtype = np.result_type(array.dtype, np.uint)
    else:
        agg.dtype = array.dtype if dtype is None else dtype
    agg.identity = xrdtypes._get_fill_value(agg.dtype, agg.identity)

    (single_axis,) = axis_  # type: ignore[misc]
    # avoid some roundoff error when we can.
    if by_.shape[-1] == 1 or by_.shape == grp_shape:
        array = array.astype(agg.dtype)
        if cast_to is not None:
            array = array.astype(cast_to)
        return array

    # Made a design choice here to have `preprocess` handle both array and group_idx
    # Example: for reversing, we need to reverse the whole array, not just reverse
    #          each block independently
    inp = AlignedArrays(array=array, group_idx=by_)
    if agg.preprocess:
        inp = agg.preprocess(inp)

    if not has_dask:
        final_state = chunk_scan(inp, axis=single_axis, agg=agg, dtype=agg.dtype)
        result = _finalize_scan(final_state, dtype=agg.dtype)
    else:
        from .dask import dask_groupby_scan

        result = dask_groupby_scan(inp.array, inp.group_idx, axes=axis_, agg=agg)

    # Made a design choice here to have `postprocess` handle both array and group_idx
    out = AlignedArrays(array=result, group_idx=by_)
    if agg.finalize:
        out = agg.finalize(out)

    if cast_to is not None:
        return out.array.astype(cast_to)
    return out.array


def chunk_scan(inp: AlignedArrays, *, axis: int, agg: Scan, dtype=None, keepdims=None) -> ScanState:
    assert axis == inp.array.ndim - 1

    # I don't think we need to re-factorize here unless we are grouping by a dask array
    accumulated = generic_aggregate(
        inp.group_idx,
        inp.array,
        axis=axis,
        engine="flox",
        func=agg.scan,
        dtype=dtype,
        fill_value=agg.identity,
    )
    result = AlignedArrays(array=accumulated, group_idx=inp.group_idx)
    return ScanState(result=result, state=None)


def grouped_reduce(inp: AlignedArrays, *, agg: Scan, axis: int, keepdims=None) -> ScanState:
    from .core import chunk_reduce

    assert axis == inp.array.ndim - 1
    reduced = chunk_reduce(
        inp.array,
        inp.group_idx,
        func=(agg.reduction,),
        axis=axis,
        engine="flox",
        dtype=inp.array.dtype,
        fill_value=agg.identity,
        expected_groups=None,
    )
    return ScanState(
        state=AlignedArrays(array=reduced["intermediates"][0], group_idx=reduced["groups"]),
        result=None,
    )


def _zip(group_idx: np.ndarray, array: np.ndarray) -> AlignedArrays:
    return AlignedArrays(group_idx=group_idx, array=array)


def _finalize_scan(block: ScanState, dtype) -> np.ndarray:
    assert block.result is not None
    return block.result.array.astype(dtype, copy=False)


__all__ = [
    "_finalize_scan",
    "_zip",
    "chunk_scan",
    "grouped_reduce",
    "groupby_scan",
]
