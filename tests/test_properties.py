import pytest

pytest.importorskip("hypothesis")

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import HealthCheck, assume, given, note, settings

from flox.core import groupby_reduce

from . import ALL_FUNCS, SCIPY_STATS_FUNCS, assert_equal

NON_NUMPY_FUNCS = ["first", "last", "nanfirst", "nanlast", "count", "any", "all"] + list(
    SCIPY_STATS_FUNCS
)


def supported_dtypes() -> st.SearchStrategy[np.dtype]:
    return (
        npst.integer_dtypes(endianness="=")
        | npst.unsigned_integer_dtypes(endianness="=")
        | npst.floating_dtypes(endianness="=")
        | npst.complex_number_dtypes(endianness="=")
        | npst.datetime64_dtypes(endianness="=")
        | npst.timedelta64_dtypes(endianness="=")
        | npst.unicode_string_dtypes(endianness="=")
    )


# TODO: stop excluding datetime64, timedelta64
array_dtype_st = supported_dtypes().filter(lambda x: x.kind not in "mMU")
by_dtype_st = supported_dtypes()
func_st = st.sampled_from([f for f in ALL_FUNCS if f not in NON_NUMPY_FUNCS])


def not_overflowing_array(array) -> bool:
    if array.dtype.kind == "f":
        info = np.finfo(array.dtype)
    elif array.dtype.kind in ["i", "u"]:
        info = np.iinfo(array.dtype)  # type: ignore[assignment]
    else:
        return True

    result = bool(np.all((array < info.max / array.size) & (array > info.min / array.size)))
    # note(f"returning {result}, {array.min()} vs {info.min}, {array.max()} vs {info.max}")
    return result


@settings(
    max_examples=300, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow]
)
@given(
    array=npst.arrays(
        elements={"allow_subnormal": False}, shape=npst.array_shapes(), dtype=array_dtype_st
    ),
    dtype=by_dtype_st,
    func=func_st,
)
def test_groupby_reduce(array, dtype, func):
    # overflow behaviour differs between bincount and sum (for example)
    assume(not_overflowing_array(array))
    # TODO: fix var for complex numbers upstream
    assume(not (("quantile" in func or "var" in func or "std" in func) and array.dtype.kind == "c"))
    # arg* with nans in array are weird
    assume("arg" not in func and not np.any(np.isnan(array).ravel()))

    axis = -1
    by = np.ones((array.shape[-1],), dtype=dtype)
    kwargs = {"q": 0.8} if "quantile" in func else {}
    flox_kwargs = {}
    # numpy-groupies always does the calculation in float64
    if (
        "var" in func or "std" in func or "sum" in func or "mean" in func
    ) and array.dtype.kind == "f":
        # bincount accumulates in float64
        flox_kwargs.setdefault("dtype", np.float64)
        cast_to = array.dtype
    else:
        cast_to = None
    note((kwargs, cast_to))

    with np.errstate(invalid="ignore", divide="ignore"):
        actual, _ = groupby_reduce(
            array, by, func=func, axis=axis, engine="numpy", **flox_kwargs, finalize_kwargs=kwargs
        )
        expected = getattr(np, func)(array, axis=axis, keepdims=True, **kwargs)
    note(("expected: ", expected, "actual: ", actual))
    tolerance = {"rtol": 1e-13, "atol": 1e-16} if "var" in func or "std" in func else {}
    if cast_to:
        actual = actual.astype(cast_to)
    assert_equal(expected, actual, tolerance)
