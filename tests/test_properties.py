import pytest

pytest.importorskip("hypothesis")

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import HealthCheck, assume, given, note, settings

from flox.core import groupby_reduce

from . import ALL_FUNCS, assert_equal

NON_NUMPY_FUNCS = ["first", "last", "nanfirst", "nanlast", "count", "any", "all", "mode", "nanmode"]


def supported_dtypes() -> st.SearchStrategy[np.dtype]:
    return (
        npst.integer_dtypes()
        | npst.unsigned_integer_dtypes()
        | npst.floating_dtypes()
        # | npst.complex_number_dtypes()
        # | npst.datetime64_dtypes()
        # | npst.timedelta64_dtypes()
        | npst.unicode_string_dtypes()
    )


dtype_st = supported_dtypes().filter(lambda x: x.byteorder == "=" and x.kind not in ["mcM"])

array_dtype_st = dtype_st.filter(lambda x: x.kind != "U")
by_dtype_st = dtype_st

func_st = st.sampled_from([f for f in ALL_FUNCS if f not in NON_NUMPY_FUNCS])


def not_overflowing_array(array) -> bool:
    if array.dtype.kind == "f":
        info = np.finfo(array.dtype)
    elif array.dtype.kind in ["i", "u"]:
        info = np.iinfo(array.dtype)  # type: ignore[assignment]
    else:
        return True

    return bool(np.all((array < info.max / array.size) & (array > info.min / array.size)))


@settings(suppress_health_check=[HealthCheck.filter_too_much])
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
    # numpy-groupies always does the calculation in float64
    assume(func != "var" and "f2" not in array.dtype.str)
    # arg* with nans in array are weird
    assume("arg" not in func and not np.any(np.isnan(array).ravel()))

    axis = -1
    by = np.ones((array.shape[-1],), dtype=dtype)
    kwargs = {"q": 0.8} if "quantile" in func else {}

    with np.errstate(invalid="ignore", divide="ignore"):
        actual, _ = groupby_reduce(
            array, by, func=func, axis=axis, engine="numpy", finalize_kwargs=kwargs
        )
        expected = getattr(np, func)(array, axis=axis, keepdims=True, **kwargs)
    note(("expected: ", expected, "actual: ", actual))
    tolerance = {"rtol": 1e-13, "atol": 1e-16} if "var" in func or "std" in func else {}
    assert_equal(expected, actual, tolerance)
