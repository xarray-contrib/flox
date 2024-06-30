import pytest

pytest.importorskip("hypothesis")

import dask
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import HealthCheck, assume, given, note, settings

from flox.core import dask_groupby_scan, groupby_reduce

from . import ALL_FUNCS, SCIPY_STATS_FUNCS, assert_equal

NON_NUMPY_FUNCS = ["first", "last", "nanfirst", "nanlast", "count", "any", "all"] + list(
    SCIPY_STATS_FUNCS
)
SKIPPED_FUNCS = ["var", "std", "nanvar", "nanstd"]


def supported_dtypes() -> st.SearchStrategy[np.dtype]:
    return (
        npst.integer_dtypes(endianness="=")
        | npst.unsigned_integer_dtypes(endianness="=")
        | npst.floating_dtypes(endianness="=", sizes=(32, 64))
        | npst.complex_number_dtypes(endianness="=")
        | npst.datetime64_dtypes(endianness="=")
        | npst.timedelta64_dtypes(endianness="=")
        | npst.unicode_string_dtypes(endianness="=")
    )


# TODO: stop excluding everything but U
array_dtype_st = supported_dtypes().filter(lambda x: x.kind not in "cmMU")
by_dtype_st = supported_dtypes()
func_st = st.sampled_from(
    [f for f in ALL_FUNCS if f not in NON_NUMPY_FUNCS and f not in SKIPPED_FUNCS]
)


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
    with np.errstate(invalid="ignore", divide="ignore"):
        actual, _ = groupby_reduce(
            array, by, func=func, axis=axis, engine="numpy", **flox_kwargs, finalize_kwargs=kwargs
        )

        # numpy-groupies always does the calculation in float64
        if (
            ("var" in func or "std" in func or "sum" in func or "mean" in func)
            and array.dtype.kind == "f"
            and array.dtype.itemsize != 8
        ):
            # bincount always accumulates in float64,
            # casting to float64 handles std more like npg does.
            # Setting dtype=float64 works fine for sum, mean.
            cast_to = array.dtype
            array = array.astype(np.float64)
            note(f"casting array to float64, cast_to={cast_to!r}")
        else:
            cast_to = None
        note(("kwargs:", kwargs, "cast_to:", cast_to))
        expected = getattr(np, func)(array, axis=axis, keepdims=True, **kwargs)
        if cast_to is not None:
            note(("casting to:", cast_to))
            expected = expected.astype(cast_to)
            actual = actual.astype(cast_to)

    note(("expected: ", expected, "actual: ", actual))
    tolerance = (
        {"rtol": 1e-13, "atol": 1e-15} if "var" in func or "std" in func else {"atol": 1e-15}
    )
    assert_equal(expected, actual, tolerance)


@st.composite
def chunked_arrays(
    draw,
    *,
    arrays=npst.arrays(
        elements={"allow_subnormal": False}, shape=npst.array_shapes(), dtype=array_dtype_st
    ),
    from_array=dask.array.from_array,
):
    array = draw(arrays)
    size = array.shape[-1]
    if size > 1:
        nchunks = draw(st.integers(min_value=1, max_value=size - 1))
        dividers = sorted(
            set(draw(st.integers(min_value=1, max_value=size - 1)) for _ in range(nchunks - 1))
        )
        chunks = tuple(a - b for a, b in zip(dividers + [size], [0] + dividers))
    else:
        chunks = (1,)
    return from_array(array, chunks=("auto",) * (array.ndim - 1) + (chunks,))


from flox.aggregations import cumsum

dask.config.set(scheduler="sync")


def test():
    array = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    da = dask.array.from_array(array, chunks=2)
    actual = dask_groupby_scan(
        da, np.array([0] * array.shape[-1]), agg=cumsum, axes=(array.ndim - 1,)
    )
    actual.compute()


@given(data=st.data(), array=chunked_arrays())
def test_scans(data, array):
    note(np.array(array))
    actual = dask_groupby_scan(
        array, np.array([0] * array.shape[-1]), agg=cumsum, axes=(array.ndim - 1,)
    )
    expected = np.cumsum(np.asarray(array), axis=-1)
    np.testing.assert_array_equal(expected, actual)
