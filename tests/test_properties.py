import pandas as pd
import pytest

pytest.importorskip("hypothesis")
pytest.importorskip("dask")
pytest.importorskip("cftime")

import cftime
import dask
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given, note

import flox
from flox.core import groupby_reduce, groupby_scan

from . import ALL_FUNCS, SCIPY_STATS_FUNCS, assert_equal

dask.config.set(scheduler="sync")


def ffill(array, axis, dtype=None):
    return flox.aggregate_flox.ffill(np.zeros(array.shape[-1], dtype=int), array, axis=axis)


def bfill(array, axis, dtype=None):
    return flox.aggregate_flox.ffill(
        np.zeros(array.shape[-1], dtype=int),
        array[::-1],
        axis=axis,
    )[::-1]


NON_NUMPY_FUNCS = ["first", "last", "nanfirst", "nanlast", "count", "any", "all"] + list(
    SCIPY_STATS_FUNCS
)
SKIPPED_FUNCS = ["var", "std", "nanvar", "nanstd"]
NUMPY_SCAN_FUNCS = {
    "nancumsum": np.nancumsum,
    "ffill": ffill,
    "bfill": bfill,
}  # "cumsum": np.cumsum,


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
numeric_arrays = npst.arrays(
    elements={"allow_subnormal": False}, shape=npst.array_shapes(), dtype=array_dtype_st
)
all_arrays = npst.arrays(
    elements={"allow_subnormal": False}, shape=npst.array_shapes(), dtype=supported_dtypes()
)

calendars = st.sampled_from(
    [
        "standard",
        "gregorian",
        "proleptic_gregorian",
        "noleap",
        "365_day",
        "360_day",
        "julian",
        "all_leap",
        "366_day",
    ]
)


@st.composite
def units(draw, *, calendar: str):
    choices = ["days", "hours", "minutes", "seconds", "milliseconds", "microseconds"]
    if calendar == "360_day":
        choices += ["months"]
    elif calendar == "noleap":
        choices += ["common_years"]
    time_units = draw(st.sampled_from(choices))

    dt = draw(st.datetimes())
    year, month, day = dt.year, dt.month, dt.day
    if calendar == "360_day":
        month %= 30
    return f"{time_units} since {year}-{month}-{day}"


@st.composite
def cftime_arrays(draw, *, shape, calendars=calendars, elements=None):
    if elements is None:
        elements = {"min_value": -10_000, "max_value": 10_000}
    cal = draw(calendars)
    values = draw(npst.arrays(dtype=np.int64, shape=shape, elements=elements))
    unit = draw(units(calendar=cal))
    return cftime.num2date(values, units=unit, calendar=cal)


def by_arrays(shape, *, elements=None):
    return st.one_of(
        npst.arrays(
            dtype=npst.integer_dtypes(endianness="=") | npst.unicode_string_dtypes(endianness="="),
            shape=shape,
            elements=elements,
        ),
        cftime_arrays(shape=shape, elements=elements),
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


@st.composite
def chunks(draw, *, shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    chunks = []
    for size in shape:
        if size > 1:
            nchunks = draw(st.integers(min_value=1, max_value=size - 1))
            dividers = sorted(
                set(draw(st.integers(min_value=1, max_value=size - 1)) for _ in range(nchunks - 1))
            )
            chunks.append(tuple(a - b for a, b in zip(dividers + [size], [0] + dividers)))
        else:
            chunks.append((1,))
    return tuple(chunks)


@st.composite
def chunked_arrays(draw, *, chunks=chunks, arrays=numeric_arrays, from_array=dask.array.from_array):
    array = draw(arrays)
    chunks = draw(chunks(shape=array.shape))

    if array.dtype.kind in "cf":
        nan_idx = draw(
            st.lists(
                st.integers(min_value=0, max_value=array.shape[-1] - 1),
                max_size=array.shape[-1] - 1,
                unique=True,
            )
        )
        if nan_idx:
            array[..., nan_idx] = np.nan

    return from_array(array, chunks=chunks)


# TODO: migrate to by_arrays but with constant value
@given(data=st.data(), array=numeric_arrays, func=func_st)
def test_groupby_reduce(data, array, func):
    # overflow behaviour differs between bincount and sum (for example)
    assume(not_overflowing_array(array))
    # TODO: fix var for complex numbers upstream
    assume(not (("quantile" in func or "var" in func or "std" in func) and array.dtype.kind == "c"))
    # arg* with nans in array are weird
    assume("arg" not in func and not np.any(np.isnan(array).ravel()))

    axis = -1
    by = data.draw(
        by_arrays(
            elements={
                "alphabet": st.just("a"),
                "min_value": 1,
                "max_value": 1,
                "min_size": 1,
                "max_size": 1,
            },
            shape=array.shape[-1],
        )
    )
    assert len(np.unique(by)) == 1
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


@given(
    data=st.data(),
    array=chunked_arrays(),
    func=st.sampled_from(tuple(NUMPY_SCAN_FUNCS)),
)
def test_scans(data, array, func):
    assume(not_overflowing_array(np.asarray(array)))

    by = data.draw(by_arrays(shape=(array.shape[-1],)))
    axis = array.ndim - 1

    # Too many float32 edge-cases!
    if "cum" in func and array.dtype.kind == "f" and array.dtype.itemsize == 4:
        array = array.astype(np.float64)
    numpy_array = array.compute()
    assume((np.abs(numpy_array) < 2**53).all())

    dtype = NUMPY_SCAN_FUNCS[func](numpy_array[..., [0]], axis=axis).dtype
    expected = np.empty_like(numpy_array, dtype=dtype)
    group_idx, uniques = pd.factorize(by)
    for i in range(len(uniques)):
        mask = group_idx == i
        if not mask.any():
            note((by, group_idx, uniques))
            raise ValueError
        expected[..., mask] = NUMPY_SCAN_FUNCS[func](numpy_array[..., mask], axis=axis, dtype=dtype)

    note((numpy_array, group_idx, array.chunks))

    tolerance = {"rtol": 1e-13, "atol": 1e-15}
    actual = groupby_scan(numpy_array, by, func=func, axis=-1, dtype=dtype)
    assert_equal(actual, expected, tolerance)

    actual = groupby_scan(array, by, func=func, axis=-1, dtype=dtype)
    assert_equal(actual, expected, tolerance)


@given(data=st.data(), array=chunked_arrays())
def test_ffill_bfill_reverse(data, array):
    # TODO: test NaT and timedelta, datetime
    assume(not_overflowing_array(np.asarray(array)))
    by = data.draw(by_arrays(shape=(array.shape[-1],)))

    def reverse(arr):
        return arr[..., ::-1]

    for a in (array, array.compute()):
        forward = groupby_scan(a, by, func="ffill")
        backward_reversed = reverse(groupby_scan(reverse(a), reverse(by), func="bfill"))
        assert_equal(forward, backward_reversed)

        backward = groupby_scan(a, by, func="bfill")
        forward_reversed = reverse(groupby_scan(reverse(a), reverse(by), func="ffill"))
        assert_equal(forward_reversed, backward)


@given(
    data=st.data(),
    array=chunked_arrays(arrays=all_arrays),
    func=st.sampled_from(["first", "last", "nanfirst", "nanlast"]),
)
def test_first_last(data, array, func):
    by = data.draw(by_arrays(shape=(array.shape[-1],)))

    INVERSES = {"first": "last", "last": "first", "nanfirst": "nanlast", "nanlast": "nanfirst"}
    MATES = {"first": "nanfirst", "last": "nanlast", "nanfirst": "first", "nanlast": "last"}
    inverse = INVERSES[func]
    mate = MATES[func]

    if func in ["first", "last"]:
        array = array.rechunk((*array.chunks[:-1], -1))

    for arr in [array, array.compute()]:
        forward, fg = groupby_reduce(arr, by, func=func, engine="flox")
        reverse, rg = groupby_reduce(arr[..., ::-1], by[..., ::-1], func=inverse, engine="flox")

        assert forward.dtype == reverse.dtype
        assert forward.dtype == arr.dtype

        assert_equal(fg, rg)
        assert_equal(forward, reverse)

    if arr.dtype.kind == "f" and not np.isnan(array.compute()).any():
        if mate in ["first", "last"]:
            array = array.rechunk((*array.chunks[:-1], -1))

        first, _ = groupby_reduce(array, by, func=func, engine="flox")
        second, _ = groupby_reduce(array, by, func=mate, engine="flox")
        assert_equal(first, second)
