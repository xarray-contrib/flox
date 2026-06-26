from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from flox.core import groupby_reduce

from . import requires_dask


def _contains_expr(expr, name, seen=None):
    if seen is None:
        seen = set()
    if id(expr) in seen:
        return False
    seen.add(id(expr))
    if type(expr).__name__ == name:
        return True
    if not hasattr(expr, "dependencies"):
        return False
    return any(_contains_expr(getattr(dep, "expr", dep), name, seen) for dep in expr.dependencies())


def test_groupby_reduce_with_dask_array_returns_dask_array(dask_array_api):
    dax = dask_array_api
    x = dax.from_array(np.arange(6), chunks=(3,))
    labels = np.array([0, 0, 1, 1, 2, 2])

    result, groups = groupby_reduce(
        x,
        labels,
        func="sum",
        expected_groups=np.array([0, 1, 2]),
    )

    assert isinstance(result, dax.Array)
    np.testing.assert_array_equal(groups, np.array([0, 1, 2]))
    np.testing.assert_array_equal(result.compute(), np.array([1, 5, 9]))
    assert "FromGraph" not in type(result.expr).__name__


def test_groupby_reduce_with_dask_array_labels(dask_array_api):
    dax = dask_array_api
    x = dax.from_array(np.arange(6), chunks=(3,))
    labels = dax.from_array(np.array([0, 0, 1, 1, 2, 2]), chunks=(3,))

    result, groups = groupby_reduce(
        x,
        labels,
        func="sum",
        expected_groups=np.array([0, 1, 2]),
    )

    assert isinstance(result, dax.Array)
    np.testing.assert_array_equal(groups, np.array([0, 1, 2]))
    np.testing.assert_array_equal(result.compute(), np.array([1, 5, 9]))


def test_groupby_reduce_with_dask_array_unknown_groups_uses_expression(dask_array_api):
    dax = dask_array_api
    x = dax.from_array(np.arange(6), chunks=(3,))
    labels = dax.from_array(np.array([0, 0, 1, 1, 2, 2]), chunks=(3,))

    result, groups = groupby_reduce(x, labels, func="sum")

    assert isinstance(result, dax.Array)
    assert isinstance(groups, dax.Array)
    assert type(groups.expr).__name__ == "FloxExtractGroups"
    np.testing.assert_array_equal(groups.compute(), np.array([0, 1, 2]))
    np.testing.assert_array_equal(result.compute(), np.array([1, 5, 9]))


def test_groupby_reduce_with_dask_array_cohorts_uses_subset_expression(dask_array_api):
    dax = dask_array_api
    x = dax.from_array(np.arange(12).reshape(3, 4), chunks=(1, 2))
    labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3]])

    result, groups = groupby_reduce(
        x,
        labels,
        func="sum",
        method="cohorts",
        expected_groups=np.array([0, 1, 2, 3]),
        axis=(0, 1),
    )

    assert isinstance(result, dax.Array)
    assert _contains_expr(result.expr, "FloxSubsetBlocks")
    assert not _contains_expr(result.expr, "FromGraph")
    np.testing.assert_array_equal(groups, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(result.compute(), np.array([10, 18, 17, 21]))


def test_xarray_reduce_with_dask_array_data(dask_array_api):
    dax = dask_array_api
    xr = pytest.importorskip("xarray")
    from flox.xarray import xarray_reduce

    x = xr.DataArray(dax.from_array(np.arange(6), chunks=(3,)), dims="x")
    labels = xr.DataArray(np.array([0, 0, 1, 1, 2, 2]), dims="x", name="group")

    result = xarray_reduce(x, labels, func="sum")

    assert isinstance(result.data, dax.Array)
    np.testing.assert_array_equal(result.compute().values, np.array([1, 5, 9]))


@requires_dask
def test_importing_flox_does_not_register_dask_array_for_legacy_xarray_dask():
    code = """
import sys
import numpy as np
import dask.array as da
import xarray as xr
from flox.xarray import xarray_reduce

assert "dask_array" not in sys.modules
x = xr.DataArray(da.from_array(np.arange(6), chunks=(3,)), dims="x")
labels = xr.DataArray(np.array([0, 0, 1, 1, 2, 2]), dims="x", name="group")
result = xarray_reduce(x, labels, func="sum")
assert "dask_array" not in sys.modules
np.testing.assert_array_equal(result.compute().values, np.array([1, 5, 9]))
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        check=True,
        text=True,
        capture_output=True,
    )


def test_mixed_dask_backends_raise_clear_error(dask_array_api):
    dax = dask_array_api
    legacy_da = pytest.importorskip("dask.array")
    x = dax.from_array(np.arange(6), chunks=(3,))
    labels = legacy_da.from_array(np.array([0, 0, 1, 1, 2, 2]), chunks=(3,))

    with pytest.raises(TypeError, match="Cannot mix dask_array.Array"):
        groupby_reduce(
            x,
            labels,
            func="sum",
            expected_groups=np.array([0, 1, 2]),
        )

    legacy_x = legacy_da.from_array(np.arange(6), chunks=(3,))
    dax_labels = dax.from_array(np.array([0, 0, 1, 1, 2, 2]), chunks=(3,))

    with pytest.raises(TypeError, match="Cannot mix dask_array.Array"):
        groupby_reduce(
            legacy_x,
            dax_labels,
            func="sum",
            expected_groups=np.array([0, 1, 2]),
        )
