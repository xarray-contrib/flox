"""Tests for topk with xarray."""

import numpy as np
import pytest

# isort: off
xr = pytest.importorskip("xarray")
# isort: on

from flox.xarray import xarray_reduce

from . import requires_dask


def test_xarray_topk_basic():
    """Test basic topk functionality with xarray."""
    # Create test data with clear ordering
    data = np.array([[5, 1, 3, 8], [2, 9, 4, 7], [6, 0, 10, 1]])

    da = xr.DataArray(
        data,
        dims=("x", "y"),
        coords={"labels": ("y", ["a", "a", "b", "b"])},
    )

    # Test k=2 (top 2 values)
    result = xarray_reduce(
        da,
        "labels",
        func="topk",
        k=2,
    )

    # Check dimensions are correct
    assert "k" in result.dims
    assert result.sizes["k"] == 2
    assert result.sizes["labels"] == 2
    assert result.sizes["x"] == 3

    print(f"Result shape: {result.shape}")
    print(f"Result dims: {result.dims}")
    print(f"Result:\n{result.values}")


def test_xarray_topk_negative_k():
    """Test topk with negative k (bottom k values)."""
    data = np.array([[5, 1, 3, 8], [2, 9, 4, 7], [6, 0, 10, 1]])

    da = xr.DataArray(
        data,
        dims=("x", "y"),
        coords={"labels": ("y", ["a", "a", "b", "b"])},
    )

    # Test k=-2 (bottom 2 values)
    result = xarray_reduce(
        da,
        "labels",
        func="topk",
        k=-2,
    )

    # Check dimensions
    assert "k" in result.dims
    assert result.sizes["k"] == 2
    assert result.sizes["labels"] == 2


@requires_dask
def test_xarray_topk_dask():
    """Test topk with dask arrays."""
    import dask.array as dask_array

    data = np.array([[5, 1, 3, 8], [2, 9, 4, 7], [6, 0, 10, 1]])

    da = xr.DataArray(
        dask_array.from_array(data, chunks=(2, 2)),
        dims=("x", "y"),
        coords={"labels": ("y", ["a", "a", "b", "b"])},
    )

    result = xarray_reduce(
        da,
        "labels",
        func="topk",
        k=2,
    )

    # Force computation
    result = result.compute()

    # Check dimensions
    assert "k" in result.dims
    assert result.sizes["k"] == 2
    assert result.sizes["labels"] == 2
