# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**flox** is a Python library providing fast GroupBy reduction operations for `dask.array`. It implements parallel-friendly GroupBy reductions using the MapReduce paradigm and integrates with xarray for labeled multidimensional arrays.

## Development Commands

### Environment Setup

```bash
# Create and activate development environment
mamba env create -f ci/environment.yml
conda activate flox-tests
python -m pip install --no-deps -e .
```

### Testing

```bash
# Run full test suite (as used in CI)
pytest --durations=20 --durations-min=0.5 -n auto --cov=./ --cov-report=xml --hypothesis-profile ci

# Run tests without coverage
pytest -n auto

# Run single test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::test_function_name
```

### Code Quality

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Format code with ruff
ruff format .

# Lint and fix with ruff
ruff check --fix .

# Type checking
mypy flox/

# Spell checking
codespell
```

### Benchmarking

```bash
# Performance benchmarking (from asv_bench/ directory)
cd asv_bench
asv run
asv publish
asv preview
```

## CI Configuration

### GitHub Workflows (`.github/workflows/`)

- **`ci.yaml`** - Main CI pipeline with test matrix across Python versions (3.11, 3.13) and operating systems (Ubuntu, Windows)
- **`ci-additional.yaml`** - Additional CI jobs including doctests and mypy type checking
- **`upstream-dev-ci.yaml`** - Tests against development versions of upstream dependencies
- **`pypi.yaml`** - PyPI publishing workflow
- **`testpypi-release.yaml`** - Test PyPI release workflow
- **`benchmarks.yml`** - Performance benchmarking workflow

### Environment Files (`ci/`)

- **`environment.yml`** - Main test environment with all dependencies
- **`minimal-requirements.yml`** - Minimal requirements testing (pandas==1.5, numpy==1.22, etc.)
- **`no-dask.yml`** - Testing without dask dependency
- **`no-numba.yml`** - Testing without numba dependency
- **`no-xarray.yml`** - Testing without xarray dependency
- **`env-numpy1.yml`** - Testing with numpy\<2 constraint
- **`docs.yml`** - Documentation building environment
- **`upstream-dev-env.yml`** - Development versions of dependencies
- **`benchmark.yml`** - Benchmarking environment

### ReadTheDocs Configuration

- **`.readthedocs.yml`** - ReadTheDocs configuration using `ci/docs.yml` environment

## Code Architecture

### Core Modules (`flox/`)

- **`core.py`** - Main reduction logic, central orchestrator of groupby operations
- **`aggregations.py`** - Defines the `Aggregation` class and built-in aggregation operations
- **`xarray.py`** - Primary integration with xarray, provides `xarray_reduce()` API
- **`dask_array_ops.py`** - Dask-specific array operations and optimizations

### Aggregation Backends (`flox/aggregate_*.py`)

- **`aggregate_flox.py`** - Native flox implementation
- **`aggregate_npg.py`** - numpy-groupies backend
- **`aggregate_numbagg.py`** - numbagg backend for JIT-compiled operations
- **`aggregate_sparse.py`** - Support for sparse arrays

### Utilities

- **`cache.py`** - Caching mechanisms for performance
- **`visualize.py`** - Tools for visualizing groupby operations
- **`lib.py`** - General utility functions
- **`xrutils.py`** & **`xrdtypes.py`** - xarray-specific utilities and types

### Main APIs

- `flox.groupby_reduce()` - Pure dask array interface
- `flox.xarray.xarray_reduce()` - Pure xarray interface

## Key Design Patterns

**Engine Selection**: The library supports multiple computation backends ("flox", "numpy", "numbagg") that can be chosen based on data characteristics and performance requirements.

**MapReduce Strategy**: Implements groupby reductions using a two-stage approach (blockwise + tree reduction) to avoid expensive sort/shuffle operations in parallel computing.

**Chunking Intelligence**: Automatically rechunks data to optimize groupby operations, particularly important for the current `auto-blockwise-rechunk` branch.

**Integration Testing**: Extensive testing against xarray's groupby functionality to ensure compatibility with the broader scientific Python ecosystem.

## Testing Configuration

- **Framework**: pytest with coverage, parallel execution (pytest-xdist), and property-based testing (hypothesis)
- **Coverage Target**: 95%
- **Test Environments**: Multiple conda environments test optional dependencies (no-dask, no-numba, no-xarray)
- **CI Matrices**: Tests across Python 3.11-3.13, Ubuntu/Windows, multiple dependency configurations

## Dependencies

**Core**: pandas>=1.5, numpy>=1.22, numpy_groupies>=0.9.19, scipy>=1.9, toolz, packaging>=21.3

**Optional**: cachey, dask, numba, numbagg, xarray (enable with `pip install flox[all]`)

## Development Notes

- Uses `setuptools_scm` for automatic versioning from git tags
- Heavy emphasis on performance with ASV benchmarking infrastructure
- Type hints throughout with mypy checking
- Pre-commit hooks enforce code quality (ruff, prettier, codespell)
- Integration testing with xarray upstream development branch
- **Python Support**: Minimum version 3.11 (updated from 3.10)
- **Git Worktrees**: `worktrees/` directory is ignored for development workflows
- **Running Tests**: Always use `uv run pytest` to run tests (not just `pytest`)

## Key Implementation Details

### Map-Reduce Combine Strategies (`flox/dask.py`)

There are two strategies for combining intermediate results in dask's tree reduction:

1. **`_simple_combine`**: Used for most reductions. Tree-reduces the reduction itself (not the groupby-reduction) for performance. Requirements:

   - All blocks must contain all groups after blockwise step (reindex.blockwise=True)
   - Must know expected_groups
   - Inserts DUMMY_AXIS=-2 via `_expand_dims`, reduces along it, then squeezes it out
   - Used when: not an arg reduction, not first/last with non-float dtype, and labels are known

1. **`_grouped_combine`**: More general solution that tree-reduces the groupby-reduction itself. Used for:

   - Arg reductions (argmax, argmin, etc.)
   - When labels are unknown (dask arrays without expected_groups)
   - First/last reductions with non-float dtypes

### Aggregations with New Dimensions

Some aggregations add new dimensions to the output (e.g., topk, quantile):

- **`new_dims_func`**: Function that returns tuple of Dim objects for new dimensions
- These MUST use `_simple_combine` because intermediate results have an extra dimension that needs to be reduced along DUMMY_AXIS
- Check if `new_dims_func(**finalize_kwargs)` returns non-empty tuple to determine if aggregation actually adds dimensions
- **Note**: argmax/argmin have `new_dims_func` but return empty tuple, so they use `_grouped_combine`

### topk Implementation

The topk aggregation is special:

- Uses `_simple_combine` (has non-empty new_dims_func)
- First intermediate (topk values) combines along axis 0, not DUMMY_AXIS
- Does NOT squeeze out DUMMY_AXIS in final aggregate step
- `_expand_dims` only expands non-topk intermediates (the second one, nanlen)

### Axis Parameter Handling

- **`_simple_combine`**: Always receives axis as tuple (e.g., `(-2,)` for DUMMY_AXIS)
- **numpy functions**: Most accept both tuple and integer axis (e.g., np.max, np.sum)
- **Exception**: argmax/argmin don't accept tuple axis, but these use `_grouped_combine`
- **Custom functions**: Like `_var_combine` should normalize axis to tuple if needed for iteration

### Test Organization

- **`test_groupby_reduce_all`**: Comprehensive test for all aggregations with various parameters (nby, chunks, etc.)

  - Tests both with and without NaN handling
  - For topk: sorts results along axis 0 before comparison (k dimension is at axis 0)
  - Uses `np.moveaxis` not `np.swapaxes` for topk to avoid swapping other dimensions

- **`test_groupby_reduce_axis_subset_against_numpy`**: Tests reductions over subsets of axes

  - Compares dask results against numpy results
  - Tests various axis combinations: None, single int, tuples
  - Skip arg reductions with axis=None or multiple axes (not supported)

### Common Pitfalls

1. **Axis transformations for topk**: Use `np.moveaxis(expected, src, 0)` not `np.swapaxes(expected, src, 0)` to move k dimension to position 0 without reordering other dimensions

1. **new_dims_func checking**: Check if it returns non-empty dimensions, not just if it exists (argmax has one that returns `()`)

1. **Axis parameter types**: Custom combine functions should handle both tuple and integer axis by normalizing at the start
