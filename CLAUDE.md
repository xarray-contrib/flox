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
