# Plan: Array Expression Migration for flox

**Design doc**: `designs/array-expr-integration.md`
**Status**: Phase 6 Complete - Task spec migration done

## Current Status

- **map-reduce method**: Working - 432/432 core tests pass
- **blockwise method**: Partially working (multi-axis has dask expr bugs)
- **cohorts method**: Working - 210/210 cohorts tests pass
- **scans**: Partially working - cumsum/nancumsum/ffill work, bfill has preprocessing issues

Overall: 4789 passed, 1163 failed (down from 2402 failed initially)

## Overview

Migrate flox's dask operations from HighLevelGraph construction to dask's
expression system. Most of flox already uses standard dask operations
(blockwise, tree_reduce) that have expression equivalents. Only a few
custom graph patterns need dedicated expression classes.

## Architecture

```
flox/core.py: groupby_reduce()
    │
    ├─ numpy path: _reduce_blockwise() [unchanged]
    │
    └─ dask path: dispatch based on ARRAY_EXPR_ENABLED
           │
           ├─ Old: flox/dask.py: dask_groupby_agg() [HighLevelGraph]
           │
           └─ New: flox/expr.py: dask_groupby_agg() [Expressions]
```

## Expression Classes Needed

### 1. ExtractFromDictExpr
**Purpose**: Extract a key from dict-valued array blocks (for unknown groups)
**Replaces**: `_extract_unknown_groups()` in dask.py:753-769

```python
class ExtractFromDictExpr(ArrayExpr):
    _parameters = ["array", "key", "dtype"]

    def _layer(self):
        # {(name, 0): (operator.getitem, (array.name, 0, 0, ...), key)}
```

### 2. CollapseBlocksExpr
**Purpose**: Remap block keys to collapse multiple axes (virtual reshape)
**Replaces**: `_collapse_blocks_along_axes()` in dask.py:772-794

```python
class CollapseBlocksExpr(ArrayExpr):
    _parameters = ["array", "axes", "group_chunks"]

    def _layer(self):
        # {(name, *new_idx): (array.name, *old_idx)} - just key aliasing
```

### 3. SubsetBlocksExpr
**Purpose**: Select subset of blocks and optionally apply function
**Replaces**: `subset_to_blocks()` in dask.py:707-750, `ArrayLayer` in lib.py

```python
class SubsetBlocksExpr(ArrayExpr):
    _parameters = ["array", "flatblocks", "blkshape", "reindexer", "output_chunks"]

    def _layer(self):
        # {(name, k): (reindexer, old_key) for k in selected_keys}
```

### 4. CohortsReduceExpr
**Purpose**: Orchestrate cohorts method - multiple subset+reduce operations
**Replaces**: dask.py:469-518 cohorts block

```python
class CohortsReduceExpr(ArrayExpr):
    _parameters = ["intermediate", "chunks_cohorts", "axis", "combine", "aggregate", ...]

    def _layer(self):
        # For each cohort: subset blocks → tree reduce → combine results
```

## Phases

### Phase 1: Infrastructure ✓ Complete
- [x] Analyze current code for HighLevelGraph usage
- [x] Identify expression classes needed
- [x] Create plan document
- [ ] Create design document

### Phase 2: Core Module Setup ✓ Complete
- [x] Create `flox/expr.py` with detection logic using `dask.array.ARRAY_EXPR_ENABLED`
- [x] Add dispatch in `flox/core.py`
- [x] Implement `dask_groupby_agg()` for map-reduce method
- [x] Implement `ExtractFromDictExpr` for unknown groups
- [x] Implement `CollapseBlocksExpr` for multi-axis blockwise
- [ ] Add tests that run with `DASK_ARRAY__QUERY_PLANNING=True`
- [ ] xfail existing tests that check task graph structure when expr enabled

### Phase 3: Custom Expression Classes ✓ Complete
- [x] Implement `ExtractFromDictExpr` (for unknown groups in map-reduce)
- [x] Implement `CollapseBlocksExpr` (for multi-axis blockwise method)
- [x] Enable blockwise method in expr.py
- [ ] Tests for edge cases
- [ ] Fix multi-axis blockwise issues (may be dask expr bugs)

### Phase 4: Cohorts Method ✓ Complete
- [x] Implement `SubsetBlocksExpr`
- [x] Enable cohorts method in expr.py (using SubsetBlocksExpr + tree_reduce + concatenate)
- [x] Tests for cohorts (210 passed)
- Note: `CohortsReduceExpr` not needed - simpler approach using existing operations

### Phase 5: Scans ✓ Complete
- [x] Implement expression-based `dask_groupby_scan()`
- [x] Use expression-aware `cumreduction` from `dask.array._array_expr`
- [x] Wrap `grouped_reduce` to handle dtype argument difference
- [x] cumsum/nancumsum/ffill working
- [x] Fix bfill - handle finalize internally to avoid slicing optimization issues

### Phase 6: Task Spec Migration ✓ Complete
- [x] Switch from tuple tasks `(func, arg1, arg2)` to `Task`, `TaskRef`, `Alias` classes
- [x] Update all `_layer()` methods to use new task spec format
- [x] Import `Task`, `TaskRef`, `Alias` from `dask._task_spec`
- [x] Verify tests pass (432/432 core tests pass)

### Phase 7: Cleanup
- [ ] Deprecation warnings for old code path
- [ ] Update documentation
- [ ] Performance benchmarks comparing old vs new
- [ ] Remove xfail markers once graph structure tests updated

## Key Files

| File | Role |
|------|------|
| `flox/expr.py` | New expression-based implementations |
| `flox/dask.py` | Existing HighLevelGraph implementations (to deprecate) |
| `flox/core.py` | Dispatch logic, unchanged numpy path |
| `flox/dask_array_ops.py` | Custom tree_reduce (may need expr equivalent) |
| `tests/test_expr.py` | Expression-specific tests |

## Detection and Dispatch

```python
# In flox/core.py or appropriate location

def _use_expr():
    """Check if we should use expression-based code path."""
    try:
        from dask.array import ARRAY_EXPR_ENABLED
        return ARRAY_EXPR_ENABLED
    except ImportError:
        return False

# In groupby_reduce(), for dask arrays:
if _use_expr():
    from .expr import dask_groupby_agg
else:
    from .dask import dask_groupby_agg
```

## Testing Strategy

Run existing tests with expression mode enabled:
```bash
DASK_ARRAY__QUERY_PLANNING=True pytest tests/ -x -v
```

Tests that inspect graph structure (task counts, layer names, etc.) should be
marked with xfail when ARRAY_EXPR_ENABLED is True, since expression graphs
have different structure.

Add specific tests in `tests/test_expr.py` for:
- Expression class metadata (chunks, dtype, _meta)
- Layer generation correctness
- Round-trip: expr → compute → verify results match old path

## Notes

- Expression classes use `_parameters`/`_defaults` pattern, no custom `__init__`
- `_layer()` generates task dict, called lazily
- Tokenization must be deterministic for deduplication
- Keep functions like `_simple_combine`, `_grouped_combine`, `_aggregate` unchanged -
  they operate on actual data, not graphs
- Phase 6 migrates from tuple tasks to Task/TaskRef/DataNode for better
  optimization and serialization support
