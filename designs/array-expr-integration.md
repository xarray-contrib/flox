# Design: Array Expression Integration for flox

## Motivation

Dask is transitioning from direct graph construction (HighLevelGraph) to an
expression-based system. This enables query optimization, compact representation,
and consistency with dask-dataframe. flox needs to work with the new system.

Benefits of expression-based flox:
- **Joint optimization**: flox operations optimized alongside other dask ops
- **Reduced overhead**: Expression trees are smaller than materialized graphs
- **Future-proof**: Aligns with dask's direction

## Core Principle: Reuse Standard Operations

Most of flox's dask code already uses standard operations:
- `dask.array.blockwise()` - chunk-level reduction
- `dask.array.reductions._tree_reduce()` - tree reduction

When `ARRAY_EXPR_ENABLED=True`, these return expression-based arrays. We reuse
them directly rather than reimplementing.

Only a few patterns require custom expression classes:
- Extracting group labels from dict-valued blocks
- Collapsing block structure across axes
- Cohorts method orchestration

## Detection and Dispatch

```python
from dask.array import ARRAY_EXPR_ENABLED

if ARRAY_EXPR_ENABLED:
    from .expr import dask_groupby_agg
else:
    from .dask import dask_groupby_agg
```

## Expression Classes

### ExtractFromDictExpr

Extracts a key from the first block of a dict-valued array. Used when group
labels are discovered at compute time.

```
reduced[0, 0, ...]["groups"] → groups array
```

### CollapseBlocksExpr

Remaps block keys to collapse multiple reduction axes into one. Pure key
aliasing, no computation.

```
blocks[(i, j, k)] → blocks[(i, linear_index(j, k))]
```

### SubsetBlocksExpr / CohortsReduceExpr (future)

For cohorts method: selects specific blocks and orchestrates multiple
independent reductions.

## Non-Goals

- Rewrite core reduction logic (reuse existing functions)
- Support non-dask backends initially (cubed etc.)
- Expression-level algebraic rewrites for groupby (future work)

## Files

- `flox/expr.py` - Expression classes and expression-aware dask_groupby_agg
- `flox/dask.py` - Original HighLevelGraph implementation (to deprecate)
- `flox/core.py` - Dispatch based on ARRAY_EXPR_ENABLED
