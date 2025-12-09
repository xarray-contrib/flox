from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from .types import DaskArray, Graph
from .xrutils import is_duck_dask_array, module_available

if TYPE_CHECKING:
    from .aggregations import Aggregation

    T_Agg: TypeAlias = str | Aggregation

T = TypeVar("T")

try:
    import dask.array as da

    dask_array_type = da.Array
except ImportError:
    dask_array_type = ()  # type: ignore[assignment, misc]

try:
    import sparse

    sparse_array_type = sparse.COO
except ImportError:
    sparse_array_type = ()

HAS_SPARSE = module_available("sparse")


@dataclass
class ArrayLayer:
    name: str
    layer: Graph
    chunks: tuple[tuple[int, ...], ...]

    def to_array(self, dep: DaskArray) -> DaskArray:
        from dask.array import Array
        from dask.highlevelgraph import HighLevelGraph

        graph = HighLevelGraph.from_collections(self.name, self.layer, dependencies=[dep])
        return Array(graph, self.name, self.chunks, meta=dep._meta)


def _postprocess_numbagg(result, *, func, fill_value, size, seen_groups):
    """Account for numbagg not providing a fill_value kwarg."""
    import numpy as np

    from .aggregate_numbagg import DEFAULT_FILL_VALUE

    if not isinstance(func, str) or func not in DEFAULT_FILL_VALUE:
        return result
    # The condition needs to be
    # len(found_groups) < size; if so we mask with fill_value (?)
    default_fv = DEFAULT_FILL_VALUE[func]
    needs_masking = fill_value is not None and not np.array_equal(fill_value, default_fv, equal_nan=True)
    groups = np.arange(size)
    if needs_masking:
        mask = np.isin(groups, seen_groups, assume_unique=True, invert=True)
        if mask.any():
            if isinstance(result, sparse_array_type):
                result.fill_value = fill_value
            else:
                result[..., groups[mask]] = fill_value
    return result


def identity(x: T) -> T:
    return x


def _issorted(arr, ascending=True) -> bool:
    if ascending:
        return bool((arr[:-1] <= arr[1:]).all())
    else:
        return bool((arr[:-1] >= arr[1:]).all())


def _should_auto_rechunk_blockwise(method, array, any_by_dask: bool, by) -> bool:
    """Check if we should attempt automatic rechunking for blockwise operations."""
    return method is None and is_duck_dask_array(array) and not any_by_dask and by.ndim == 1 and _issorted(by)


def _is_nanlen(reduction) -> bool:
    return isinstance(reduction, str) and reduction == "nanlen"


def _is_arg_reduction(func: T_Agg) -> bool:
    from .aggregations import Aggregation

    if isinstance(func, str) and func in ["argmin", "argmax", "nanargmax", "nanargmin"]:
        return True
    if isinstance(func, Aggregation) and func.reduction_type == "argreduce":
        return True
    return False


def _is_minmax_reduction(func: T_Agg) -> bool:
    return not _is_arg_reduction(func) and (isinstance(func, str) and ("max" in func or "min" in func))


def _is_first_last_reduction(func: T_Agg) -> bool:
    from .aggregations import Aggregation

    if isinstance(func, Aggregation):
        func = func.name
    return func in ["nanfirst", "nanlast", "first", "last"]


def _is_bool_supported_reduction(func: T_Agg) -> bool:
    from .aggregations import Aggregation

    if isinstance(func, Aggregation):
        func = func.name
    return (
        func in ["all", "any"]
        # TODO: enable in npg
        # or _is_first_last_reduction(func)
        # or _is_minmax_reduction(func)
    )


def _is_sparse_supported_reduction(func: T_Agg) -> bool:
    from .aggregations import SCANS, Aggregation

    if isinstance(func, Aggregation):
        func = func.name
    if func in SCANS:
        return False
    return not _is_arg_reduction(func) and any(f in func for f in ["len", "sum", "max", "min", "mean"])


def _is_reindex_sparse_supported_reduction(func: T_Agg) -> bool:
    from .aggregations import Aggregation

    if isinstance(func, Aggregation):
        func = func.name
    return HAS_SPARSE and all(f not in func for f in ["first", "last", "prod", "var", "std"])
