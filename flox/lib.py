from dataclasses import dataclass

from .types import DaskArray, Graph

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
