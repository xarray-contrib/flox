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
