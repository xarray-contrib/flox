from dataclasses import dataclass

import numpy as np
import pandas as pd

from .types import DaskArray, Graph


@dataclass
class ArrayLike:
    name: str
    layer: Graph
    chunks: tuple[tuple[int, ...], ...]
    prev_layer_name: str

    def to_array(self, dep: DaskArray) -> DaskArray:
        from dask.array import Array
        from dask.highlevelgraph import HighLevelGraph

        graph = HighLevelGraph.from_collections(self.name, self.layer, dependencies=[dep])
        return Array(graph, self.name, self.chunks, meta=dep._meta)


def _unique(a: np.ndarray) -> np.ndarray:
    """Much faster to use pandas unique and sort the results.
    np.unique sorts before uniquifying and is slow."""
    return np.sort(pd.unique(a.reshape(-1)))
