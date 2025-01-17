from typing import Any

try:
    import cubed.Array as CubedArray
except ImportError:
    CubedArray = Any

try:
    import dask.array.Array as DaskArray
    from dask.typing import Graph
except ImportError:
    DaskArray = Any
    Graph = Any
