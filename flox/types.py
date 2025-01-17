from typing import Any, TypeAlias

try:
    import cubed.Array as CubedArray
except ImportError:
    CubedArray = Any

try:
    import dask.array.Array as DaskArray
    from dask.typing import Graph
except ImportError:
    DaskArray = Any
    Graph: TypeAlias = Any  # type: ignore[no-redef,misc]
