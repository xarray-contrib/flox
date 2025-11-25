from collections import namedtuple
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

import numpy as np

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


if TYPE_CHECKING:
    T_DuckArray: TypeAlias = np.ndarray | DaskArray | CubedArray
    T_By: TypeAlias = T_DuckArray
    T_Bys = tuple[T_By, ...]
    T_Axis = int
    T_Axes = tuple[T_Axis, ...]


IntermediateDict = dict[str | Callable, Any]
FinalResultsDict = dict[str, DaskArray | CubedArray | np.ndarray]


class FactorizeKwargs(TypedDict, total=False):
    """Used in _factorize_multiple"""

    by: "T_Bys"
    axes: "T_Axes"
    fastpath: bool
    reindex: bool
    sort: bool


FactorProps = namedtuple("FactorProps", "offset_group nan_sentinel nanmask")
