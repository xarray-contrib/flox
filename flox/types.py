from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
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

# Only define these types when type checking to avoid import issues
if TYPE_CHECKING:
    # Core array types
    T_DuckArray: TypeAlias = "np.ndarray | DaskArray | CubedArray"
    T_By: TypeAlias = T_DuckArray
    T_Bys = "tuple[T_By, ...]"

    # Expected groups types
    T_ExpectIndex = "pd.Index"
    T_ExpectIndexTuple = "tuple[T_ExpectIndex, ...]"
    T_ExpectIndexOpt = "T_ExpectIndex | None"
    T_ExpectIndexOptTuple = "tuple[T_ExpectIndexOpt, ...]"
    T_Expect = "Sequence | np.ndarray | T_ExpectIndex"
    T_ExpectTuple = "tuple[T_Expect, ...]"
    T_ExpectOpt = "Sequence | np.ndarray | T_ExpectIndexOpt"
    T_ExpectOptTuple = "tuple[T_ExpectOpt, ...]"
    T_ExpectedGroups = "T_Expect | T_ExpectOptTuple"
    T_ExpectedGroupsOpt = "T_ExpectedGroups | None"

    # Function and aggregation types
    T_Func = "str | Callable"
    T_Funcs = "T_Func | Sequence[T_Func]"
    T_Agg = "str"  # Will be "str | Aggregation" but avoiding circular import
    T_Scan = "str"  # Will be "str | Scan" but avoiding circular import

    # Axis types
    T_Axis = int
    T_Axes = "tuple[T_Axis, ...]"
    T_AxesOpt = "T_Axis | T_Axes | None"

    # Data types
    T_Dtypes = "np.typing.DTypeLike | Sequence[np.typing.DTypeLike] | None"
    T_FillValues = "np.typing.ArrayLike | Sequence[np.typing.ArrayLike] | None"

    # Engine and method types
    T_Engine = Literal["flox", "numpy", "numba", "numbagg"]
    T_EngineOpt = "None | T_Engine"
    T_Method = Literal["map-reduce", "blockwise", "cohorts"]
    T_MethodOpt = "None | Literal['map-reduce', 'blockwise', 'cohorts']"

    # Binning types
    T_IsBins = "bool | Sequence[bool]"

    # Factorize types
    FactorProps = "namedtuple('FactorProps', 'offset_group nan_sentinel nanmask')"
