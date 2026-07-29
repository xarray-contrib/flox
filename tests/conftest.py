import pytest
from hypothesis import HealthCheck, Verbosity, settings

from . import has_dask_array, requires_numbagg

settings.register_profile(
    "ci",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)
settings.register_profile(
    "default",
    max_examples=300,
    deadline=500,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    verbosity=Verbosity.verbose,
)
settings.load_profile("default")


@pytest.fixture(
    scope="module",
    params=[
        "flox",
        "numpy",
        # pytest.param("numba", marks=requires_numba),
        pytest.param("numbagg", marks=requires_numbagg),
    ],
)
def engine(request):
    return request.param


_MISSING = object()


def _snapshot_xarray_dask_manager():
    try:
        from xarray.namedarray.parallelcompat import list_chunkmanagers
    except ImportError:
        return None, _MISSING

    managers = list_chunkmanagers()
    return managers, managers.get("dask", _MISSING)


def _restore_xarray_dask_manager(managers, original):
    if managers is None:
        return
    if original is _MISSING:
        managers.pop("dask", None)
    else:
        managers["dask"] = original


def _activate_dask_array():
    import dask_array as da

    da.xarray.register()
    return da


@pytest.fixture
def dask_array_api():
    if not has_dask_array:
        pytest.skip("requires dask_array")

    managers, original = _snapshot_xarray_dask_manager()
    da = _activate_dask_array()

    try:
        yield da
    finally:
        _restore_xarray_dask_manager(managers, original)


@pytest.fixture
def chunked_array_api(request):
    try:
        backend = request.param
    except AttributeError:
        raise ValueError("chunked_array_api must be parametrized indirectly") from None

    if backend == "dask":
        import dask.array as da

        yield da
        return

    if backend != "dask_array":
        raise ValueError(f"Unknown chunked array backend {backend!r}")

    if not has_dask_array:
        pytest.skip("requires dask_array")

    managers, original = _snapshot_xarray_dask_manager()
    da = _activate_dask_array()
    try:
        yield da
    finally:
        _restore_xarray_dask_manager(managers, original)
