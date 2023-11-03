import pytest

from . import requires_numba, requires_numbagg


@pytest.fixture(
    scope="module",
    params=[
        "flox",
        "numpy",
        pytest.param("numba", marks=requires_numba),
        pytest.param("numbagg", marks=requires_numbagg),
    ],
)
def engine(request):
    return request.param


@pytest.fixture(scope="module", params=["numpy", "cupy"])
def array_module(request):
    if request.param == "cupy":
        try:
            import cupy  # noqa

            return cupy
        except ImportError:
            pytest.xfail()
    elif request.param == "numpy":
        import numpy

        return numpy
