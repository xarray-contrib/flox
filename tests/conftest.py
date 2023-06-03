import pytest


@pytest.fixture(scope="module", params=["flox", "numpy", "numba"])
def engine(request):
    if request.param == "numba":
        try:
            import numba  # noqa
        except ImportError:
            pytest.xfail()
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
