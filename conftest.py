"""Configuration for pytest."""

import pytest


@pytest.fixture(scope="module", params=["flox", "numpy", "numba"])
def engine(request):
    if request.param == "numba":
        try:
            import numba  # noqa: F401
        except ImportError:
            pytest.xfail()
    return request.param
