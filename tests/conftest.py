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
