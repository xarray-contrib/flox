import pytest


@pytest.fixture(
    scope="module",
    params=[
        # "flox",
        "numpy",
        # pytest.param("numba", marks=requires_numba)
    ],
)
def engine(request):
    return request.param
