import pytest
from hypothesis import HealthCheck, Verbosity, settings

from . import requires_numbagg

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
