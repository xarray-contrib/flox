import pytest


@pytest.fixture(scope="module", params=["numbagg"])
def engine(request):
    if request.param == "numba":
        try:
            import numba  # noqa
        except ImportError:
            pytest.skip()
    if request.param == "numbagg":
        try:
            import numbagg
        except ImportError:
            pytest.skip()

    return request.param
