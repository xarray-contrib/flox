import pytest
import xarray as xr


def test_foo():
    assert "foo" == "foo"


@pytest.mark.parametrize("dataset_name", ["rasm", "air_temperature"])
def test_mean_func(dataset_name):
    from dask_groupby.core import my_mean_func

    ds = xr.tutorial.open_dataset(dataset_name)
    results = my_mean_func(ds)
    assert isinstance(results, xr.Dataset)
