[![GitHub Workflow CI Status](https://img.shields.io/github/workflow/status/dcherian/flox/CI?logo=github&style=for-the-badge)](https://github.com/dcherian/flox/actions)[![GitHub Workflow Code Style Status](https://img.shields.io/github/workflow/status/dcherian/flox/code-style?label=Code%20Style&style=for-the-badge)](https://github.com/dcherian/flox/actions)[![image](https://img.shields.io/codecov/c/github/dcherian/flox.svg?style=for-the-badge)](https://codecov.io/gh/dcherian/flox)[![PyPI](https://img.shields.io/pypi/v/flox.svg?style=for-the-badge)](https://pypi.org/project/flox/)[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/flox.svg?style=for-the-badge)](https://anaconda.org/conda-forge/flox)[![Documentation Status](https://readthedocs.org/projects/flox/badge/?version=latest)](https://flox.readthedocs.io/en/latest/?badge=latest)

# flox

This project explores strategies for fast GroupBy reductions with dask.array. It used to be called `dask_groupby`
It was motivated by

1.  Dask Dataframe GroupBy
    [blogpost](https://blog.dask.org/2019/10/08/df-groupby)
2.  numpy_groupies in Xarray
    [issue](https://github.com/pydata/xarray/issues/4473)

(See a
[presentation](https://docs.google.com/presentation/d/1YubKrwu9zPHC_CzVBhvORuQBW-z148BvX3Ne8XcvWsQ/edit?usp=sharing)
about this package, from the Pangeo Showcase).

## Acknowledgements

This work was funded in part by NASA-ACCESS 80NSSC18M0156 "Community tools for analysis of NASA Earth Observing System
Data in the Cloud" (PI J. Hamman), and [NCAR's Earth System Data Science Initiative](https://ncar.github.io/esds/).
It was motivated by many discussions in the [Pangeo](https://pangeo.io) community.

## API

There are two main functions
1.  `flox.groupby_reduce(dask_array, by_dask_array, "mean")`
    "pure" dask array interface
1.  `flox.xarray.xarray_reduce(xarray_object, by_dataarray, "mean")`
    "pure" xarray interface; though [work is ongoing](https://github.com/pydata/xarray/pull/5734) to integrate this
    package in xarray.


## Implementation

See [the documentation](https://flox.readthedocs.io/en/latest/implementation.html) for details on the implementation.

## Custom reductions

`flox` implements all common reductions provided by `numpy_groupies` in `aggregations.py`.
It also allows you to specify a custom Aggregation (again inspired by dask.dataframe),
though this might not be fully functional at the moment. See `aggregations.py` for examples.

``` python
    mean = Aggregation(
        # name used for dask tasks
        name="mean",
        # operation to use for pure-numpy inputs
        numpy="mean",
        # blockwise reduction
        chunk=("sum", "count"),
        # combine intermediate results: sum the sums, sum the counts
        combine=("sum", "sum"),
        # generate final result as sum / count
        finalize=lambda sum_, count: sum_ / count,
        # Used when "reindexing" at combine-time
        fill_value=0,
        # Used when any member of `expected_groups` is not found
        final_fill_value=np.nan,
    )
```
