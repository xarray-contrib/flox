[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/xarray-contrib/flox/ci.yaml?branch=main&logo=github&style=flat)](https://github.com/xarray-contrib/flox/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/xarray-contrib/flox/main.svg)](https://results.pre-commit.ci/latest/github/xarray-contrib/flox/main)
[![image](https://img.shields.io/codecov/c/github/xarray-contrib/flox.svg?style=flat)](https://codecov.io/gh/xarray-contrib/flox)
[![Documentation Status](https://readthedocs.org/projects/flox/badge/?version=latest)](https://flox.readthedocs.io/en/latest/?badge=latest)

[![PyPI](https://img.shields.io/pypi/v/flox.svg?style=flat)](https://pypi.org/project/flox/)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/flox.svg?style=flat)](https://anaconda.org/conda-forge/flox)

[![NASA-80NSSC18M0156](https://img.shields.io/badge/NASA-80NSSC18M0156-blue)](https://earthdata.nasa.gov/esds/competitive-programs/access/pangeo-ml)
[![NASA-80NSSC22K0345](https://img.shields.io/badge/NASA-80NSSC22K0345-blue)](https://science.nasa.gov/open-science-overview)

# flox

This project explores strategies for fast GroupBy reductions with dask.array. It used to be called `dask_groupby`
It was motivated by

1. Dask Dataframe GroupBy
   [blogpost](https://blog.dask.org/2019/10/08/df-groupby)
1. [numpy_groupies](https://github.com/ml31415/numpy-groupies) in Xarray
   [issue](https://github.com/pydata/xarray/issues/4473)

(See a
[presentation](https://docs.google.com/presentation/d/1YubKrwu9zPHC_CzVBhvORuQBW-z148BvX3Ne8XcvWsQ/edit?usp=sharing)
about this package, from the Pangeo Showcase).

## Acknowledgements

This work was funded in part by

1. NASA-ACCESS 80NSSC18M0156 "Community tools for analysis of NASA Earth Observing System
   Data in the Cloud" (PI J. Hamman, NCAR),
1. NASA-OSTFL 80NSSC22K0345 "Enhancing analysis of NASA data with the open-source Python Xarray Library" (PIs Scott Henderson, University of Washington; Deepak Cherian, NCAR; Jessica Scheick, University of New Hampshire), and
1. [NCAR's Earth System Data Science Initiative](https://ncar.github.io/esds/).

It was motivated by [very](https://github.com/pangeo-data/pangeo/issues/266) [very](https://github.com/pangeo-data/pangeo/issues/271) [many](https://github.com/dask/distributed/issues/2602) [discussions](https://github.com/pydata/xarray/issues/2237) in the [Pangeo](https://pangeo.io) community.

## API

There are two main functions

1. `flox.groupby_reduce(dask_array, by_dask_array, "mean")`
   "pure" dask array interface
1. `flox.xarray.xarray_reduce(xarray_object, by_dataarray, "mean")`
   "pure" xarray interface; though [work is ongoing](https://github.com/pydata/xarray/pull/5734) to integrate this
   package in xarray.

## Implementation

See [the documentation](https://flox.readthedocs.io/en/latest/implementation.html) for details on the implementation.

## Custom reductions

`flox` implements all common reductions provided by `numpy_groupies` in `aggregations.py`.
It also allows you to specify a custom Aggregation (again inspired by dask.dataframe),
though this might not be fully functional at the moment. See `aggregations.py` for examples.

```python
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
