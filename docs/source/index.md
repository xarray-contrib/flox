# flox: fast & furious GroupBy reductions for `dask.array`

## Overview

[![GitHub Workflow CI Status](https://img.shields.io/github/workflow/status/dcherian/flox/CI?logo=github&style=flat)](https://github.com/dcherian/flox/actions)
[![GitHub Workflow Code Style Status](https://img.shields.io/github/workflow/status/dcherian/flox/code-style?label=Code%20Style&style=flat)](https://github.com/dcherian/flox/actions)
[![image](https://img.shields.io/codecov/c/github/dcherian/flox.svg?style=flat)](https://codecov.io/gh/dcherian/flox)
[![PyPI](https://img.shields.io/pypi/v/flox.svg?style=flat)](https://pypi.org/project/flox/)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/flox.svg?style=flat)](https://anaconda.org/conda-forge/flox)

This project explores strategies for fast GroupBy reductions with dask.array. It used to be called `dask_groupby`. It was motivated by

1.  Dask Dataframe GroupBy
    [blogpost](https://blog.dask.org/2019/10/08/df-groupby)
2.  numpy_groupies in Xarray
    [issue](https://github.com/pydata/xarray/issues/4473)

See a presentation ([video](https://discourse.pangeo.io/t/november-17-2021-flox-fast-furious-groupby-reductions-with-dask-at-pangeo-scale/2016), [slides](https://docs.google.com/presentation/d/1YubKrwu9zPHC_CzVBhvORuQBW-z148BvX3Ne8XcvWsQ/edit?usp=sharing)) about this package, from the Pangeo Showcase.

## Installing

``` shell
$ pip install flox
```

``` shell
$ conda install -c conda-forge flox
```

## API

There are two main functions
1.  {py:func}`flox.core.groupby_reduce`
    "pure" dask array interface
1.  {py:func}`flox.xarray.xarray_reduce`
    "pure" xarray interface; though [work is ongoing](https://github.com/pydata/xarray/pull/5734) to integrate this
    package in xarray.

## Acknowledgements

This work was funded in part by NASA-ACCESS 80NSSC18M0156 "Community tools for analysis of NASA Earth Observing System
Data in the Cloud" (PI J. Hamman), and [NCAR's Earth System Data Science Initiative](https://ncar.github.io/esds/).
It was motivated by many discussions in the [Pangeo](https://pangeo.io) community.

## Contents
```{eval-rst}
.. toctree::
   :maxdepth: 1

   implementation.md
   custom.md
   api.rst
   user-stories.md
```
