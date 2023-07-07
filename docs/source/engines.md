(engines)=

# Engines

`flox` provides multiple options, using the `engine` kwarg, for computing the core GroupBy reduction on numpy or other array types other than dask.

1. `engine="numpy"` wraps `numpy_groupies.aggregate_numpy`. This uses indexing tricks and functions like `np.bincount`, or the ufunc `.at` methods
   (.e.g `np.maximum.at`) to provided reasonably performant aggregations.
1. `engine="numba"` wraps `numpy_groupies.aggregate_numba`. This uses `numba` kernels for the core aggregation.
1. `engine="flox"` uses the `ufunc.reduceat` method after first argsorting the array so that all group members occur sequentially. This was copied from
   a [gist by Stephan Hoyer](https://gist.github.com/shoyer/f538ac78ae904c936844)

See [](arrays) for more details.

## Tradeoffs

For the common case of reducing a nD array by a 1D array of group labels (e.g. `groupby("time.month")`), `engine="flox"` *can* be faster.

The reason is that `numpy_groupies` converts all groupby problems to a 1D problem, this can involve [some overhead](https://github.com/ml31415/numpy-groupies/pull/46).
It is possible to optimize this a bit in `flox` or `numpy_groupies`, but the work has not been done yet.
The advantage of `engine="numpy"` is that it tends to work for more array types, since it appears to be more common to implement `np.bincount`, and not `np.add.reduceat`.

```{tip}
Other potential engines we could add are [`numbagg`](https://github.com/numbagg/numbagg) ([stalled PR here](https://github.com/xarray-contrib/flox/pull/72)) and [`datashader`](https://github.com/xarray-contrib/flox/issues/142).
Both use numba for high-performance aggregations. Contributions or discussion is very welcome!
```
