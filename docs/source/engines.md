(engines)=

# Engines

`flox` provides multiple options, using the `engine` kwarg, for computing the core GroupBy reduction on numpy or other array types other than dask.

1. `engine="numpy"` wraps `numpy_groupies.aggregate_numpy`. This uses indexing tricks and functions like `np.bincount`, or the ufunc `.at` methods
   (.e.g `np.maximum.at`) to provided reasonably performant aggregations.
1. `engine="numba"` wraps `numpy_groupies.aggregate_numba`. This uses `numba` kernels for the core aggregation.
1. `engine="flox"` uses the `ufunc.reduceat` method after first argsorting the array so that all group members occur sequentially. This was copied from
   a [gist by Stephan Hoyer](https://gist.github.com/shoyer/f538ac78ae904c936844)
1. `engine="numbagg"` uses the reductions available in [`numbagg.grouped`](https://github.com/numbagg/numbagg/blob/main/numbagg/grouped.py)
   from the [numbagg](https://github.com/numbagg/numbagg) project.

See [](arrays) for more details.

## Tradeoffs

For the common case of reducing a nD array by a 1D array of group labels (e.g. `groupby("time.month")`), `engine="numbagg"` is almost always faster, and `engine="flox"` _can_ be faster.

The reason is that `numpy_groupies` converts all groupby problems to a 1D problem, this can involve [some overhead](https://github.com/ml31415/numpy-groupies/pull/46).
It is possible to optimize this a bit in `flox` or `numpy_groupies`, but the work has not been done yet.
The advantage of `engine="numpy"` is that it tends to work for more array types, since it appears to be more common to implement `np.bincount`, and not `np.add.reduceat`.

```{tip}
One other potential engine we could add is [`datashader`](https://github.com/xarray-contrib/flox/issues/142).
Contributions or discussion is very welcome!
```
