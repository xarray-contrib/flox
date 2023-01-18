(xarray)=

# Xarray

Xarray will use flox by default (if installed) for DataArrays containing numpy and dask arrays. The default choice is `method="cohorts"` which generalizes
the best. Pass flox-specific kwargs to the specific reduction method:

```python
ds.groupby("time.month").mean(method="map-reduce", engine="flox")
ds.groupby_bins("lon", bins=[0, 10, 20]).mean(method="map-reduce")
ds.resample(time="M").mean(method="blockwise")
```

Xarray's GroupBy operations are currently limited:

1. One can only group by a single variable.
1. When grouping by a dask array, that array will be computed to discover the unique group labels, and their locations

These limitations can be avoided by using {py:func}`flox.xarray.xarray_reduce` which allows grouping by multiple variables, lazy grouping by dask variables,
as well as an arbitrary combination of categorical grouping and binning. For example,

```python
flox.xarray.xarray_reduce(
    ds,
    ds.time.dt.month,
    ds.lon,
    func="mean",
    expected_groups=[None, [0, 10, 20]],
    isbin=[False, True],
    method="map-reduce",
)
```
