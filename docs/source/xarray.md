(xarray)=

# Xarray

Xarray will use flox by default (if installed) for DataArrays containing numpy and dask arrays. The default choice is `method="cohorts"` which generalizes
the best. Pass flox-specific kwargs to the specific reduction method:

```python
ds.groupby("time.month").mean(method="map-reduce", engine="flox")
ds.groupby_bins("lon", bins=[0, 10, 20]).mean(method="map-reduce")
ds.resample(time="M").mean(method="blockwise")
```

{py:func}`flox.xarray.xarray_reduce` used to provide extra functionality, but now Xarray's GroupBy object has been upgraded to match those capabilities with better API!
