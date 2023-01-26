---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

```{eval-rst}
.. currentmodule:: flox
```

# 10 minutes to flox

## GroupBy single variable

```{code-cell}
import numpy as np
import xarray as xr

from flox.xarray import xarray_reduce

labels = xr.DataArray(
    [1, 2, 3, 1, 2, 3, 0, 0, 0],
    dims="x",
    name="label",
)
labels
```

### With numpy

```{code-cell}
da = xr.DataArray(
    np.ones((9,)), dims="x", name="array"
)
```

Apply the reduction using {py:func}`flox.xarray.xarray_reduce` specifying the reduction operation in `func`

```{code-cell}
xarray_reduce(da, labels, func="sum")
```

### With dask

Let's first chunk `da` and `labels`

```{code-cell}
da_chunked = da.chunk(x=2)
labels_chunked = labels.chunk(x=3)
```

Grouping a dask array by a numpy array is unchanged

```{code-cell}
xarray_reduce(da_chunked, labels, func="sum")
```

When grouping **by** a dask array, we need to specify the "expected group labels" on the output so we can construct the result DataArray.
Without the `expected_groups` kwarg, an error is raised

```{code-cell}
---
tags: [raises-exception]
---
xarray_reduce(da_chunked, labels_chunked, func="sum")
```

Now we specify `expected_groups`:

```{code-cell}
dask_result = xarray_reduce(
    da_chunked, labels_chunked, func="sum", expected_groups=[0, 1, 2, 3],
)
dask_result
```

Note that any group labels not present in `expected_groups` will be ignored.
You can also provide `expected_groups` for the pure numpy GroupBy.

```{code-cell}
numpy_result = xarray_reduce(
    da, labels, func="sum", expected_groups=[0, 1, 2, 3],
)
numpy_result
```

The two are identical:

```{code-cell}
numpy_result.identical(dask_result)
```

## Binning by a single variable

For binning, specify the bin edges in `expected_groups` using {py:class}`pandas.IntervalIndex`:

```{code-cell}
import pandas as pd

xarray_reduce(
    da,
    labels,
    func="sum",
    expected_groups=pd.IntervalIndex.from_breaks([0.5, 1.5, 2.5, 6]),
)
```

Similarly for dask inputs

```{code-cell}
xarray_reduce(
    da_chunked,
    labels_chunked,
    func="sum",
    expected_groups=pd.IntervalIndex.from_breaks([0.5, 1.5, 2.5, 6]),
)
```

For more control over the binning (which edge is closed), pass the appropriate kwarg to {py:class}`pandas.IntervalIndex`:

```{code-cell}
xarray_reduce(
    da_chunked,
    labels_chunked,
    func="sum",
    expected_groups=pd.IntervalIndex.from_breaks([0.5, 1.5, 2.5, 6], closed="left"),
)
```

## Grouping by multiple variables

```{code-cell}
arr = np.ones((4, 12))
labels1 = np.array(["a", "a", "c", "c", "c", "b", "b", "c", "c", "b", "b", "f"])
labels2 = np.array([1, 2, 2, 1])

da = xr.DataArray(
    arr, dims=("x", "y"), coords={"labels2": ("x", labels2), "labels1": ("y", labels1)}
)
da
```

To group by multiple variables simply pass them as `*args`:

```{code-cell}
xarray_reduce(da, "labels1", "labels2", func="sum")
```

## Histogramming (Binning by multiple variables)

An unweighted histogram is simply a groupby multiple variables with count.

```{code-cell} python
arr = np.ones((4, 12))
labels1 = np.array(np.linspace(0, 10, 12))
labels2 = np.array([1, 2, 2, 1])

da = xr.DataArray(
    arr, dims=("x", "y"), coords={"labels2": ("x", labels2), "labels1": ("y", labels1)}
)
da
```

Specify bins in `expected_groups`

```{code-cell} python
xarray_reduce(
    da,
    "labels1",
    "labels2",
    func="count",
    expected_groups=(
        pd.IntervalIndex.from_breaks([-0.5, 4.5, 6.5, 8.9]),  # labels1
        pd.IntervalIndex.from_breaks([0.5, 1.5, 1.9]),  # labels2
    ),
)
```

## Resampling

Use the xarray interface i.e. `da.resample(time="M").mean()`.

Optionally pass [`method="blockwise"`](method-blockwise): `da.resample(time="M").mean(method="blockwise")`
