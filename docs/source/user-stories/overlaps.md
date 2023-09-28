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

# Overlapping Groups

This post is motivated by the problem of computing the [Meridional Overturning Circulation](https://en.wikipedia.org/wiki/Atlantic_meridional_overturning_circulation).
One of the steps is a binned average over latitude, over regions of the World Ocean. Commonly we want to average
globally, as well as over the Atlantic, and the Indo-Pacific. Generally group-by problems involve non-overlapping
groups. In this example, the "global" group overlaps with the "Indo-Pacific" and "Atlantic" groups. Below we consider a simplified version of this problem.

Consider the following labels:

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

These labels are non-overlapping. So when we reduce this data array over those labels along `x`

```{code-cell}
da = xr.ones_like(labels)
da
```

we get (note the reduction over `x` is implicit here):

```{code-cell}
xarray_reduce(da, labels, func="sum")
```

Now let's _also_ calculate the `sum` where `labels` is either `1` or `2`.
We could easily compute this using the grouped result but here we use this simple example for illustration.
The trick is to add a new dimension with new labels (here `4`) in the appropriate locations.

```{code-cell}
# assign 4 where label == 1 or 2, and -1 otherwise
newlabels = xr.where(labels.isin([1, 2]), 4, -1)

# concatenate along a new dimension y;
# note y is not present on da
expanded = xr.concat([labels, newlabels], dim="y")
expanded
```

Now we reduce over `x` _and_ the new dimension `y` (again implicitly) to get the appropriate sum under
`label=4` (and `label=-1`). We can discard the value accumulated under `label=-1` later.

```{code-cell}
xarray_reduce(da, expanded, func="sum")
```

This way we compute all the reductions we need, in a single pass over the data.

This technique generalizes to more complicated aggregations. The trick is to

- generate appropriate labels
- concatenate these new labels along a new dimension (`y`) absent on the object being reduced (`da`), and
- reduce over that new dimension in addition to any others.
