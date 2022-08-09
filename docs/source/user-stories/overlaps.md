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

Generally group-by problems involve non-overlapping groups. Consider the following group of labels:

```{code-cell}
import numpy as np
import xarray as xr

from flox.xarray import xarray_reduce

labels = xr.DataArray(
    [1, 2, 3, 1, 2, 3, 0, 0, 0],
    dims="x", name="label"
)
labels
```

These labels are non-overlapping. So when we reduce this data array over those labels
```{code-cell}
da = xr.ones_like(labels)
da
```
we get

```{code-cell}
xarray_reduce(da, labels, func="sum")
```

Now let's calculate the `sum` where `labels` is either `1` or `2`. The trick is to add a new dimension to `labels` of size `2` and assign a new label `4` in the appropriate locations.
```{code-cell}
# Add a new dimension `y`. This is a "view" and not writeable by default
# So we add a copy
expanded = labels.expand_dims(y=2).copy()
# Along y=0, expanded and labels are identical
# Along y=1, assign expanded=4 where label == 1 or 2, and -1 otherwise
expanded.loc[{"y": 1}] = xr.where(labels.isin([1, 2]), 4, -1)
expanded
```

Now when we reduce, we get the appropriate sum under `label=4`, and can discard the rest accumulated under `label=-1`.
```{code-cell}
xarray_reduce(da, expanded, func="sum")
```
