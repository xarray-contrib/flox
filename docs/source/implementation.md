(algorithms)=

# Parallel Algorithms

`flox` outsources the core GroupBy operation to the vectorized implementations controlled by the
[`engine` kwarg](engines.md). Applying these implementations on a parallel array type like dask
can be hard. Performance strongly depends on how the groups are distributed amongst the blocks of an array.

`flox` implements 4 strategies for grouped reductions, each is appropriate for a particular distribution of groups
among the blocks of a dask array. Switch between the various strategies by passing `method`
and/or `reindex` to either {py:func}`flox.groupby_reduce` or {py:func}`flox.xarray.xarray_reduce`.

Your options are:

1. [`method="map-reduce"` with `reindex=False`](map-reindex-false)
1. [`method="map-reduce"` with `reindex=True`](map-reindex-True)
1. [`method="blockwise"`](method-blockwise)
1. [`method="cohorts"`](method-cohorts)

The most appropriate strategy for your problem will depend on the chunking of your dataset,
and the distribution of group labels across those chunks.

```{tip}
Currently these strategies are implemented for dask. We would like to generalize to other parallel array types
as appropriate (e.g. Ramba, cubed, arkouda). Please open an issue to discuss if you are interested.
```

(xarray-split)=

## Background: Xarray's current GroupBy strategy

Xarray's current strategy is to find all unique group labels, index out each group,
and then apply the reduction operation. Note that this only works if we know the group
labels (i.e. you cannot use this strategy to group by a dask array).

Schematically, this looks like (colors indicate group labels; separated groups of colors
indicate different blocks of an array):

```{image} ../diagrams/new-split-apply-combine-annotated.svg
---
alt: xarray-current-strategy
width: 100%
---
```

The first step is to extract all members of a group, which involves a _lot_ of
communication and is quite expensive (in dataframe terminology, this is a "shuffle").
This is fundamentally why many groupby reductions don't work well right now with
big datasets.

## `method="map-reduce"`

![map-reduce-strategy-schematic](/../diagrams/map-reduce.png)

The "map-reduce" strategy is inspired by `dask.dataframe.groupby`).
The GroupBy reduction is first applied blockwise. Those intermediate results are
combined by concatenating to form a new array which is then reduced
again. The combining of intermediate results uses dask's `_tree_reduce`
till all group results are in one block. At that point the result is
"finalized" and returned to the user.

### General Tradeoffs

1. This approach works well when either the initial blockwise reduction is effective, or if the
   reduction at the first combine step is effective. Here "effective" means we have multiple members of a single
   group in a block so the blockwise application of groupby-reduce actually reduces values and releases some memory.
1. One downside is that the final result will only have one chunk along the new group axis.
1. We have two choices for how to construct the intermediate arrays. See below.

(map-reindex-True)=

### `reindex=True`

If we know all the group labels, we can do so right at the blockwise step (`reindex=True`). This matches `dask.array.histogram` and
`xhistogram`, where the bin edges, or group labels oof the output, are known. The downside is the potential of large memory use
if number of output groups is much larger than number of groups in a block.

```{image} ../diagrams/new-map-reduce-reindex-True-annotated.svg
---
alt: map-reduce-reindex-True-strategy-schematic
width: 100%
---
```

(map-reindex-False)=

### `reindex=False`

We can `reindex` at the combine stage to groups present in the blocks being combined (`reindex=False`). This can limit memory use at the cost
of a performance reduction due to extra copies of the intermediate data during reindexing.

```{image} ../diagrams/new-map-reduce-reindex-False-annotated.svg
---
alt: map-reduce-reindex-True-strategy-schematic
width: 100%
---
```

This approach allows grouping by a dask array so group labels can be discovered at compute time, similar to `dask.dataframe.groupby`.

### Example

For example, consider `groupby("time.month")` with monthly frequency data and chunksize of 4 along `time`.
![cohorts-schematic](/../diagrams/cohorts-month-chunk4.png)
With `reindex=True`, each block will become 3x its original size at the blockwise step: input blocks have 4 timesteps while output block
has a value for all 12 months. One could use `reindex=False` to control memory usage but also see [`method="cohorts"`](method-cohorts) below.

(method-blockwise)=

## `method="blockwise"`

One case where `method="map-reduce"` doesn't work well is the case of "resampling" reductions. An
example here is resampling from daily frequency to monthly frequency data: `da.resample(time="M").mean()`
For resampling type reductions,

1. Group members occur sequentially (all days in January 2001 occur one after the other)
1. All groups not of exactly equal length (31 days in January but 28 in most Februaries)
1. All members in a group are next to each other (if the time series is sorted, which it
   usually is).
1. Because there can be a large number of groups, concatenating results for all groups in a single chunk could be catastrophic.

In this case, it makes sense to use `dask.dataframe` resample strategy which is to rechunk using {py:func}`flox.rechunk_for_blockwise`
so that all members of a group are in a single block. Then, the groupby operation can be applied blockwise.

```{image} ../diagrams/new-blockwise-annotated.svg
---
alt: blockwise-strategy-schematic
width: 100%
---
```

_Tradeoffs_

1. Only works for certain groupings.
1. Group labels must be known at graph construction time, so this only works for numpy arrays
1. Currently the rechunking is only implemented for 1D arrays (being motivated by time resampling),
   but a nD generalization seems possible.
1. Only can use the `blockwise` strategy for grouping by `nD` arrays.
1. Works better when multiple groups are already in a single block; so that the initial
   rechunking only involves a small amount of communication.

(method-cohorts)=

## `method="cohorts"`

The `map-reduce` strategy is quite effective but can involve some unnecessary communication. It can be possible to exploit
patterns in how group labels are distributed across chunks (similar to `method="blockwise"` above). Two cases are illustrative:

1. Groups labels can be _approximately-periodic_: e.g. `time.dayofyear` (period 365 or 366) or `time.month` (period 12).
   Consider our earlier example, `groupby("time.month")` with monthly frequency data and chunksize of 4 along `time`.
   ![cohorts-schematic](/../diagrams/cohorts-month-chunk4.png)
   Because a chunksize of 4 evenly divides the number of groups (12) all we need to do is index out blocks
   0, 3, 7 and then apply the `"map-reduce"` strategy to form the final result for months Jan-Apr. Repeat for the
   remaining groups of months (May-Aug; Sep-Dec) and then concatenate.

1. Groups can be _spatially localized_ like the blockwise case above, for example grouping by country administrative boundaries like
   counties or districts. In this case, concatenating the result for the northwesternmost county or district and the southeasternmost
   district can involve a lot of wasteful communication (again depending on chunking).

For such cases, we can adapt xarray's shuffling or subsetting strategy by indexing out "cohorts" or group labels
that tend to occur next to each other.

### A motivating example : time grouping

One example is the construction of "climatologies" which is a climate science term for something like `groupby("time.month")`
("monthly climatology") or `groupby("time.dayofyear")` ("daily climatology"). In these cases,

1. Groups occur sequentially (day 2 is always after day 1; and February is always after January)
1. Groups are approximately periodic (some years have 365 days and others have 366)

Consider our earlier example, `groupby("time.month")` with monthly frequency data and chunksize of 4 along `time`.
![cohorts-schematic](/../diagrams/cohorts-month-chunk4.png)

With `method="map-reduce", reindex=True`, each block will become 3x its original size at the blockwise step: input blocks have 4 timesteps while output block
has a value for all 12 months. Note that the blockwise groupby-reduction _does not reduce_ the data since there is only one element in each
group. In addition, since `map-reduce` will make the final result have only one chunk of size 12 along the new `month`
dimension, the final result has chunk sizes 3x that of the input, which may not be ideal.

However, because a chunksize of 4 evenly divides the number of groups (12) all we need to do is index out blocks
0, 3, 7 and then apply the `"map-reduce"` strategy to form the final result for months Jan-Apr. Repeat for the
remaining groups of months (May-Aug; Sep-Dec) and then concatenate. This is the essence of `method="cohorts"`

### Summary

We can generalize this idea for more complicated problems (inspired by the `split_out`kwarg in `dask.dataframe.groupby`)
We first apply the groupby-reduction blockwise, then split and reindex blocks to create a new array with which we complete the reduction
using `map-reduce`. Because the split or shuffle step occurs after the blockwise reduction, we _sometimes_ communicate a significantly smaller
amount of data than if we split or shuffled the input array.

```{image} /../diagrams/new-cohorts-annotated.svg
---
alt: cohorts-strategy-schematic
width: 100%
---
```

### Tradeoffs

1. Group labels must be known at graph construction time, so this only works for numpy arrays.
1. This does require more tasks and a more complicated graph, but the communication overhead can be significantly lower.
1. The detection of "cohorts" is currently slow but could be improved.
1. The extra effort of detecting cohorts and multiple copying of intermediate blocks may be worthwhile only if the chunk sizes are small
   relative to the approximate period of group labels, or small relative to the size of spatially localized groups.

### Example : sensitivity to chunking

One annoyance is that if the chunksize doesn't evenly divide the number of groups, we still end up splitting a number of chunks.
Consider our earlier example, `groupby("time.month")` with monthly frequency data and chunksize of 4 along `time`.
![cohorts-schematic](/../diagrams/cohorts-month-chunk4.png)

`flox` can find these cohorts, below it identifies the cohorts with labels `1,2,3,4`; `5,6,7,8`, and `9,10,11,12`.

```python
>>> flox.find_group_cohorts(labels, array.chunks[-1]).values()
[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]  # 3 cohorts
```

Now consider `chunksize=5`.
![cohorts-schematic](/../diagrams/cohorts-month-chunk5.png)

```python
>>> flox.core.find_group_cohorts(labels, array.chunks[-1]).values()
[[1], [2, 3], [4, 5], [6], [7, 8], [9, 10], [11], [12]]  # 8 cohorts
```

We find 8 cohorts (note the original xarray strategy is equivalent to constructing 12 cohorts).
In this case, it seems to better to rechunk to a size of `4` along `time`.
If you have ideas for improving this case, please open an issue.

### Example : spatial grouping
