# Algorithms

`flox` outsources the core GroupBy operation to the vectorized implementations in
[numpy_groupies](https://github.com/ml31415/numpy-groupies). Constructing
an efficient groupby reduction with dask is hard, and depends on how the
groups are distributed amongst the blocks of an array. `flox` implements 4 strategies for
grouped reductions, each is appropriate for a particular distribution of groups
among the blocks of a dask array.

Switch between the various strategies by passing `method` to either {py:func}`flox.core.groupby_reduce`
or `xarray_reduce`.


First we describe xarray's current strategy

## `method="split-reduce"`: Xarray's current GroupBy strategy

Xarray's current strategy is to find all unique group labels, index out each group,
and then apply the reduction operation. Note that this only works if we know the group
labels (i.e. you cannot use this strategy to group by a dask array).

Schematically, this looks like (colors indicate group labels; separated groups of colors
indicate different blocks of an array):
![xarray-current-strategy](../diagrams/split-reduce.png)

The first step is to extract all members of a group, which involves a *lot* of
communication and is quite expensive (in dataframe terminology, this is a "shuffle").
This is fundamentally why many groupby reductions don't work well right now with
big datasets.

## `method="map-reduce"`

The first idea is to use the "map-reduce" strategy (inspired by `dask.dataframe`).

![map-reduce-strategy-schematic](/../diagrams/map-reduce.png)

The GroupBy reduction is first applied blockwise. Those intermediate results are
combined by concatenating to form a new array which is then reduced
again. The combining of intermediate results uses dask\'s `_tree_reduce`
till all group results are in one block. At that point the result is
\"finalized\" and returned to the user.

*Tradeoffs*:
1. Allows grouping by a dask array so group labels need not be known at graph construction
   time.
1. Works well when either the initial blockwise reduction is effective, or if the
   reduction at the first combine step is effective. "effective" means we actually
   reduce values and release some memory.

## `method="blockwise"`

One case where `"map-reduce"` doesn't work well is the case of "resampling" reductions. An
example here is resampling from daily frequency to monthly frequency data:  `da.resample(time="M").mean()`
For resampling type reductions,
1. Group members occur sequentially (all days in January 2001 occur one after the other)
2. All groups are roughly equal length (31 days in January but 28 in most Februaries)
3. All members in a group are next to each other (if the time series is sorted, which it
   usually is).

In this case, it makes sense to use `dask.dataframe` resample strategy which is to rechunk
so that all members of a group are in a single block. Then, the groupby operation can be applied blockwise.

![blockwise-strategy-schematic](/../diagrams/blockwise.png)

*Tradeoffs*
1. Only works for certain groupings.
1. Group labels must be known at graph construction time, so this only works for numpy arrays
1. Currently the rechunking is only implemented for 1D arrays (being motivated by time resampling),
   but a nD generalization seems possible.
1. Works better when multiple groups are already in a single block; so that the intial
   rechunking only involves a small amount of communication.

## `method="cohorts"`

We can combine all of the above ideas for cases where members from different groups tend to occur close to each other.
One example is the construction of "climatologies" which is a climate science term for something like `groupby("time.month")`
("monthly climatology") or `groupby("time.dayofyear")` ("daily climatology"). In these cases,
1. Groups occur sequentially (day 2 is always after day 1; and February is always after January)
2. Groups are approximately periodic (some years have 365 days and others have 366)

The idea here is to copy xarray's subsetting strategy but instead index out "cohorts" or group labels
that tend to occur next to each other.

Consider this example of monthly average data; where 4 months are present in a single block (i.e. chunksize=4)
![cohorts-schematic](/../diagrams/cohorts-month-chunk4.png)

Because a chunksize of 4 evenly divides the number of groups (12) all we need to do is index out blocks
0, 3, 7 and then apply the `"map-reduce"` strategy to form the final result for months Jan-Apr. Repeat for the
remaining groups of months (May-Aug; Sep-Dec) and then concatenate.

`flox` can find these cohorts, below it identifies the cohorts with labels `1,2,3,4`; `5,6,7,8`, and `9,10,11,12`.
``` python
>>> flox.core.find_group_cohorts(labels, array.chunks[-1]))
[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]  # 3 cohorts
```
For each cohort, it counts the number of blocks that need to be reduced. If `1` then it applies the reduction blockwise.
If > 1; then it uses `"map-reduce"`.

One annoyance is that if the chunksize doesn't evenly divide the number of groups, we still end up splitting a number of chunks.
For example, when `chunksize=5`
![cohorts-schematic](/../diagrams/cohorts-month-chunk5.png)

``` python
>>> flox.core.find_group_cohorts(labels, array.chunks[-1]))
[[1], [2, 3], [4, 5], [6], [7, 8], [9, 10], [11], [12]]  # 8 cohorts
```
We find 8 cohorts (note the original xarray strategy is equivalent to constructing 12 cohorts).

It's possible that some initial rechunking makes the situation better (just rechunk from 5-4), but it isn't an obvious improvement.
If you have ideas for improving this case, please open an issue.

*Tradeoffs*
1. Generalizes well; when there's exactly one groups per chunk, this replicates Xarray's
   strategy which is optimal. For resampling type reductions, as long as the array
   is chunked appropriately ({py:func}`flox.core.rechunk_for_blockwise`, {py:func}`flox.xarray.rechunk_for_blockwise`), `method="cohorts"` is equivalent to `method="blockwise"`!
1. Group labels must be known at graph construction time, so this only works for numpy arrays
1. Currenltly implemented for grouping by 1D arrays. An nD generalization seems possible,
   but hard?
