from functools import cached_property

import dask
import numpy as np
import pandas as pd

import flox

from .helpers import codes_for_resampling


class Cohorts:
    """Time the core reduction function."""

    def setup(self, *args, **kwargs):
        raise NotImplementedError

    @cached_property
    def result(self):
        return flox.groupby_reduce(self.array, self.by, func="sum", axis=self.axis)[0]

    def containment(self):
        asfloat = self.bitmask().astype(float)
        chunks_per_label = asfloat.sum(axis=0)
        containment = (asfloat.T @ asfloat) / chunks_per_label
        return containment.todense()

    def chunks_cohorts(self):
        return flox.core.find_group_cohorts(
            self.by,
            [self.array.chunks[ax] for ax in self.axis],
            expected_groups=self.expected,
        )[1]

    def bitmask(self):
        chunks = [self.array.chunks[ax] for ax in self.axis]
        return flox.core._compute_label_chunk_bitmask(self.by, chunks, self.expected[-1] + 1)

    def time_find_group_cohorts(self):
        flox.core.find_group_cohorts(
            self.by,
            [self.array.chunks[ax] for ax in self.axis],
            expected_groups=self.expected,
        )
        # The cache clear fails dependably in CI
        # Not sure why
        try:
            flox.cache.cache.clear()
        except AttributeError:
            pass

    def track_num_cohorts(self):
        return len(self.chunks_cohorts())

    def time_graph_construct(self):
        flox.groupby_reduce(self.array, self.by, func="sum", axis=self.axis)

    def track_num_tasks(self):
        return len(self.result.dask.to_dict())

    def track_num_tasks_optimized(self):
        (opt,) = dask.optimize(self.result)
        return len(opt.dask.to_dict())

    def track_num_layers(self):
        return len(self.result.dask.layers)

    track_num_cohorts.unit = "cohorts"  # type: ignore[attr-defined] # Lazy
    track_num_tasks.unit = "tasks"  # type: ignore[attr-defined] # Lazy
    track_num_tasks_optimized.unit = "tasks"  # type: ignore[attr-defined] # Lazy
    track_num_layers.unit = "layers"  # type: ignore[attr-defined] # Lazy
    for f in [
        track_num_tasks,
        track_num_tasks_optimized,
        track_num_layers,
        track_num_cohorts,
    ]:
        f.repeat = 1  # type: ignore[attr-defined] # Lazy
        f.rounds = 1  # type: ignore[attr-defined] # Lazy
        f.number = 1  # type: ignore[attr-defined] # Lazy


class NWMMidwest(Cohorts):
    """2D labels, ireregular w.r.t chunk size.
    Mimics National Weather Model, Midwest county groupby."""

    def setup(self, *args, **kwargs):
        x = np.repeat(np.arange(30), 150)
        y = np.repeat(np.arange(30), 60)
        by = x[np.newaxis, :] * y[:, np.newaxis]

        self.by = flox.core._factorize_multiple((by,), expected_groups=(None,), any_by_dask=False)[0][0]

        self.array = dask.array.ones(self.by.shape, chunks=(350, 350))
        self.axis = (-2, -1)
        self.expected = pd.RangeIndex(self.by.max() + 1)


class ERA5Dataset:
    """ERA5"""

    def __init__(self, *args, **kwargs):
        self.time = pd.Series(pd.date_range("2016-01-01", "2018-12-31 23:59", freq="h"))
        self.axis = (-1,)
        self.array = dask.array.random.random((721, 1440, len(self.time)), chunks=(-1, -1, 48))

    def rechunk(self):
        self.array = flox.core.rechunk_for_cohorts(
            self.array,
            -1,
            self.by,
            force_new_chunk_at=[1],
            chunksize=48,
            ignore_old_chunks=True,
        )


class ERA5Resampling(Cohorts):
    def setup(self, *args, **kwargs):
        super().__init__()
        # nyears is number of years, adjust to make bigger,
        # full dataset is 60-ish years.
        nyears = 5
        shape = (37, 721, 1440, nyears * 365 * 24)
        chunks = (-1, -1, -1, 1)
        time = pd.date_range("2001-01-01", periods=shape[-1], freq="h")

        self.array = dask.array.random.random(shape, chunks=chunks)
        self.by = codes_for_resampling(time, "D")
        self.axis = (-1,)
        self.expected = np.unique(self.by)


class ERA5DayOfYear(ERA5Dataset, Cohorts):
    def setup(self, *args, **kwargs):
        super().__init__()
        self.by = self.time.dt.dayofyear.values - 1
        self.expected = pd.RangeIndex(self.by.max() + 1)


# class ERA5DayOfYearRechunked(ERA5DayOfYear, Cohorts):
#     def setup(self, *args, **kwargs):
#         super().setup()
#         self.array = dask.array.random.random((721, 1440, len(self.time)), chunks=(-1, -1, 24))
#         self.expected = pd.RangeIndex(self.by.max() + 1)


class ERA5MonthHour(ERA5Dataset, Cohorts):
    def setup(self, *args, **kwargs):
        super().__init__()
        by = (self.time.dt.month.values, self.time.dt.hour.values)
        ret = flox.core._factorize_multiple(
            by,
            (pd.Index(np.arange(1, 13)), pd.Index(np.arange(1, 25))),
            any_by_dask=False,
        )
        # Add one so the rechunk code is simpler and makes sense
        self.by = ret[0][0]
        self.expected = pd.RangeIndex(self.by.max() + 1)


class ERA5MonthHourRechunked(ERA5MonthHour, Cohorts):
    def setup(self, *args, **kwargs):
        super().setup()
        super().rechunk()


class PerfectMonthly(Cohorts):
    """Perfectly chunked for a "cohorts" monthly mean climatology"""

    def setup(self, *args, **kwargs):
        self.time = pd.Series(pd.date_range("1961-01-01", "2018-12-31 23:59", freq="ME"))
        self.axis = (-1,)
        self.array = dask.array.random.random((721, 1440, len(self.time)), chunks=(-1, -1, 4))
        self.by = self.time.dt.month.values - 1
        self.expected = pd.RangeIndex(self.by.max() + 1)

    def rechunk(self):
        self.array = flox.core.rechunk_for_cohorts(
            self.array,
            -1,
            self.by,
            force_new_chunk_at=[1],
            chunksize=4,
            ignore_old_chunks=True,
        )


# class PerfectMonthlyRechunked(PerfectMonthly):
#     def setup(self, *args, **kwargs):
#         super().setup()
#         super().rechunk()


class ERA5Google(Cohorts):
    def setup(self, *args, **kwargs):
        TIME = 900  # 92044 in Google ARCO ERA5
        self.time = pd.Series(pd.date_range("1959-01-01", freq="6h", periods=TIME))
        self.axis = (2,)
        self.array = dask.array.ones((721, 1440, TIME), chunks=(-1, -1, 1))
        self.by = self.time.dt.day.values - 1
        self.expected = pd.RangeIndex(self.by.max() + 1)


class PerfectBlockwiseResampling(Cohorts):
    """Perfectly chunked for blockwise resampling."""

    def setup(self, *args, **kwargs):
        index = pd.date_range("1959-01-01", freq="D", end="1962-12-31")
        self.time = pd.Series(index)
        TIME = len(self.time)
        self.axis = (2,)
        self.array = dask.array.ones((721, 1440, TIME), chunks=(-1, -1, 10))
        self.by = codes_for_resampling(index, freq="5D")
        self.expected = pd.RangeIndex(self.by.max() + 1)


class SingleChunk(Cohorts):
    """Single chunk along reduction axis: always blockwise."""

    def setup(self, *args, **kwargs):
        index = pd.date_range("1959-01-01", freq="D", end="1962-12-31")
        self.time = pd.Series(index)
        TIME = len(self.time)
        self.axis = (2,)
        self.array = dask.array.ones((721, 1440, TIME), chunks=(-1, -1, -1))
        self.by = codes_for_resampling(index, freq="5D")
        self.expected = pd.RangeIndex(self.by.max() + 1)


class OISST(Cohorts):
    def setup(self, *args, **kwargs):
        self.array = dask.array.ones((1, 14532), chunks=(1, 10))
        self.axis = (1,)
        index = pd.date_range("1981-09-01 12:00", "2021-06-14 12:00", freq="D")
        self.time = pd.Series(index)
        self.by = self.time.dt.dayofyear.values - 1
        self.expected = pd.RangeIndex(self.by.max() + 1)


class RandomBigArray(Cohorts):
    def setup(self, *args, **kwargs):
        M, N = 100_000, 20_000
        self.array = dask.array.random.normal(size=(M, N), chunks=(10_000, N // 5)).T
        self.by = np.random.choice(5_000, size=M)
        self.expected = pd.RangeIndex(5000)
        self.axis = (1,)
