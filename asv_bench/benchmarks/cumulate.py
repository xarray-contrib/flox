import numpy as np
import numpy_groupies as npg
import pandas as pd

import flox

from . import parameterized

N = 1000
funcs = ["cumsum"]
engines = ["numpy"]
expected_groups = [None, pd.IntervalIndex.from_breaks([1, 2, 4])]


class ChunkCumulate:
    """Time the core reduction function."""

    def setup(self, *args, **kwargs):
        # pre-compile jitted funcs
        if "numba" in engines:
            for func in funcs:
                npg.aggregate_numba.aggregate(
                    np.ones((100,), dtype=int), np.ones((100,), dtype=int), func=func
                )
        raise NotImplementedError

    @parameterized("func, engine, expected_groups", [funcs, engines, expected_groups])
    def time_cumulate(self, func, engine, expected_groups):
        flox.groupby_reduce(
            self.array,
            self.labels,
            func=func,
            engine=engine,
            axis=self.axis,
            expected_groups=expected_groups,
        )

    @parameterized("func, engine, expected_groups", [funcs, engines, expected_groups])
    def peakmem_cumulate(self, func, engine, expected_groups):
        flox.groupby_reduce(
            self.array,
            self.labels,
            func=func,
            engine=engine,
            axis=self.axis,
            expected_groups=expected_groups,
        )


class ChunkCumulate1D(ChunkCumulate):
    def setup(self, *args, **kwargs):
        self.array = np.ones((N,))
        self.labels = np.repeat(np.arange(5), repeats=N // 5)
        self.axis = -1


class ChunkCumulate2D(ChunkCumulate):
    def setup(self, *args, **kwargs):
        self.array = np.ones((N, N))
        self.labels = np.repeat(np.arange(N // 5), repeats=5)
        self.axis = -1


class ChunkCumulate2DAllAxes(ChunkCumulate):
    def setup(self, *args, **kwargs):
        self.array = np.ones((N, N))
        self.labels = np.repeat(np.arange(N // 5), repeats=5)
        self.axis = None
