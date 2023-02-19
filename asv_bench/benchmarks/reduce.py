import numpy as np
import numpy_groupies as npg
import pandas as pd

import flox

from . import parameterized

N = 1000
funcs = ["sum", "nansum", "mean", "nanmean", "max", "var", "nanvar", "count"]
engines = ["flox", "numpy"]
expected_groups = [None, pd.IntervalIndex.from_breaks([1, 2, 4])]


class ChunkReduce:
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
    def time_reduce(self, func, engine, expected_groups):
        flox.groupby_reduce(
            self.array,
            self.labels,
            func=func,
            engine=engine,
            axis=self.axis,
            expected_groups=expected_groups,
        )

    @parameterized("func, engine, expected_groups", [funcs, engines, expected_groups])
    def peakmem_reduce(self, func, engine, expected_groups):
        flox.groupby_reduce(
            self.array,
            self.labels,
            func=func,
            engine=engine,
            axis=self.axis,
            expected_groups=expected_groups,
        )


class ChunkReduce1D(ChunkReduce):
    def setup(self, *args, **kwargs):
        self.array = np.ones((N,))
        self.labels = np.repeat(np.arange(5), repeats=N // 5)
        self.axis = -1


class ChunkReduce2D(ChunkReduce):
    def setup(self, *args, **kwargs):
        self.array = np.ones((N, N))
        self.labels = np.repeat(np.arange(N // 5), repeats=5)
        self.axis = -1


class ChunkReduce2DAllAxes(ChunkReduce):
    def setup(self, *args, **kwargs):
        self.array = np.ones((N, N))
        self.labels = np.repeat(np.arange(N // 5), repeats=5)
        self.axis = None
