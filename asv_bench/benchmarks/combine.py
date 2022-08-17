import numpy as np

import flox

from . import parameterized

N = 1000


class Combine:
    def setup(self, *args, **kwargs):
        raise NotImplementedError

    @parameterized("kind", ("cohorts", "mapreduce"))
    def time_combine(self, kind):
        flox.core._grouped_combine(
            getattr(self, f"x_chunk_{kind}"),
            **self.kwargs,
            keepdims=True,
            engine="numpy",
        )

    @parameterized("kind", ("cohorts", "mapreduce"))
    def peakmem_combine(self, kind):
        flox.core._grouped_combine(
            getattr(self, f"x_chunk_{kind}"),
            **self.kwargs,
            keepdims=True,
            engine="numpy",
        )


class Combine1d(Combine):
    """
    Time the combine step for dask reductions,
    this is for reducting along a single dimension
    """

    def setup(self, *args, **kwargs):
        def construct_member(groups):
            return {
                "groups": groups,
                "intermediates": [
                    np.ones((40, 120, 120, 4), dtype=float),
                    np.ones((40, 120, 120, 4), dtype=int),
                ],
            }

        # motivated by
        self.x_chunk_mapreduce = [
            construct_member(groups)
            for groups in [
                np.array((1, 2, 3, 4)),
                np.array((5, 6, 7, 8)),
                np.array((9, 10, 11, 12)),
            ]
            * 2
        ]

        self.x_chunk_cohorts = [construct_member(groups) for groups in [np.array((1, 2, 3, 4))] * 4]
        self.kwargs = {"agg": flox.aggregations.mean, "axis": (3,), "neg_axis": (-1,)}
