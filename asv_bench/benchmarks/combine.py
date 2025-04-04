from functools import partial
from typing import Any

import numpy as np

import flox

from . import parameterized

N = 1000


def _get_combine(combine):
    if combine == "grouped":
        return partial(flox.core._grouped_combine, engine="numpy")
    else:
        try:
            reindex = flox.ReindexStrategy(blockwise=False)
        except AttributeError:
            reindex = False
        return partial(flox.core._simple_combine, reindex=reindex)


class Combine:
    def setup(self, *args, **kwargs):
        raise NotImplementedError

    @parameterized(("kind", "combine"), (("reindexed", "not_reindexed"), ("grouped", "simple")))
    def time_combine(self, kind, combine):
        _get_combine(combine)(
            getattr(self, f"x_chunk_{kind}"),
            **self.kwargs,
            keepdims=True,
        )

    @parameterized(("kind", "combine"), (("reindexed", "not_reindexed"), ("grouped", "simple")))
    def peakmem_combine(self, kind, combine):
        _get_combine(combine)(
            getattr(self, f"x_chunk_{kind}"),
            **self.kwargs,
            keepdims=True,
        )


class Combine1d(Combine):
    """
    Time the combine step for dask reductions,
    this is for reducting along a single dimension
    """

    def setup(self, *args, **kwargs) -> None:
        def construct_member(groups) -> dict[str, Any]:
            return {
                "groups": groups,
                "intermediates": [
                    np.ones((40, 120, 120, 4), dtype=float),
                    np.ones((40, 120, 120, 4), dtype=int),
                ],
            }

        # motivated by
        self.x_chunk_not_reindexed = [
            construct_member(groups)
            for groups in [
                np.array((1, 2, 3, 4)),
                np.array((5, 6, 7, 8)),
                np.array((9, 10, 11, 12)),
            ]
            * 2
        ]

        self.x_chunk_reindexed = [construct_member(groups) for groups in [np.array((1, 2, 3, 4))] * 4]
        self.kwargs = {
            "agg": flox.aggregations._initialize_aggregation("sum", "float64", np.float64, 0, 0, {}),
            "axis": (3,),
        }
