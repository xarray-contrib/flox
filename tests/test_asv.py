# Run asv benchmarks as tests

import pytest

pytest.importorskip("dask")

from asv_bench.benchmarks import reduce


@pytest.mark.parametrize("problem", [reduce.ChunkReduce1D, reduce.ChunkReduce2D, reduce.ChunkReduce2DAllAxes])
def test_reduce(problem) -> None:
    testcase = problem()
    testcase.setup()
    for args in zip(*testcase.time_reduce.params):
        testcase.time_reduce(*args)


def test_reduce_bare() -> None:
    testcase = reduce.ChunkReduce1D()
    testcase.setup()
    for args in zip(*testcase.time_reduce_bare.params):
        testcase.time_reduce_bare(*args)
