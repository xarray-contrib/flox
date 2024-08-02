# Snapshot tests for cohorts detection

import pytest

from asv_bench.benchmarks import cohorts

from . import requires_dask


@requires_dask
@pytest.mark.parametrize(
    "testcase",
    [
        cohorts.ERA5DayOfYear,
        cohorts.ERA5Google,
        cohorts.ERA5MonthHour,
        cohorts.ERA5MonthHourRechunked,
        cohorts.OISST,
        cohorts.PerfectBlockwiseResampling,
        cohorts.PerfectMonthly,
        cohorts.RandomBigArray,
        cohorts.SingleChunk,
        cohorts.NWMMidwest,
    ],
)
def test_snapshot_cohorts(testcase, snapshot):
    problem = testcase()
    problem.setup()
    chunks_cohorts = problem.chunks_cohorts()
    assert chunks_cohorts == snapshot
