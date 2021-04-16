def _atleast_1d(inp):
    if isinstance(inp, str):
        inp = (inp,)
    return inp


class Aggregation:
    def __init__(
        self, name, chunk, combine, aggregate=None, finalize=None, fill_value=None, dtype=None
    ):
        self.name = name
        # initialize blockwise reduction
        self.chunk = _atleast_1d(chunk)
        # how to aggregate results after first round of reduction
        self.combine = _atleast_1d(combine)
        # final aggregation
        self.aggregate = aggregate if aggregate else combine
        # finalize results (see mean)
        self.finalize = finalize if finalize else lambda x: x[self.chunk[0]]
        # fill_value is used to reindex to expected_groups.
        # They should make sense when aggregated together with results from other blocks
        self.fill_value = fill_value
        self.dtype = dtype


count = Aggregation("count", chunk="count", combine="sum", fill_value=0, dtype=int)
sum = Aggregation("sum", chunk="sum", combine="sum", fill_value=0)
nansum = Aggregation("nansum", chunk="nansum", combine="sum", fill_value=0)
prod = Aggregation("prod", chunk="prod", combine="prod", fill_value=1)
nanprod = Aggregation("nanprod", chunk="nanprod", combine="prod", fill_value=1)
mean = Aggregation(
    "mean",
    chunk=("sum", "count"),
    combine=("sum", "sum"),
    finalize=lambda x: x["sum"] / x["count"],
    fill_value=0,
)
nanmean = Aggregation(
    "nanmean",
    chunk=("nansum", "count"),
    combine=("sum", "sum"),
    finalize=lambda x: x["nansum"] / x["count"],
    fill_value=0,
)
