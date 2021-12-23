from functools import partial

try:
    import cachey
    import dask

    # 1MB cache
    cohorts_cache = cachey.Cache(1e6)
    memoize = partial(cohorts_cache.memoize, key=dask.base.tokenize)
except ImportError:
    memoize = lambda x: x
