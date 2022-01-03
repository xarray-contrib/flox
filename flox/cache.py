from functools import partial

try:
    import cachey
    import dask

    # 1MB cache
    cache = cachey.Cache(1e6)
    memoize = partial(cache.memoize, key=dask.base.tokenize)
except ImportError:
    memoize = lambda x: x
