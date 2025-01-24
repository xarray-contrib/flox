import numpy as np
import pandas as pd


def codes_for_resampling(group_as_index: pd.Index, freq: str) -> np.ndarray:
    s = pd.Series(np.arange(group_as_index.size), group_as_index)
    grouped = s.groupby(pd.Grouper(freq=freq))
    first_items = grouped.first()
    counts = grouped.count()
    codes = np.repeat(np.arange(len(first_items)), counts)
    return codes
