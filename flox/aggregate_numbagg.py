from numbagg.grouped import group_nanmean, group_nansum


def nansum_of_squares(
    group_idx, array, *, axis=-1, func="sum", size=None, fill_value=None, dtype=None
):

    return group_nansum(
        array**2,
        group_idx,
        axis=axis,
        num_labels=size,
        # fill_value=fill_value,
        # dtype=dtype,
    )


def nansum(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    # npg takes out NaNs before calling np.bincount
    # This means that all NaN groups are equivalent to absent groups
    # This behaviour does not work for xarray
    return group_nansum(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        # fill_value=fill_value,
        # dtype=dtype,
    )


def nanmean(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
    # npg takes out NaNs before calling np.bincount
    # This means that all NaN groups are equivalent to absent groups
    # This behaviour does not work for xarray

    return group_nanmean(
        array,
        group_idx,
        axis=axis,
        num_labels=size,
        # fill_value=fill_value,
        # dtype=dtype,
    )


sum = nansum
mean = nanmean
sum_of_squares = nansum_of_squares

# def nanprod(group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None):
#     # npg takes out NaNs before calling np.bincount
#     # This means that all NaN groups are equivalent to absent groups
#     # This behaviour does not work for xarray

#     return npg.aggregate_numpy.aggregate(
#         group_idx,
#         np.where(np.isnan(array), 1, array),
#         axis=axis,
#         func="prod",
#         size=size,
#         fill_value=fill_value,
#         dtype=dtype,
#     )
