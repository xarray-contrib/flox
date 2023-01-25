def _strip_units(*arrays):
    try:
        import pint

        pint_quantity = pint.Quantity

    except ImportError:
        pint_quantity = None

    bare = tuple(array.magnitude if isinstance(array, pint_quantity) else array for array in arrays)
    units = [array.units if isinstance(array, pint_quantity) else None for array in arrays]

    return bare[0], bare[1:], units


def _reattach_units(*arrays, units):
    try:
        import pint

        return tuple(
            pint.Quantity(array, unit) if unit is not None else array
            for array, unit in zip(arrays, units)
        )
    except ImportError:
        return arrays
