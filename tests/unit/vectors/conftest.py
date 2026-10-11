"""Shared helpers for the vector tests."""

import unxt as u


def wrong_dimension(data, chart, dims, /):
    """Swap the first unit-carrying component for one of another dimension.

    Returns ``(key, data)``, or ``(None, None)`` if no component is checkable: a
    ``None`` dimension is unconstrained and a bare array has no unit.
    """
    for k, d in zip(chart.components, dims, strict=True):
        if d is not None and isinstance(data[k], u.AbstractQuantity):
            unit = "m" if d == u.dimension("time") else "s"
            return k, {**data, k: u.Q(1.0, unit)}
    return None, None
