"""Hypothesis strategies for coordinax representations."""

__all__ = ("can_point_transform",)


from typing import Any

import coordinax.charts as cxc


def can_point_transform(
    to_chart: cxc.AbstractChart[Any, Any], from_chart: cxc.AbstractChart[Any, Any], /
) -> bool:
    """Return True if ``point_transform`` can convert between the two reps."""
    if type(to_chart) is type(from_chart):
        return True
    try:
        _ = to_chart.cartesian
        _ = from_chart.cartesian
    except (NotImplementedError, ValueError):
        return False
    return True
