"""Utility functions for charts."""

__all__ = ("guess_chart",)

from typing import Any, Final

import plum

import unxt as u

from .base import CHART_CLASSES, AbstractChart
from .euclidean import cart0d, cart1d, cart2d, cart3d
from coordinax._src import api
from coordinax._src.custom_types import CsDict

# ==============================================================
# Chart guessing utilities


@plum.dispatch
def guess_chart(obj: CsDict, /) -> AbstractChart:  # type: ignore[type-arg]
    """Infer a chart from the keys of a component dictionary.

    Examples
    --------
    >>> import coordinax as cx
    >>> d = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> chart = cx.charts.guess_chart(d)
    >>> chart
    Cart3D()

    """
    # Infer the representation from the keys
    obj_keys = set(obj.keys())
    chart = None
    for chart_cls in CHART_CLASSES:
        chart_instance = chart_cls()
        if set(chart_instance.components) == obj_keys:
            chart = chart_instance
            break

    if chart is None:
        msg = f"Cannot infer representation from keys {obj_keys}"
        raise ValueError(msg)

    return chart


SHAPE_CHART_MAP: Final[dict[int, AbstractChart[Any, Any]]] = {
    0: cart0d,
    1: cart1d,
    2: cart2d,
    3: cart3d,
}


@plum.dispatch
def guess_chart(obj: u.AbstractQuantity, /) -> AbstractChart:  # type: ignore[type-arg]
    """Infer a Cartesian chart from the shape of a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> q = u.Q([1.0, 2.0, 3.0], "m")
    >>> cx.charts.guess_chart(q)
    Cart3D()

    """
    try:
        chart = SHAPE_CHART_MAP[obj.shape[-1]]
    except KeyError as e:
        msg = (
            f"Cannot infer Cartesian chart from shape {obj.shape}. "
            "Supported shapes end with 0, 1, 2, or 3."
        )
        raise ValueError(msg) from e

    return chart


# ==============================================================
# CsDict construction utilities


@plum.dispatch
def cdict(obj: CsDict, /) -> CsDict:
    """Return a dictionary as-is.

    Examples
    --------
    >>> import coordinax as cx
    >>> d = {"x": 1.0, "y": 2.0}
    >>> cx.cdict(d)
    {'x': 1.0, 'y': 2.0}

    """
    return dict(obj)


@plum.dispatch
def cdict(obj: u.AbstractQuantity, /) -> CsDict:
    """Extract component dictionary from a Quantity.

    Treats the Quantity as a Cartesian vector with components in the last
    dimension. The appropriate Cartesian chart is determined from the last
    dimension of the quantity.

    Raises
    ------
    ValueError
        If the last dimension of the quantity doesn't match a known Cartesian
        chart (0D, 1D, 2D, or 3D).

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u
    >>> q = u.Q([1.0, 2.0, 3.0], "m")
    >>> d = cx.cdict(q)
    >>> d
    {'x': Quantity(Array(1., dtype=float64), unit='m'),
     'y': Quantity(Array(2., dtype=float64), unit='m'),
     'z': Quantity(Array(3., dtype=float64), unit='m')}

    """
    chart = guess_chart(obj)
    return api.cdict(obj, chart)


@plum.dispatch
def cdict(obj: u.AbstractQuantity, chart: AbstractChart, /) -> CsDict:  # type: ignore[type-arg]
    """Extract component dictionary from a Quantity.

    Treats the Quantity as a Cartesian vector with components in the last
    dimension. The appropriate Cartesian chart is determined from the last
    dimension of the quantity.

    Raises
    ------
    ValueError
        If the last dimension of the quantity doesn't match a known Cartesian
        chart (0D, 1D, 2D, or 3D).

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u
    >>> q = u.Q([1.0, 2.0, 3.0], "m")
    >>> d = cx.cdict(q, cx.charts.cart3d)
    >>> d
    {'x': Quantity(Array(1., dtype=float64), unit='m'),
     'y': Quantity(Array(2., dtype=float64), unit='m'),
     'z': Quantity(Array(3., dtype=float64), unit='m')}

    """
    # TODO: some check the chart is Cartesian?
    if obj.shape[-1] != len(chart.components):
        msg = (
            f"Quantity last dimension {obj.shape[-1]} does not match "
            f"chart with {len(chart.components)} components."
        )
        raise ValueError(msg)
    return {k: obj[..., i] for i, k in enumerate(chart.components)}
