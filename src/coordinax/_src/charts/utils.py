"""Utility functions for charts."""

__all__ = ("guess_chart",)

import functools as ft

from jaxtyping import Array, ArrayLike, Shaped

import plum

import quaxed.numpy as jnp
import unxt as u

from .base import NON_ABC_CHART_CLASSES, AbstractChart
from .euclidean import cart1d, cart2d, cart3d
from coordinax._src import api
from coordinax._src.custom_types import CsDict

# ==============================================================
# Chart guessing utilities


@plum.dispatch
@ft.cache
def guess_chart(obj: frozenset[str], /) -> AbstractChart:  # type: ignore[type-arg]
    """Infer a chart from the keys of a component dictionary.

    Note that many charts may share the same component names (e.g., Spherical3D
    and MathSpherical3D both use 'r', 'theta', 'phi'). These are completely
    indistinguishable from component names alone, so this function will return
    the first matching chart it finds. Since the function is cached, the result
    will be consistent across calls.

    Examples
    --------
    >>> import coordinax as cx
    >>> d = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> chart = cx.charts.guess_chart(d)
    >>> chart
    Cart3D()

    """
    # Infer the representation from the keys
    chart = None
    for chart_cls in NON_ABC_CHART_CLASSES:
        try:
            chart_instance = chart_cls()
        except TypeError:  # TODO: more efficient way to generate components list
            continue
        if frozenset(chart_instance.components) == obj:
            chart = chart_instance
            break

    if chart is None:
        msg = f"Cannot infer representation from keys {obj}"
        raise ValueError(msg)

    return chart


@plum.dispatch
def guess_chart(obj: CsDict, /) -> AbstractChart:  # type: ignore[type-arg]
    """Infer a chart from the keys of a component dictionary.

    Note that many charts may share the same component names (e.g., Spherical3D
    and MathSpherical3D both use 'r', 'theta', 'phi'). These are completely
    indistinguishable from component names alone, so this function will return
    the first matching chart it finds. Since the function is cached, the result
    will be consistent across calls.

    Examples
    --------
    >>> import coordinax as cx
    >>> d = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> chart = cx.charts.guess_chart(d)
    >>> chart
    Cart3D()

    """
    return api.guess_chart(frozenset(obj.keys()))


@plum.dispatch
def guess_chart(
    obj: Shaped[Array | u.AbstractQuantity, "*batch 1"], /
) -> AbstractChart:  # type: ignore[type-arg]
    """Infer a 1D Cartesian chart from last dimension of a value / quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> q = u.Q([1.0], "m")
    >>> cx.charts.guess_chart(q)
    Cart1D()

    """
    return cart1d


@plum.dispatch
def guess_chart(
    obj: Shaped[Array | u.AbstractQuantity, "*batch 2"], /
) -> AbstractChart:  # type: ignore[type-arg]
    """Infer a 2D Cartesian chart from last dimension of a value / quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> q = u.Q([1.0, 2.0], "m")
    >>> cx.charts.guess_chart(q)
    Cart2D()

    """
    return cart2d


@plum.dispatch
def guess_chart(
    obj: Shaped[Array | u.AbstractQuantity, "*batch 3"], /
) -> AbstractChart:  # type: ignore[type-arg]
    """Infer a 3D Cartesian chart from last dimension of a value / quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> q = u.Q([1.0, 2.0, 3.0], "m")
    >>> cx.charts.guess_chart(q)
    Cart3D()

    """
    return cart3d


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
    return obj


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

    Treats the Quantity as a vector with components in the last dimension,
    splitting along that axis according to the chart's component names.

    This function requires that:
    1. The last dimension of the quantity matches the number of chart components
    2. The chart has homogeneous coordinate dimensions (all components have the
       same physical dimension, like Cartesian charts)

    Raises
    ------
    ValueError
        If the last dimension of the quantity doesn't match the chart's
        component count, or if dimensions don't match.

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
    if obj.shape[-1] != len(chart.components):
        msg = (
            f"Quantity last dimension {obj.shape[-1]} does not match "
            f"chart with {len(chart.components)} components."
        )
        raise ValueError(msg)

    return {k: obj[..., i] for i, k in enumerate(chart.components)}


@plum.dispatch
def cdict(obj: ArrayLike, chart: AbstractChart, /) -> CsDict:  # type: ignore[type-arg]
    """Extract component dictionary from an array.

    Treats the array as a Cartesian vector with components in the last
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
    >>> import jax.numpy as jnp
    >>> arr = jnp.array([1.0, 2.0, 3.0])
    >>> d = cx.cdict(arr, cx.charts.cart3d)
    >>> d
    {'x': Array(1., dtype=float64), 'y': Array(2., dtype=float64),
     'z': Array(3., dtype=float64)}

    """
    obj = jnp.asarray(obj)
    if obj.shape[-1] != len(chart.components):
        msg = (
            f"Array last dimension {obj.shape[-1]} does not match "
            f"chart with {len(chart.components)} components."
        )
        raise ValueError(msg)
    return {k: obj[..., i] for i, k in enumerate(chart.components)}
