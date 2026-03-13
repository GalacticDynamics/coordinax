"""Utility functions for charts."""

__all__: tuple[str, ...] = ()

import functools as ft

from jaxtyping import Array, Shaped

import plum

import unxt as u

import coordinax.api.charts as cxcapi
from .base import NON_ABC_CHART_CLASSES, AbstractChart
from .custom_types import CDict
from .d1 import cart1d
from .d2 import cart2d
from .d3 import cart3d
from .dn import cartnd


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
    >>> import coordinax.charts as cxc
    >>> d = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> chart = cxc.guess_chart(d)
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
def guess_chart(obj: CDict, /) -> AbstractChart:  # type: ignore[type-arg]
    """Infer a chart from the keys of a component dictionary.

    Note that many charts may share the same component names (e.g., Spherical3D
    and MathSpherical3D both use 'r', 'theta', 'phi'). These are completely
    indistinguishable from component names alone, so this function will return
    the first matching chart it finds. Since the function is cached, the result
    will be consistent across calls.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> d = {"x": 1.0, "y": 2.0, "z": 3.0}
    >>> chart = cxc.guess_chart(d)
    >>> chart
    Cart3D()

    """
    return cxcapi.guess_chart(frozenset(obj.keys()))


@plum.dispatch
def guess_chart(
    _: Shaped[Array, "*batch 1"] | Shaped[u.AbstractQuantity, "*batch 1"],
    /,
) -> AbstractChart:  # type: ignore[type-arg]
    """Infer a 1D Cartesian chart from last dimension of a value / quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cx
    >>> q = u.Q([1.0], "m")
    >>> cxc.guess_chart(q)
    Cart1D()

    """
    return cart1d


@plum.dispatch
def guess_chart(
    _: Shaped[Array, "*batch 2"] | Shaped[u.AbstractQuantity, "*batch 2"], /
) -> AbstractChart:  # type: ignore[type-arg]
    """Infer a 2D Cartesian chart from last dimension of a value / quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cx
    >>> q = u.Q([1.0, 2.0], "m")
    >>> cxc.guess_chart(q)
    Cart2D()

    """
    return cart2d


@plum.dispatch
def guess_chart(
    _: Shaped[Array, "*batch 3"] | Shaped[u.AbstractQuantity, "*batch 3"], /
) -> AbstractChart:  # type: ignore[type-arg]
    """Infer a 3D Cartesian chart from last dimension of a value / quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cx
    >>> q = u.Q([1.0, 2.0, 3.0], "m")
    >>> cxc.guess_chart(q)
    Cart3D()

    """
    return cart3d


@plum.dispatch(precedence=-1)
def guess_chart(
    _: Shaped[Array, "*batch N"] | Shaped[u.AbstractQuantity, "*batch N"], /
) -> AbstractChart:  # type: ignore[type-arg]
    """Infer a N-dimensional Cartesian chart from last dimension of a value / quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cx
    >>> q = u.Q([1.0, 2.0, 3.0, 4.0], "m")
    >>> cxc.guess_chart(q)
    CartND()

    """
    return cartnd
