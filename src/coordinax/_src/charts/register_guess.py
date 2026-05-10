"""Utility functions for charts."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, Shaped
from typing import Any, cast

import plum

import unxt as u

import coordinax.api.charts as cxcapi
from .d1 import cart1d
from .d2 import cart2d
from .d3 import cart3d
from .dn import cartnd
from coordinax._src.base_charts import (
    NON_ABC_CHART_CLASSES,
    AbstractChart,
    AbstractFixedComponentsChart,
)
from coordinax._src.custom_types import CDict


# TODO: speed this up. The problem is that caching the results breaks something,
# causing functions in other modules to fail type(x) is type(y) checks.
def guess_chart_cls(obj: frozenset[str]) -> type[AbstractChart[Any, Any]]:
    """Infer a chart class from the keys of a component dictionary.

    This only works on charts with fixed components.

    """
    for chart_cls in NON_ABC_CHART_CLASSES:
        if (
            issubclass(chart_cls, AbstractFixedComponentsChart)
            and frozenset(chart_cls._components) == obj
        ):
            return chart_cls

    msg = f"Cannot infer representation from keys {obj}"
    raise ValueError(msg)


@plum.dispatch
def guess_chart(obj: frozenset[str], /) -> AbstractChart:
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
    chart_cls = guess_chart_cls(obj)

    # Instantiate the chart class
    chart = chart_cls()

    # This should never happen since guess_chart_cls should raise if no chart is
    # found, but we check just in case.
    if chart is None:
        msg = f"Cannot infer representation from keys {obj}"
        raise ValueError(msg)

    return chart


@plum.dispatch
def guess_chart(obj: CDict, /) -> AbstractChart:
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
    out = cxcapi.guess_chart(frozenset(obj.keys()))
    return cast("AbstractChart", out)


@plum.dispatch
def guess_chart(
    _: Shaped[Array, "*batch 1"] | Shaped[u.AbstractQuantity, "*batch 1"],
    /,
) -> AbstractChart:
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
) -> AbstractChart:
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
) -> AbstractChart:
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


@plum.dispatch(precedence=-1)  # ty: ignore[no-matching-overload]
def guess_chart(
    _: Shaped[Array, "*batch N"] | Shaped[u.AbstractQuantity, "*batch N"], /
) -> AbstractChart:
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
