"""Hypothesis strategies for coordinax charts."""

import warnings

__all__ = (
    "find_chart_strategy",
    "register_chart_strategy",
    "any_charts",
    "CHART_STRATEGIES",
)


import functools as ft

from collections.abc import Callable
from typing import Any, Final, TypeVar

import hypothesis.strategies as st
import plum
from hypothesis import assume

import coordinax.charts as cxc
from .chart_kwargs import chart_init_kwargs

T = TypeVar("T")


CHART_STRATEGIES: Final[
    dict[type[cxc.AbstractChart], Callable[..., st.SearchStrategy]]
] = {}


@ft.cache
def find_chart_strategy(
    chart_cls: type[cxc.AbstractChart], /
) -> Callable[..., st.SearchStrategy]:
    """Determine the strategy to use for a given chart class.

    Parameters
    ----------
    chart_cls
        The chart class to find a strategy for.

    Returns
    -------
    Callable[..., st.SearchStrategy]
        A function that returns a Hypothesis strategy for the given chart class.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> from coordinax.hypothesis.charts import find_chart_strategy

    >>> find_chart_strategy(cxc.Cart3D).__name__
    'any_charts'

    >>> find_chart_strategy(cxc.CartesianProductChart).__name__
    'cartesian_product_charts'

    >>> find_chart_strategy(cxc.SpaceTimeCT).__name__
    'spacetimect_charts'

    """
    # Ignore the search-strategy return type
    with warnings.catch_warnings(action="ignore", category=UserWarning):
        return _find_chart_strategy_impl.resolve_method((chart_cls,))[0]


@plum.dispatch.abstract
def _find_chart_strategy_impl(
    chart_cls: type[cxc.AbstractChart], /
) -> Callable[..., st.SearchStrategy]:
    del chart_cls
    raise NotImplementedError  # pragma: no-cover


def register_chart_strategy(
    chart_cls: type[cxc.AbstractChart[Any, Any]], /
) -> Callable[[T], T]:
    """Return decorator to register a strategy for a specific chart class.

    Parameters
    ----------
    chart_cls
        The chart class to register a strategy for.

    Returns
    -------
    Callable
        A decorator function that registers a strategy for the given chart class.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> from coordinax.hypothesis.charts import register_chart_strategy

    >>> @register_chart_strategy(cxc.Cart3D)  # doctest: +SKIP
    ... @st.composite
    ... def cart3d_strategy(draw):
    ...     # Strategy implementation for Cart3D
    ...     pass

    """

    def inner(func: T, /) -> T:
        """Register a strategy for a specific chart class."""
        # Register the strategy for the chart class
        _find_chart_strategy_impl.dispatch_multi((type[chart_cls],))(func)
        # Return the original function unmodified
        return func

    return inner


@register_chart_strategy(cxc.AbstractChart)
@st.composite
def any_charts(
    draw: st.DrawFn, chart_cls: type[cxc.AbstractChart], /, ndim: int | None = None
) -> cxc.AbstractChart:
    """Strategy to draw any chart instance of a given class.

    Parameters
    ----------
    draw : `hypothesis.strategies.DrawFn`
        Draw function.
    chart_cls : type[`coordinax.charts.AbstractChart`]
        The chart class to draw.
    ndim : int | None
        Target dimensionality. If specified, only charts with this dimensionality
        will be drawn.

    Returns
    -------
    `coordinax.charts.AbstractChart`
        An instance of the specified chart class.

    """
    # Build and draw kwargs for required parameters
    kwargs = draw(chart_init_kwargs(chart_cls, ndim=ndim))

    # Create the instance
    chart = chart_cls(**kwargs)

    # Filter by ndim if specified
    if ndim is not None:
        assume(chart.ndim == ndim)

    return chart
