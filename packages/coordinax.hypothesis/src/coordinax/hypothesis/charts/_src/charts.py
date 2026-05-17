"""Hypothesis strategies for coordinax charts."""

__all__ = ("charts",)


import inspect
from typing import Any

import hypothesis.strategies as st
import plum
from hypothesis import assume

import coordinax.charts as cxc

from .chart_kwargs import chart_init_kwargs
from .classes import chart_classes
from coordinax.hypothesis.utils import (
    annotations,
    draw_if_strategy,
    strip_return_annotation,
)


@plum.dispatch.abstract
def charts(
    draw: st.DrawFn,
    chart_cls: Any,
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | None | st.SearchStrategy = None,
) -> Any:
    """Strategy to determine and draw chart instances.

    Parameters
    ----------
    draw
        The draw function used by the hypothesis composite strategy.
        Automatically provided by hypothesis.
    chart_cls
        The chart class to draw an instance of. If provided, the strategy will
        draw an instance of this class. If None, the strategy will draw an
        instance of any chart class that satisfies the filter and exclude
        criteria.
    filter
        A class or tuple of classes to limit the charts to, by default `()` (no
        additional filter). Can be a single type or a tuple of types.

        For example:

        - `coordinax.charts.Abstract0D` to limit to 0D charts.
        - `coordinax.charts.Abstract1D` to limit to 1D charts.
        - `coordinax.charts.Abstract2D` to limit to 2D charts.
        - `coordinax.charts.Abstract3D` to limit to 3D charts.

        In combination, this can be used to draw charts that satisfy multiple
        criteria, e.g., ``filter=(coordinax.charts.Abstract3D,
        coordinax.charts.AbstractSpherical3D)``.  Just note that some charts
        are not subclasses of `coordinax.charts.Abstract3D` (e.g. `MinkowskiCT`)
        and will be excluded unless explicitly added.
    exclude
        Specific classes to exclude, by default ().
    ndim
        `chart.ndim` constraint. Can be: - `None`: No constraint - An integer:
        Exact ndim match - A strategy: Draw ndim from strategy (e.g.,
          `st.integers(min_value=1, max_value=2)`)

    Returns
    -------
    AbstractChart
        An instance of a chart class.

    Raises
    ------
    NotImplementedError
        If no strategy is registered for the drawn chart class.
    ValueError
        If a `chart_cls` is provided and `filter` or `exclude` aren't empty.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.hypothesis.main as cxst
    >>> import hypothesis.strategies as st

    >>> # Draw any chart instance (dimensionality > 0 by default)
    >>> chart_strategy = cxst.charts()

    >>> # Draw charts with exact ndim
    >>> exact_2d_strategy = cxst.charts(ndim=2)

    >>> # Include 0-dimensional charts
    >>> all_dim_strategy = cxst.charts(ndim=None, exclude=())

    >>> # Use a strategy to draw ndim
    >>> strategy_dim = cxst.charts(ndim=st.integers(min_value=1, max_value=2))

    """


@plum.dispatch
@strip_return_annotation
@st.composite
def charts(  # noqa: F811
    draw: st.DrawFn,
    chart_cls: None = None,
    /,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    *,
    exclude: tuple[type, ...] = (cxc.Abstract0D, cxc.SphericalTwoSphere),
    ndim: int | None | st.SearchStrategy = None,
) -> cxc.AbstractChart:
    """Strategy to determine and draw chart instances."""
    # Handle ndim parameter
    ndim = draw_if_strategy(draw, ndim)
    # Exclude all dimensional flags except the target
    if isinstance(ndim, int) and ndim in cxc.DIMENSIONAL_FLAGS:
        exclude = exclude + tuple(
            flag for i, flag in cxc.DIMENSIONAL_FLAGS.items() if i != ndim
        )

    # Draw the chart class
    chart_cls = draw(  # ty: ignore[invalid-assignment]
        chart_classes(
            filter=draw_if_strategy(draw, filter),
            exclude_abstract=True,
            exclude=exclude,
        )
    )
    # Redispatch to the specific chart class strategy
    return draw(charts(chart_cls, ndim=ndim))


@plum.dispatch
@strip_return_annotation
@st.composite
def charts(  # noqa: F811
    draw: st.DrawFn,
    chart_cls: st.SearchStrategy,
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: Any = None,
) -> Any:
    """Strategy to draw chart instances."""
    # Check that the filter and exclude are blank
    if filter or exclude:
        raise ValueError(
            "When chart_cls is provided, filter and exclude must be empty."
        )
    elif ndim is not None:
        raise ValueError("When chart_cls is provided, ndim must be None.")

    # Draw the chart class from the provided strategy
    chart_cls = draw(chart_cls)
    # Redispatch to the specific chart class strategy
    return draw(charts(chart_cls, ndim=ndim))


@plum.dispatch
@strip_return_annotation
@st.composite
def charts(  # noqa: F811
    draw: st.DrawFn,
    chart_cls: type[cxc.AbstractChart],
    /,
    *,
    filter: type | tuple[type, ...] | st.SearchStrategy = (),
    exclude: tuple[type, ...] = (),
    ndim: int | None | st.SearchStrategy = None,
) -> Any:
    """Strategy to determine and draw chart instances."""
    if filter or exclude:
        raise ValueError(
            "When chart_cls is provided, filter and exclude must be empty."
        )

    if inspect.isabstract(chart_cls):
        return draw(charts(filter=chart_cls, exclude=(), ndim=ndim))  # ty: ignore[missing-argument]

    # Build and draw kwargs for required parameters
    kwargs = draw(chart_init_kwargs(chart_cls, ndim=ndim))

    # Create the instance
    chart = chart_cls(**kwargs)

    # Filter by ndim if specified
    if ndim is not None:
        assume(chart.ndim == ndim)

    return chart


#####################################################################


@plum.dispatch
def strategy_for_annotation(
    ann: type[cxc.AbstractChart],
    /,
    *,
    meta: annotations.Metadata,
) -> st.SearchStrategy:
    """Strategy for chart-typed annotations.

    For chart-valued ``__init__`` parameters, we must draw a valid chart
    *instance* rather than instantiating abstract base classes like
    ``AbstractFixedComponentsChart()``.

    """
    del meta
    return charts(filter=ann) if inspect.isclass(ann) else charts(ann)  # ty: ignore[invalid-return-type, missing-argument]
