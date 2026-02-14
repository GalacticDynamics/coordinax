"""Hypothesis strategies for coordinax representations."""

__all__ = ("charts",)


from typing import Any

import hypothesis.strategies as st
import plum
from hypothesis import assume

import coordinax.charts as cxc
from .chart_kwargs import chart_init_kwargs
from .classes import chart_classes
from coordinax_hypothesis.utils import annotations, draw_if_strategy


@st.composite
def charts(
    draw: st.DrawFn,
    /,
    filter: type | tuple[type, ...] | st.SearchStrategy[type | tuple[type, ...]] = (),
    *,
    exclude: tuple[type, ...] = (cxc.Abstract0D, cxc.TwoSphere),
    ndim: (int | None | st.SearchStrategy[int | None]) = None,
    dimensionality: (int | None | st.SearchStrategy[int | None]) = None,
) -> cxc.AbstractChart[Any, Any]:
    """Strategy to draw representation instances.

    Parameters
    ----------
    draw
        The draw function used by the hypothesis composite strategy.
        Automatically provided by hypothesis.
    filter
        A class or tuple of classes to limit the representations to, by default
        `()` (no additional filter). Can be a single type or a tuple of types.

        For example: - `coordinax.charts.Abstract0D` to limit to 0D representations.
        - `coordinax.charts.Abstract1D` to limit to 1D representations.  -
        `coordinax.charts.Abstract2D` to limit to 2D representations.  -
        `coordinax.charts.Abstract3D` to limit to 3D representations.

        In combination, this can be used to draw representations that satisfy
        multiple criteria, e.g., `filter=(coordinax.charts.Abstract3D,
        coordinax.charts.AbstractSpherical3D)`.  Just note that some 3-D
        representations, e.g. SpaceTimeCT[Cart2D] are not subclasses of
        `coordinax.charts.Abstract3D` and will be excluded unless explicitly added.
    exclude
        Specific classes to exclude, by default ().
    ndim
        `chart.ndim` constraint. Can be: - `None`: No constraint - An integer:
        Exact ndim match - A strategy: Draw ndim from strategy (e.g.,
          `st.integers(min_value=1, max_value=2)`)
    dimensionality
        Deprecated alias for `ndim`.

    Returns
    -------
    AbstractChart
        An instance of a representation class.

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax_hypothesis.core as cxst
    >>> import hypothesis.strategies as st

    >>> # Draw any representation instance (dimensionality > 0 by default)
    >>> chart_strategy = cxst.charts()

    >>> # Draw charts with exact ndim
    >>> exact_2d_strategy = cxst.charts(ndim=2)

    >>> # Include 0-dimensional charts
    >>> all_dim_strategy = cxst.charts(ndim=None, exclude=())

    >>> # Use a strategy to draw ndim
    >>> strategy_dim = cxst.charts(ndim=st.integers(min_value=1, max_value=2))

    """
    if ndim is None and dimensionality is not None:
        ndim = dimensionality

    # Handle ndim parameter
    ndim = draw_if_strategy(draw, ndim)
    # Exclude all dimensional flags except the target
    if isinstance(ndim, int) and ndim in cxc.DIMENSIONAL_FLAGS:
        exclude = exclude + tuple(
            flag for i, flag in cxc.DIMENSIONAL_FLAGS.items() if i != ndim
        )

    # Draw the representation class
    chart_cls = draw(
        chart_classes(
            filter=draw_if_strategy(draw, filter),
            exclude_abstract=True,
            exclude=exclude,
        )
    )

    # Build and draw kwargs for required parameters
    kwargs = draw(chart_init_kwargs(chart_cls, ndim=ndim))

    # Create the instance
    chart = chart_cls(**kwargs)

    # Filter by ndim if specified
    if ndim is not None:
        assume(chart.ndim == ndim)

    return chart


@plum.dispatch
def strategy_for_annotation(
    ann: type[cxc.AbstractChart],  # type: ignore[type-arg]
    /,
    *,
    meta: annotations.Metadata,
) -> st.SearchStrategy:
    """Strategy for chart-typed annotations.

    For chart-valued `__init__` parameters (e.g. ``SpaceTimeCT.spatial_chart``),
    we must draw a valid chart *instance* rather than instantiating abstract
    base classes like ``AbstractFixedComponentsChart()``.
    """
    del meta
    return charts(filter=ann)
