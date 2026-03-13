"""Hypothesis strategies for coordinax charts."""

__all__ = ("charts",)


from typing import Any

import hypothesis.strategies as st
import plum

import coordinax.charts as cxc
from .charts_specific import find_chart_strategy
from .classes import chart_classes
from coordinax.hypothesis.utils import annotations, draw_if_strategy


@st.composite
def charts(
    draw: st.DrawFn,
    /,
    filter: type | tuple[type, ...] | st.SearchStrategy[type | tuple[type, ...]] = (),
    *,
    exclude: tuple[type, ...] = (cxc.Abstract0D, cxc.SphericalTwoSphere),
    ndim: int | None | st.SearchStrategy[int | None] = None,
) -> cxc.AbstractChart[Any, Any]:
    """Strategy to draw chart instances.

    Parameters
    ----------
    draw
        The draw function used by the hypothesis composite strategy.
        Automatically provided by hypothesis.
    filter
        A class or tuple of classes to limit the charts to, by default
        `()` (no additional filter). Can be a single type or a tuple of types.

        For example:

        - `coordinax.charts.Abstract0D` to limit to 0D charts.
        - `coordinax.charts.Abstract1D` to limit to 1D charts.
        - `coordinax.charts.Abstract2D` to limit to 2D charts.
        - `coordinax.charts.Abstract3D` to limit to 3D charts.

        In combination, this can be used to draw charts that satisfy multiple
        criteria, e.g., ``filter=(coordinax.charts.Abstract3D,
        coordinax.charts.AbstractSpherical3D)``.  Just note that some 3-D
        charts, e.g. SpaceTimeCT[Cart2D] are not subclasses of
        {class}`coordinax.charts.Abstract3D` and will be excluded unless
        explicitly added.
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
    # Handle ndim parameter
    ndim = draw_if_strategy(draw, ndim)
    # Exclude all dimensional flags except the target
    if isinstance(ndim, int) and ndim in cxc.DIMENSIONAL_FLAGS:
        exclude = exclude + tuple(
            flag for i, flag in cxc.DIMENSIONAL_FLAGS.items() if i != ndim
        )

    # Draw the chart class
    chart_cls = draw(
        chart_classes(
            filter=draw_if_strategy(draw, filter),  # type: ignore[arg-type]
            exclude_abstract=True,
            exclude=exclude,
        )
    )

    # Draw the chart instance using the appropriate strategy for the chart class
    specific_chart_strategy = find_chart_strategy(chart_cls)
    return draw(specific_chart_strategy(chart_cls, ndim=ndim))


# ---------------------------------------------------------


@plum.dispatch
def strategy_for_annotation(
    ann: type[cxc.AbstractChart],  # type: ignore[type-arg]
    /,
    *,
    meta: annotations.Metadata,
) -> st.SearchStrategy:
    """Strategy for chart-typed annotations.

    For chart-valued ``__init__`` parameters (e.g.
    ``SpaceTimeCT.spatial_chart``), we must draw a valid chart *instance* rather
    than instantiating abstract base classes like
    ``AbstractFixedComponentsChart()``.

    """
    del meta
    return charts(filter=ann)
