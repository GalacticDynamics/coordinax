"""Hypothesis strategies for coordinax charts."""

__all__ = ("charts_like",)


from typing import Any

import hypothesis.strategies as st
from hypothesis import assume

import coordinax.charts as cxc
from .charts import charts
from .utils import can_point_realization_map
from coordinax.hypothesis.utils import draw_if_strategy


@st.composite
def charts_like(
    draw: st.DrawFn,
    /,
    chart: cxc.AbstractChart[Any, Any] | st.SearchStrategy[cxc.AbstractChart[Any, Any]],
) -> cxc.AbstractChart[Any, Any]:
    """Generate charts similar to the provided one.

    This strategy inspects the provided chart to determine its flags
    (e.g., Abstract1D, Abstract2D, Abstract3D, AbstractSpherical3D, etc.) and
    dimensionality, then generates new charts matching those criteria.

    Parameters
    ----------
    draw
        The draw function used by the hypothesis composite strategy.
        Automatically provided by hypothesis.
    chart
        The template chart to match, or a strategy that generates one.

    Returns
    -------
    AbstractChart
        A new chart instance with the same flags and dimensionality
        as the template.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.hypothesis.main as cxst
    >>> from hypothesis import given

    >>> # Generate charts
    >>> @given(chart=cxst.charts_like(cxc.cart3d))
    ... def test_3d(rep):
    ...     assert isinstance(rep, cxc.Abstract3D)
    ...     assert rep.ndim == 3

    >>> # Generate charts like a 2D chart
    >>> @given(chart=cxst.charts_like(cxc.polar2d))
    ... def test_2d(rep):
    ...     assert isinstance(rep, cxc.Abstract2D)
    ...     assert rep.ndim == 2

    """
    # Draw the template chart if it's a strategy
    template = draw_if_strategy(draw, chart)

    # Extract flags by looking through the MRO for AbstractDimensionalFlag subclasses
    flags: tuple[type[cxc.AbstractDimensionalFlag | cxc.AbstractChart[Any, Any]], ...]
    flags = tuple(
        base
        for base in type(template).mro()
        if (
            issubclass(base, cxc.AbstractDimensionalFlag)
            and base is not cxc.AbstractDimensionalFlag
        )
    )

    # If no flags found, just use the base type
    if not flags:
        flags = (type(template),)

    # Generate a new chart with the same flags and dimensionality.
    # Keep the template's class available even if it's excluded by default.
    exclude = tuple(
        ex
        for ex in (cxc.Abstract0D, cxc.SphericalTwoSphere)
        if not isinstance(template, ex)
    )
    chart = draw(charts(filter=flags, ndim=template.ndim, exclude=exclude))
    assume(can_point_realization_map(chart, template))
    assume(can_point_realization_map(template, chart))
    return chart
