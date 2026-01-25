"""Hypothesis strategies for coordinax charts."""

__all__ = ("chart_classes",)


from typing import Any

import hypothesis.strategies as st

import coordinax.charts as cxc
from coordinax_hypothesis._src.utils import draw_if_strategy, get_all_subclasses

# =============================================================================
# Chart classes


@st.composite
def chart_classes(
    draw: st.DrawFn,
    /,
    filter: type
    | tuple[type, ...]
    | st.SearchStrategy[type | tuple[type, ...]] = object,
    *,
    exclude_abstract: bool | st.SearchStrategy[bool] = True,
    exclude: tuple[type, ...] = (),
) -> type[cxc.AbstractChart[Any, Any]]:
    """Strategy to draw chart classes.

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    filter
        A class or tuple of classes to limit the charts to, by default
        `object`.  Can be a single type or a tuple of types.
    exclude_abstract
        Whether to exclude abstract subclasses, by default True.
    exclude
        Specific classes to exclude, by default
        ``(coordinax.charts.Abstract0D, coordinax.charts.TwoSphere)``.

    Returns
    -------
    type[AbstractChart]
        A chart class.

    Examples
    --------
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst

    >>> # Draw any chart class
    >>> chart_class_strategy = cxst.chart_classes()
    >>> # Draw 1D velocity chart classes only
    >>> vel_1d_chart_class_strategy = cxst.chart_classes(filter=cxc.Abstract1D)

    """
    classes = get_all_subclasses(
        cxc.AbstractChart,
        filter=draw_if_strategy(draw, filter),
        exclude_abstract=draw_if_strategy(draw, exclude_abstract),
        exclude=exclude,
    )
    return draw(st.sampled_from(classes))
