"""Hypothesis strategies for coordinax representations."""

__all__ = ("chart_init_kwargs",)

import functools as ft
import inspect
from typing import Any

import hypothesis.strategies as st
import plum

import coordinax.charts as cxc

from .utils import get_init_params
from coordinax.hypothesis.utils import (
    annotations as antns,
    draw_if_strategy,
    strip_return_annotation,
)


@plum.dispatch.abstract
def chart_init_kwargs(
    draw: st.DrawFn,
    chart_class: Any,
    /,
    *,
    ndim: int | None | st.SearchStrategy = None,
) -> dict[str, Any]:
    """Strategy to draw initialization kwargs for a chart class.

    This strategy generates valid keyword arguments that can be used to
    instantiate a given chart class. It inspects the chart class's
    initialization signature and generates appropriate values for all
    required parameters.

    Parameters
    ----------
    draw
        Hypothesis draw function. Automatically provided by hypothesis.
    chart_class
        The chart class to generate init kwargs for, or a strategy that
        generates one. Must be a subclass of `AbstractChart`.
    ndim
        Optional `chart.ndim` constraint (currently unused, reserved for future
        functionality). By default None.

    Returns
    -------
    dict[str, Any]
        A dictionary of keyword arguments suitable for instantiating the
        chart class.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.hypothesis.main as cxst
    >>> from hypothesis import given

    Generate init kwargs for a specific chart class:

    >>> @given(kwargs=cxst.chart_init_kwargs(cxc.SpaceTimeCT))
    ... def test_spacetime_kwargs(kwargs):
    ...     # kwargs will contain 'spatial_chart' and 'c' keys
    ...     assert 'spatial_chart' in kwargs
    ...     chart = cxc.SpaceTimeCT(**kwargs)
    ...     assert isinstance(chart, cxc.SpaceTimeCT)

    Use with chart_classes strategy:

    >>> @given(
    ...     data=st.data(),
    ...     chart_cls=cxst.chart_classes(filter=cxc.Abstract3D),
    ... )
    ... def test_3d_chart_construction(data, chart_cls):
    ...     kwargs = data.draw(cxst.chart_init_kwargs(chart_cls))
    ...     chart = chart_cls(**kwargs)
    ...     assert chart.ndim == 3

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
@ft.lru_cache(maxsize=128, typed=True)
@strip_return_annotation
@st.composite
def chart_init_kwargs(  # noqa: F811
    draw: st.DrawFn,
    chart_class: type[cxc.AbstractChart],
    /,
    *,
    ndim: int | None | st.SearchStrategy = None,
) -> dict[str, Any]:
    """Strategy to draw initialization kwargs for a chart class.

    >>> import coordinax.charts as cxc
    >>> import hypothesis.strategies as st
    >>> import coordinax.hypothesis.main as cxst
    >>> from hypothesis import given

    >>> @given(
    ...     data=st.data(),
    ...     chart_cls=cxst.chart_classes(filter=cxc.Abstract3D),
    ... )
    ... def test_3d_chart_construction(data, chart_cls):
    ...     kwargs = data.draw(cxst.chart_init_kwargs(chart_cls))
    ...     chart = chart_cls(**kwargs)
    ...     assert chart.ndim == 3

    """
    # Handle dimensionality alias and strategy
    ndim = draw_if_strategy(draw, ndim)

    # Get a dictionary of all the required parameters for cls.__init__
    required_params = get_init_params(chart_class)

    # If there are no required parameters, return empty dict strategy
    if not required_params:
        # No required parameters
        return {}

    # Build a strategy for each parameter.
    strategies = {}
    for k, param in required_params.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            msg = f"Parameter '{k}' of {chart_class} has no type annotation"
            raise ValueError(msg)

        # Generate strategy for this parameter's annotation.
        # Use cached version for performance.
        # Need to wrap annotation if it's not directly inspectable.
        wrapped_ann = antns.wrap_if_not_inspectable(ann)
        strategies[k] = antns.cached_strategy_for_annotation(wrapped_ann)

    # Combine all parameter strategies into a single kwargs dict strategy
    return draw(st.fixed_dictionaries(strategies))


@plum.dispatch
@ft.lru_cache(maxsize=128, typed=True)
@strip_return_annotation
@st.composite
def chart_init_kwargs(  # noqa: F811
    draw: st.DrawFn,
    chart_class: st.SearchStrategy,
    /,
    *,
    ndim: int | None | st.SearchStrategy = None,
) -> dict[str, Any]:
    """Strategy to draw initialization kwargs for a chart class.

    >>> import coordinax.charts as cxc
    >>> import hypothesis.strategies as st
    >>> import coordinax.hypothesis.main as cxst
    >>> from hypothesis import given

    >>> @given(
    ...     data=st.data(),
    ...     chart_cls=cxst.chart_classes(filter=cxc.Abstract3D),
    ... )
    ... def test_3d_chart_construction(data, chart_cls):
    ...     kwargs = data.draw(cxst.chart_init_kwargs(chart_cls))
    ...     chart = chart_cls(**kwargs)
    ...     assert chart.ndim == 3

    """
    # Draw the chart class if it's a strategy
    chart_class = draw_if_strategy(draw, chart_class)

    # Delegate to the non-strategy version for the actual kwargs generation
    return draw(chart_init_kwargs(chart_class, ndim=ndim))
