"""Hypothesis strategies for coordinax representations."""

__all__ = ("chart_init_kwargs", "build_init_kwargs_strategy")

import functools as ft
import inspect

from typing import Any

import hypothesis.strategies as st
import plum

import coordinax.charts as cxc
from .utils import get_init_params
from coordinax_hypothesis.utils import annotations as antns, draw_if_strategy


@st.composite
def chart_init_kwargs(
    draw: st.DrawFn,
    /,
    chart_class: type[cxc.AbstractChart] | st.SearchStrategy[type[cxc.AbstractChart]],  # type: ignore[type-arg]
    *,
    ndim: (int | None | st.SearchStrategy[int | None]) = None,
    dimensionality: (int | None | st.SearchStrategy[int | None]) = None,
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
    dimensionality
        Deprecated alias for `ndim`.

    Returns
    -------
    dict[str, Any]
        A dictionary of keyword arguments suitable for instantiating the
        chart class.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax_hypothesis.core as cxst
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
    ...     chart_cls=cxst.chart_classes(filter=cxc.Abstract3D),
    ...     kwargs=cxst.chart_init_kwargs(
    ...         cxst.chart_classes(filter=cxc.Abstract3D)
    ...     )
    ... )
    ... def test_3d_chart_construction(chart_cls, kwargs):
    ...     # Construct chart from generated kwargs
    ...     chart = chart_cls(**kwargs)
    ...     assert chart.ndim == 3

    """
    # Draw the chart class if it's a strategy
    chart_class = draw_if_strategy(draw, chart_class)

    # Handle dimensionality alias and strategy
    if ndim is None and dimensionality is not None:
        ndim = dimensionality
    ndim = draw_if_strategy(draw, ndim)

    # Build and draw the kwargs strategy for this chart class
    kwargs_strategy = cached_build_init_kwargs_strategy(chart_class, dim=ndim)
    return draw(kwargs_strategy)


# ============================================================================


# Cache build_init_kwargs_strategy since it's called repeatedly for the same
# classes Strategies are immutable and deterministic, so this is safe
@ft.lru_cache(maxsize=128)
def cached_build_init_kwargs_strategy(
    cls: type[cxc.AbstractChart],  # type: ignore[type-arg]
    *,
    dim: int | None,
) -> st.SearchStrategy:
    """Return cached wrapper around build_init_kwargs_strategy."""
    return build_init_kwargs_strategy(cls, dim=dim)


# -------------------------------------------------------------------


@plum.dispatch.abstract
def build_init_kwargs_strategy(cls: type, /, *, dim: int | None) -> st.SearchStrategy:
    pass


@plum.dispatch(precedence=-1)
def build_init_kwargs_strategy(
    cls: type[cxc.AbstractChart],  # type: ignore[type-arg]
    /,
    *,
    dim: int | None,
) -> st.SearchStrategy:
    """Build a strategy that generates valid __init__ kwargs for a class.

    Parameters
    ----------
    cls : type
        The class to generate kwargs for.
    dim : int | None
        Optional dimensionality constraint for the generated arguments.

    Returns
    -------
    st.SearchStrategy[dict[str, Any]]
        A strategy that generates dictionaries of keyword arguments
        suitable for passing to cls.__init__().

    """
    # Get a dictionary of all the required parameters for cls.__init__
    required_params = get_init_params(cls)

    # If there are no required parameters, return empty dict strategy
    if not required_params:
        # No required parameters
        return st.just({})

    # Build a strategy for each parameter.
    strategies = {}
    for k, param in required_params.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            msg = f"Parameter '{k}' of {cls} has no type annotation"
            raise ValueError(msg)

        # Generate strategy for this parameter's annotation.
        # Use cached version for performance.
        # Need to wrap annotation if it's not directly inspectable.
        wrapped_ann = antns.wrap_if_not_inspectable(ann)
        strategies[k] = antns.cached_strategy_for_annotation(wrapped_ann)

    # Combine all parameter strategies into a single kwargs dict strategy
    return st.fixed_dictionaries(strategies)
