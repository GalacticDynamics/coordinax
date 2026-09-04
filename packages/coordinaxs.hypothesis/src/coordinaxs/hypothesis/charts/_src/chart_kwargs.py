"""Hypothesis strategies for coordinax representations."""

__all__ = ("chart_init_kwargs",)

import functools as ft
import inspect

from typing import Any

import hypothesis.strategies as st
import plum
import unxt as u
import unxts.hypothesis as ust

import coordinax.charts as cxc

from .utils import get_init_params
from coordinaxs.hypothesis.utils import (
    annotations as antns,
    draw_if_strategy,
    strip_return_annotation,
)


@plum.dispatch.abstract
def chart_init_kwargs(
    draw: st.DrawFn, chart_class: Any, /, *, ndim: int | None | st.SearchStrategy = None
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
    >>> import coordinaxs.hypothesis.main as cxst
    >>> from hypothesis import given

    Generate init kwargs for a specific chart class:

    >>> @given(kwargs=cxst.chart_init_kwargs(cxc.MinkowskiCT))
    ... def test_minkowskict_kwargs(kwargs):
    ...     # MinkowskiCT has no required init params; kwargs will be empty
    ...     chart = cxc.MinkowskiCT(**kwargs)
    ...     assert isinstance(chart, cxc.MinkowskiCT)

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
def chart_init_kwargs(
    draw: st.DrawFn,
    chart_class: type[cxc.AbstractChart],
    /,
    *,
    ndim: int | None | st.SearchStrategy = None,
) -> dict[str, Any]:
    """Strategy to draw initialization kwargs for a chart class.

    >>> import coordinax.charts as cxc
    >>> import hypothesis.strategies as st
    >>> import coordinaxs.hypothesis.main as cxst
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
    # TODO: Handle dimensionality alias and strategy
    # ndim = draw_if_strategy(draw, ndim)

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
def chart_init_kwargs(
    draw: st.DrawFn,
    chart_class: st.SearchStrategy,
    /,
    *,
    ndim: int | None | st.SearchStrategy = None,
) -> dict[str, Any]:
    """Strategy to draw initialization kwargs for a chart class.

    >>> import coordinax.charts as cxc
    >>> import hypothesis.strategies as st
    >>> import coordinaxs.hypothesis.main as cxst
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


@plum.dispatch
@ft.lru_cache(maxsize=128, typed=True)
@strip_return_annotation
@st.composite
def chart_init_kwargs(
    draw: st.DrawFn,
    chart_class: type[cxc.ProlateSpheroidal3D],
    /,
    *,
    ndim: int | None | st.SearchStrategy = None,
) -> dict[str, Any]:
    """`Delta` is a focal *length*, which its annotation cannot say.

    The generic version reads each parameter's jaxtyping annotation, and
    `Delta`'s pins the container and shape but not the dimension -- it cannot,
    without narrowing to one quantity type and losing the
    `Quantity`/`StaticQuantity` choice that makes differentiability opt-in.
    Drawing from the annotation alone gives seconds as readily as parsecs, and
    `ProlateSpheroidal3D` rejects those at construction.

    Both containers are drawn, because which one is used is a real distinction
    for this chart: a `StaticQuantity` contributes no pytree leaves and a
    `Quantity` contributes one.

    The magnitude is kept modest. `Delta` is squared to bound `mu`, so a draw
    near the float ceiling squares to infinity -- representable as a chart,
    but not as the domain the strategies then draw points from. The limits are
    powers of two so that they are exact at float32, which `floats(width=32)`
    requires of its bounds.

    >>> import coordinax.charts as cxc
    >>> import coordinaxs.hypothesis.main as cxst
    >>> from hypothesis import given
    >>> import unxt as u

    >>> @given(kwargs=cxst.chart_init_kwargs(cxc.ProlateSpheroidal3D))
    ... def test_delta_is_a_length(kwargs):
    ...     assert u.dimension_of(kwargs["Delta"]) == u.dimension("length")
    ...     cxc.ProlateSpheroidal3D(**kwargs)

    """
    unit = draw(ust.units(u.dimension("length")))
    value = draw(
        st.floats(
            min_value=0.125,
            max_value=1024.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        )
    )
    container = draw(st.sampled_from([u.Q, u.quantity.StaticQuantity]))
    return {"Delta": container(value, unit)}
