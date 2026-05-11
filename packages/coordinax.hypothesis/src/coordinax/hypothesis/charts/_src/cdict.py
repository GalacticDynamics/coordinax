"""Hypothesis strategies for CDict objects.

A CDict is a mapping from component-name strings to quantity-like array values.
This module provides strategies for generating valid CDict objects that match
chart component schemas.

"""

__all__ = ("cdicts",)

from typing import Any

import hypothesis.strategies as st
import jax.numpy as jnp
import plum
import unxt as u
import unxt_hypothesis as ust
from hypothesis.extra.array_api import make_strategies_namespace

import coordinax.charts as cxc

from coordinax.hypothesis.utils import (
    CDict,
    Shape,
    draw_if_strategy,
    strip_return_annotation,
)

# Create array API strategies namespace for JAX
xps = make_strategies_namespace(jnp)


@plum.dispatch.abstract
def cdicts(*args: Any, **kwargs: Any) -> CDict:
    """Generate a valid CDict matching chart components and role constraints.

    A CDict is a mapping from component-name strings to quantity-like values,
    constrained by:
    - Keys must exactly match `chart.components`
    - For physical tangent roles (Pos, Vel, PhysAcc), all component values must
      have the same physical dimension (length, length/time, or length/time²)
    - For Point role, component dimensions follow `chart.coord_dimensions`

    Parameters
    ----------
    draw
        The Hypothesis draw function (provided automatically)
    chart
        Chart instance or strategy generating one, defining the component schema.
    dtype
        Data type for array components (default: jnp.float32)
    shape
        Shape for array components. Can be int, tuple of ints, or strategy.
        Default is scalar (shape=())
    elements
        Strategy for generating individual float values. If None, uses finite floats.

    Returns
    -------
    dict[str, Any]
        A mapping from component names to quantity-like values

    Examples
    --------
    >>> import coordinax.hypothesis.main as cxst
    >>> from hypothesis import given
    >>> import coordinax.charts as cxc

    Generate CDict for Cartesian chart:

    >>> @given(p=cxst.cdicts(cxc.cart3d))
    ... def test_cdict(p):
    ...     assert set(p.keys()) == {"x", "y", "z"}
    ...     assert all(isinstance(v, u.Q) for v in p.values())

    Generate CDict with chart as a strategy (draws chart first, then builds CDict):

    >>> @given(p=cxst.cdicts(cxst.charts(filter=cxc.Abstract3D)))
    ... def test_any_3d_chart_cdict(p):
    ...     assert len(p) == 3  # 3D charts have 3 components

    This can also be called without specifying a chart strategy, in which case
    it defaults to drawing from all charts:

    >>> @given(p=cxst.cdicts())
    ... def test_any_chart_cdict(p):
    ...     assert isinstance(p, dict)

    """
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
@strip_return_annotation
@st.composite
def cdicts(  # noqa: F811
    draw: st.DrawFn,
    chart: st.SearchStrategy = st.deferred(lambda: cxc.charts()),  # ty: ignore[unresolved-attribute]
    /,
    **kwargs: Any,
) -> CDict:
    """Draw a CDict from a strategy that generates charts.

    >>> import coordinax.hypothesis.main as cxst
    >>> from hypothesis import given
    >>> import coordinax.charts as cxc

    >>> @given(p=cxst.cdicts(cxst.charts(filter=cxc.Abstract3D)))
    ... def test_any_3d_chart_cdict(p):
    ...     assert len(p) == 3  # 3D charts have 3 components

    This can also be called without specifying a chart strategy, in which case
    it defaults to drawing from all charts:

    >>> @given(p=cxst.cdicts())
    ... def test_any_chart_cdict(p):
    ...     assert isinstance(p, dict)

    """
    # Draw chart
    chart = draw(chart)

    # Redispatch to the more specific implementation
    return draw(cdicts(chart, **kwargs))


@plum.dispatch
@strip_return_annotation
@st.composite
def cdicts(  # noqa: F811
    draw: st.DrawFn,
    chart: cxc.AbstractChart,
    /,
    *,
    dtype: Any | st.SearchStrategy = jnp.float32,
    shape: int | tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] = (),
    elements: st.SearchStrategy[float] | None = None,
) -> CDict:
    """Generate a valid CDict matching chart components and role constraints.

    >>> import coordinax.hypothesis.main as cxst
    >>> from hypothesis import given
    >>> import coordinax.charts as cxc

    >>> @given(p=cxst.cdicts(cxc.cart3d))
    ... def test_cdict(p):
    ...     assert set(p.keys()) == {"x", "y", "z"}
    ...     assert all(isinstance(v, u.Q) for v in p.values())

    """
    # Draw shape if it's a strategy
    shape: Shape = draw_if_strategy(draw, shape)

    # Build data dictionary
    data: CDict = {}

    for cname, cdim in zip(chart.components, chart.coord_dimensions, strict=True):
        # Get the dimension of the component
        dim = u.dimension(cdim) if isinstance(cdim, str) and cdim is not None else cdim

        # Generate quantity for this component
        data[cname] = draw(
            ust.quantities(unit=dim, dtype=dtype, shape=shape, elements=elements)
        )

    return data
