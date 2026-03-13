"""Hypothesis strategies for CDict objects.

A CDict is a mapping from component-name strings to quantity-like array values.
This module provides strategies for generating valid CDict objects that match
chart component schemas.

"""

__all__ = ("cdicts",)

from typing import Any

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis.extra.array_api import make_strategies_namespace

import unxt as u
import unxt_hypothesis as ust

import coordinax.charts as cxc
from coordinax.hypothesis.utils import draw_if_strategy
from coordinax.internal.custom_types import CDict, Shape

# Create array API strategies namespace for JAX
xps = make_strategies_namespace(jnp)


@st.composite
def cdicts(
    draw: st.DrawFn,
    chart: cxc.AbstractChart[Any, Any] | st.SearchStrategy[cxc.AbstractChart[Any, Any]],
    *,
    dtype: Any | st.SearchStrategy = jnp.float32,
    shape: int | tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] = (),
    elements: st.SearchStrategy[float] | None = None,
) -> CDict:
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
    >>> from hypothesis import given
    >>> import coordinax.main as cx
    >>> import coordinax.hypothesis.main as cxst

    Generate CDict for Cartesian chart:

    >>> @given(p=cxst.cdicts(cxc.cart3d))
    ... def test_point_pdict(p):
    ...     assert set(p.keys()) == {"x", "y", "z"}
    ...     assert all(isinstance(v, u.Q) for v in p.values())

    Generate CDict with chart as a strategy (draws chart first, then builds CDict):

    >>> @given(p=cxst.cdicts(cxst.charts(filter=cxc.Abstract3D)))
    ... def test_any_3d_chart_pdict(p):
    ...     assert len(p) == 3  # 3D charts have 3 components

    """
    # Draw chart if it's a strategy
    chart = draw_if_strategy(draw, chart)

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
