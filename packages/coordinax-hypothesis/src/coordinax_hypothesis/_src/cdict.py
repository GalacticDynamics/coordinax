"""Hypothesis strategies for CsDict objects.

A CsDict is a mapping from component-name strings to quantity-like array values.
This module provides strategies for generating valid CsDict objects that match
chart component schemas and role dimension requirements.
"""

__all__ = ("cdicts",)

from typing import Any

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis.extra.array_api import make_strategies_namespace

import unxt as u
import unxt_hypothesis as ust

import coordinax.charts as cxc
import coordinax.roles as cxr
from .utils import draw_if_strategy
from coordinax._src.constants import ACCELERATION, LENGTH, SPEED
from coordinax._src.custom_types import Shape
from coordinax.api import CsDict

# Create array API strategies namespace for JAX
xps = make_strategies_namespace(jnp)


def _get_role_dimension_constraint(
    role: cxr.AbstractRole,
) -> u.AbstractDimension | None:
    """Return the physical dimension constraint for a role, or None if unconstrained.

    Parameters
    ----------
    role
        The role object

    Returns
    -------
    u.AbstractDimension | None
        Physical dimension that all components must satisfy, or None if
        component dimensions are determined by chart.coord_dimensions

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u
    >>> _get_role_dimension_constraint(cxr.point) is None
    True

    >>> _get_role_dimension_constraint(cxr.phys_disp)
    PhysicalType('length')

    """
    if isinstance(role, cxr.Point):
        return None  # Point: no additional constraint; use chart.coord_dimensions
    if isinstance(role, cxr.PhysDisp):
        # Pos requires physical dimension = length
        return LENGTH
    if isinstance(role, cxr.PhysVel):
        # Vel requires physical dimension = length/time
        return SPEED
    if isinstance(role, cxr.PhysAcc):
        # PhysAcc requires physical dimension = length/time^2
        return ACCELERATION
    if isinstance(role, (cxr.CoordDisp, cxr.CoordVel, cxr.CoordAcc)):
        # Coordinate-basis roles have heterogeneous per-component dimensions.
        return None
    # For unknown/future roles, return None (unconstrained)
    return None


@st.composite
def cdicts(
    draw: st.DrawFn,
    chart: cxc.AbstractChart[Any, Any] | st.SearchStrategy[cxc.AbstractChart[Any, Any]],
    role: cxr.AbstractRole | st.SearchStrategy[cxr.AbstractRole] | None = None,
    *,
    dtype: Any | st.SearchStrategy = jnp.float32,
    shape: int | tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] = (),
    elements: st.SearchStrategy[float] | None = None,
) -> CsDict:
    """Generate a valid CsDict matching chart components and role constraints.

    A CsDict is a mapping from component-name strings to quantity-like values,
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
    role
        Role instance or strategy generating one. If None, defaults to Point role.
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
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst

    Generate CsDict for Cartesian chart with Point role:

    >>> @given(p=cxst.cdicts(cxc.cart3d, cxr.point))
    ... def test_point_pdict(p):
    ...     assert set(p.keys()) == {"x", "y", "z"}
    ...     assert all(isinstance(v, u.Q) for v in p.values())

    Generate CsDict with Pos role (uniform length dimension):

    >>> @given(p=cxst.cdicts(cxc.cart3d, cxr.phys_disp))
    ... def test_disp_pdict(p):
    ...     assert all(u.dimension_of(v) == u.dimension("length") for v in p.values())

    Generate CsDict with chart as a strategy (draws chart first, then builds CsDict):

    >>> @given(p=cxst.cdicts(cxst.charts(filter=cxc.Abstract3D), cxr.point))
    ... def test_any_3d_chart_pdict(p):
    ...     assert len(p) == 3  # 3D charts have 3 components

    """
    # Draw chart if it's a strategy
    chart = draw_if_strategy(draw, chart)

    # Draw role if it's a strategy
    role = cxr.point if role is None else draw_if_strategy(draw, role)

    # Draw shape if it's a strategy
    shape: Shape = draw_if_strategy(draw, shape)

    # Get dimension constraint from role
    role_dim = _get_role_dimension_constraint(role)

    # Build data dictionary
    data: CsDict = {}

    role_dims = (
        role.dimensions(chart)
        if isinstance(role, (cxr.CoordDisp, cxr.CoordVel, cxr.CoordAcc))
        else None
    )

    for component_name, component_dim in zip(
        chart.components, chart.coord_dimensions, strict=True
    ):
        if role_dims is not None:
            dim = role_dims[component_name]
        elif role_dim is not None:
            # Physical tangent role: use role dimension for all components
            dim = role_dim
        else:
            # Point or unconstrained role: use chart's per-component dimension
            # Convert string dimensions to AbstractDimension if needed
            dim = (
                u.dimension(component_dim)
                if isinstance(component_dim, str) and component_dim is not None
                else component_dim
            )

        # Generate quantity for this component
        data[component_name] = draw(
            ust.quantities(
                unit=dim,
                dtype=dtype,
                shape=shape,
                elements=elements,
            )
        )

    return data
