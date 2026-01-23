"""Hypothesis strategies for CsDict objects.

A CsDict is a mapping from component-name strings to quantity-like array values.
This module provides strategies for generating valid CsDict objects that match
chart component schemas and role dimension requirements.
"""

from coordinax._src.custom_types import CsDict

__all__ = ("pdicts",)

from typing import Any

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis.extra.array_api import make_strategies_namespace

import unxt as u
import unxt_hypothesis as ust

import coordinax as cx
from .utils import draw_if_strategy

# Create array API strategies namespace for JAX
xps = make_strategies_namespace(jnp)


def _get_role_dimension_constraint(
    role: cx.roles.AbstractRole,
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
    >>> _get_role_dimension_constraint(cx.roles.point)
    None  # Point has no constraint; chart.coord_dimensions apply

    >>> _get_role_dimension_constraint(cx.roles.phys_disp)
    <Unit "m">  # Pos requires length dimension

    """
    if isinstance(role, cx.roles.Point):
        return None  # Point: no additional constraint; use chart.coord_dimensions
    if isinstance(role, cx.roles.PhysDisp):
        # Pos requires physical dimension = length
        return u.dimension("length")
    if isinstance(role, cx.roles.PhysVel):
        # Vel requires physical dimension = length/time
        return u.dimension("speed")
    if isinstance(role, cx.roles.PhysAcc):
        # PhysAcc requires physical dimension = length/time^2
        return u.dimension("acceleration")
    # For unknown/future roles, return None (unconstrained)
    return None


@st.composite  # type: ignore[untyped-decorator]
def pdicts(
    draw: st.DrawFn,
    chart: cx.charts.AbstractChart[Any, Any],
    role: cx.roles.AbstractRole
    | st.SearchStrategy[cx.roles.AbstractRole]
    | None = None,
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
        Chart instance defining the component schema
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

    >>> @given(p=cxst.pdicts(cx.charts.cart3d, cx.roles.point))
    ... def test_point_pdict(p):
    ...     assert set(p.keys()) == {"x", "y", "z"}
    ...     assert all(isinstance(v, u.Q) for v in p.values())

    Generate CsDict with Pos role (uniform length dimension):

    >>> @given(p=cxst.pdicts(cx.charts.cart3d, cx.roles.phys_disp))
    ... def test_disp_pdict(p):
    ...     assert all(u.dimension_of(v) == u.dimension("length") for v in p.values())

    """
    # Draw role if it's a strategy
    role = cx.roles.point if role is None else draw_if_strategy(draw, role)

    # Draw shape if it's a strategy
    shape = draw_if_strategy(draw, shape)

    # Get dimension constraint from role
    role_dim = _get_role_dimension_constraint(role)

    # Build data dictionary
    data: CsDict = {}

    for component_name, component_dim in zip(
        chart.components, chart.coord_dimensions, strict=True
    ):
        if role_dim is not None:
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
