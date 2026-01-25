"""Hypothesis strategies for Coordinax vectors."""

__all__ = (
    "pointedvectors",
    "roles",
    "point_role",
    "physical_roles",
    "coord_roles",
    "vectors",
    "vectors_with_target_chart",
)

from typing import Any, Final, TypeVar

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis.extra.array_api import make_strategies_namespace

import unxt as u

import coordinax as cx
import coordinax.charts as cxc
import coordinax.roles as cxr
from .cdict import cdicts
from .charts import chart_time_chain, charts, charts_like
from .utils import draw_if_strategy, get_all_subclasses
from coordinax._src.constants import TIME

Ks = TypeVar("Ks", bound=tuple[str, ...])
Ds = TypeVar("Ds", bound=tuple[str | None, ...])

ROLES: Final = get_all_subclasses(cxr.AbstractPhysRole, exclude_abstract=True)
PHYS_ROLES: Final = tuple(r for r in ROLES if issubclass(r, cxr.AbstractPhysRole))
COORD_ROLES: Final = tuple(r for r in ROLES if issubclass(r, cxr.AbstractCoordRole))

# Create array API strategies namespace for JAX
xps = make_strategies_namespace(jnp)


def _d_dt_dim(
    dim: u.AbstractDimension | str | None, order: int
) -> u.AbstractDimension | None:
    if dim is None:
        return None
    return u.dimension(dim) / (TIME**order)


# ==============================================================================
# Role strategies


@st.composite
def roles(
    draw: st.DrawFn,
    *,
    include: tuple[type[cxr.AbstractRole], ...] | None = None,
    exclude: tuple[type[cxr.AbstractRole], ...] = (),
) -> cxr.AbstractRole:
    """Generate random Coordinax role flags.

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    include
        If provided, only generate roles from this tuple. Otherwise, all roles
        are considered (Point, PhysDisp, etc.).
    exclude
        Roles to exclude from generation. Default is empty (no exclusions).

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst

    >>> @given(role=cxst.roles())
    ... def test_any_role(role):
    ...     assert isinstance(role, cxr.AbstractRole)

    >>> @given(role=cxst.roles(include=(cxr.PhysDisp, cxr.PhysDisp)))
    ... def test_position_like_roles(role):
    ...     assert isinstance(role, (cxr.PhysDisp, cxr.PhysDisp))

    """
    # Determine candidate roles
    candidates = ROLES if include is None else include

    # Filter out excluded roles
    candidates = tuple(r for r in candidates if r not in exclude)

    if not candidates:
        msg = "No roles left after exclusions"
        raise ValueError(msg)

    # Sample one role class and instantiate it
    role_cls = draw(st.sampled_from(candidates))
    return role_cls()


@st.composite
def point_role(draw: st.DrawFn) -> cxr.Point:
    """Generate the Point role.

    Point represents an affine point on the manifold (not a tangent vector).
    This strategy always returns the Point role instance.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.roles as cxr
    >>> import coordinax_hypothesis as cxst

    >>> @given(role=cxst.point_role())
    ... def test_point_only(role):
    ...     assert isinstance(role, cxr.Point)

    """
    # Return Point role (use draw with st.just for consistency with Hypothesis patterns)
    return draw(st.just(cxr.point))


@st.composite
def physical_roles(draw: st.DrawFn) -> cxr.AbstractRole:
    """Generate physical tangent role flags (PhysDisp, PhysVel, PhysAcc).

    These are roles representing physical tangent vectors that require uniform
    physical dimension across components.

    Returns
    -------
    cxr.AbstractRole
        A physical tangent role instance: `coordinax.roles.phys_disp`,
        `coordinax.roles.phys_vel`, or `coordinax.roles.phys_acc`.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.roles as cxr
    >>> import coordinax_hypothesis as cxst

    >>> @given(role=cxst.physical_roles())
    ... def test_tangent_role(role):
    ...     # Only PhysDisp, Vel, PhysAcc (not Point)
    ...     assert isinstance(role, cxr.AbstractPhysRole)

    """
    role_cls = draw(st.sampled_from(PHYS_ROLES))
    return role_cls()


@st.composite
def coord_roles(draw: st.DrawFn) -> cxr.AbstractCoordRole:
    """Generate coordinate role flags (CoordDisp, CoordVel, CoordAcc).

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax.roles as cxr
    >>> import coordinax_hypothesis as cxst

    >>> @given(role=cxst.coord_roles())
    ... def test_tangent_role(role):
    ...     # Only CoordDisp, CoordVel, CoordAcc (not Point)
    ...     assert isinstance(role, cxr.AbstractCoordRole)

    """
    role_cls = draw(st.sampled_from(COORD_ROLES))
    return role_cls()


# ==============================================================================
# Vector strategies


@st.composite
def vectors(
    draw: st.DrawFn,
    chart: cxc.AbstractChart[Ks, Ds]
    | st.SearchStrategy[cxc.AbstractChart[Ks, Ds]] = charts(exclude=(cxc.Abstract0D,)),
    role: cxr.AbstractRole | st.SearchStrategy[cxr.AbstractRole] = point_role(),
    *,
    dtype: Any | st.SearchStrategy = jnp.float32,
    shape: int | tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] = (),
    elements: st.SearchStrategy[float] | None = None,
) -> cx.Vector[cxc.AbstractChart[Any, Any], Any, Any]:
    """Generate random `coordinax` vectors of specified dimension.

    Parameters
    ----------
    draw
        The draw function provided by {mod}`hypothesis`.
    chart
        A {mod}`coordinax` chart instance or a strategy to generate one.
        By default, a random chart is drawn using the
        {func}`~coordinax_hypothesis.charts` strategy.
    role
        The role flag for the vector. By default, the position role
        ({class}`~coordinax.roles.Point`) is used.
    dtype
        The data type for array components (default: `~jax.numpy.float32`).
    shape
        The shape for the vector components. Can be an integer (for 1D), a tuple
        of integers, or a strategy. Default is scalar (shape=()).
    elements
        Strategy for generating element values. If None, uses finite floats.

    Returns
    -------
    cx.Vector
        A {mod}`coordinax` vector of the specified dimension.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst

    >>> # Generate any vector
    >>> @given(vec=cxst.vectors())
    ... def test_vector(vec):
    ...     assert isinstance(vec, cx.Vector)

    >>> # Generate vectors with a specific representation
    >>> @given(vec=cxst.vectors(chart=cxc.cart3d))
    ... def test_cartesian_3d(vec):
    ...     assert vec.chart == cxc.cart3d

    >>> # Generate vectors with specific shape
    >>> @given(vec=cxst.vectors(shape=(10,)))
    ... def test_batched_vector(vec):
    ...     assert vec.shape == (10,)

    """
    # Draw if it's a strategy
    chart = draw_if_strategy(draw, chart)
    role = draw_if_strategy(draw, role)
    role_inst = role() if isinstance(role, type) else role

    # Generate CsDict data using the cdict strategy
    data = draw(
        cdicts(
            chart=chart,
            role=role_inst,
            dtype=dtype,
            shape=shape,
            elements=elements,
        )
    )

    return cx.Vector(data=data, chart=chart, role=role_inst)


@st.composite
def vectors_with_target_chart(
    draw: st.DrawFn,
    /,
    chart: cxc.AbstractChart[Ks, Ds]
    | st.SearchStrategy[cxc.AbstractChart[Ks, Ds]] = charts(),
    role: cxr.AbstractRole | st.SearchStrategy[cxr.AbstractRole] = point_role(),
    *,
    dtype: Any | st.SearchStrategy = jnp.float32,
    shape: int | tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] = (),
    elements: st.SearchStrategy[float] | None = None,
) -> tuple[
    cx.Vector[cxc.AbstractChart[Any, Any], Any], tuple[cxc.AbstractChart[Any, Any], ...]
]:
    """Generate a vector and a time-derivative chain with matching flags.

    This is useful for testing conversion operations where you need a vector
    and a full set of target representations (following the time antiderivative
    chain) that it can be converted to.

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    chart
        A Coordinax chart instance or a strategy to generate one
        for the source vector. By default, a random chart is drawn.
    role
        The role flag for the source vector. By default, the position role
        ({class}`~coordinax.roles.Point`) is used.
    dtype
        The data type for array components (default: jnp.float32).
    shape
        The shape for the vector components. Default is scalar (shape=()).
    elements
        Strategy for generating element values. If None, uses finite floats.

    Returns
    -------
    tuple[cx.Vector, tuple[cxc.AbstractChart, ...]]
        A tuple of (vector, target_chain) where target_chain is a tuple of
        representations following the time antiderivative pattern, all matching
        the flags of the source vector's representation.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst

    >>> # Generate a position vector and target chain for conversion
    >>> @given(
    ...     vec_charts=cxst.vectors_with_target_chart(
    ...         chart=cxc.cart3d, role=cxr.Point
    ...     )
    ... )
    ... def test_conversion(vec_charts):
    ...     vec, target_chain = vec_charts
    ...     # target_chain is (Point,)
    ...     for target_chart in target_chain:
    ...         converted = vec.vconvert(target_chart)
    ...         assert converted.chart == target_chart

    """
    role = draw_if_strategy(draw, role)

    # Draw the source vector
    vec = draw(vectors(chart, role, dtype=dtype, shape=shape, elements=elements))

    # Draw a target representation with matching dimensionality
    target_chart = draw(charts_like(vec.chart))

    # For Point role, there is no time-antiderivative chain; just return the
    # single chart. For physical tangent roles (Pos, Vel, PhysAcc), generate the
    # full time-derivative chain.
    target_chain: tuple[cxc.AbstractChart[Any, Any], ...]
    if isinstance(role, cxr.Point):
        target_chain = (target_chart,)
    else:
        # Generate the full time-derivative chain from the target representation
        target_chain = draw(chart_time_chain(role, target_chart))

    return vec, target_chain


# ==============================================================================
# Bundle strategies


@st.composite
def pointedvectors(
    draw: st.DrawFn,
    *,
    base_chart: cxc.AbstractChart[Any, Any]
    | st.SearchStrategy[cxc.AbstractChart[Any, Any]] = charts(
        exclude=(cxc.Abstract0D,)
    ),
    field_keys: tuple[str, ...] = ("velocity",),
    field_roles: tuple[cxr.AbstractRole, ...]
    | st.SearchStrategy[tuple[cxr.AbstractRole, ...]]
    | None = None,
    dtype: Any | st.SearchStrategy = jnp.float32,
    shape: int | tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] = (),
    elements: st.SearchStrategy[float] | None = None,
) -> cx.PointedVector:
    """Generate random PointedVector instances.

    Parameters
    ----------
    draw
        The draw function provided by Hypothesis.
    base_chart
        Chart for the base vector, or strategy to generate one.
        Default generates random representations.
    field_keys
        Names of field vectors. Default is ("velocity",).
    field_roles
        Tuple of roles for field vectors, matching length of field_keys.
        If `None`, generates tangent-like roles (Vel, PhysAcc) randomly.
        Cannot include Point role (enforced by PointedVector validation).
    dtype
        Data type for array components (default: jnp.float32).
    shape
        Shape for vector components. Can be integer, tuple, or strategy.
        Default is scalar (shape=()).
    elements
        Strategy for generating element values. If None, uses finite floats.

    Returns
    -------
    cx.PointedVector
        A bundle with base (role=Point) and field vectors.

    Raises
    ------
    ValueError
        If field_roles contains Point role or length doesn't match field_keys.

    Examples
    --------
    >>> from hypothesis import given
    >>> import coordinax as cx
    >>> import coordinax_hypothesis as cxst

    Generate default bundle with velocity:

    >>> @given(bundle=cxst.pointedvectors())
    ... def test_bundle(bundle):
    ...     assert isinstance(bundle, cx.PointedVector)
    ...     assert isinstance(bundle.base.role, cxr.Point)
    ...     assert "velocity" in bundle.keys()

    Generate bundle with multiple fields:

    >>> @given(bundle=cxst.pointedvectors(
    ...     field_keys=("velocity", "acceleration"),
    ...     field_roles=(cxr.PhysVel, cxr.PhysAcc),
    ... ))
    ... def test_multi_field(bundle):
    ...     assert "velocity" in bundle.keys()
    ...     assert "acceleration" in bundle.keys()

    Generate batched bundles:

    >>> @given(bundle=cxst.pointedvectors(shape=(5,)))
    ... def test_batched(bundle):
    ...     assert bundle.shape == (5,)

    """
    # Draw representation if strategy
    base_chart = draw_if_strategy(draw, base_chart)

    # Generate base vector (always Point role - affine point)
    base = draw(
        vectors(
            chart=base_chart,
            role=cxr.point,
            dtype=dtype,
            shape=shape,
            elements=elements,
        )
    )

    # Determine field roles
    if field_roles is None:
        # Generate random tangent-like roles (exclude Point - only Vel, PhysAcc)
        tangent_roles: tuple[cxr.AbstractRole, ...] = (cxr.phys_vel, cxr.phys_acc)
        field_roles = tuple(draw(st.sampled_from(tangent_roles)) for _ in field_keys)
    else:
        field_roles = draw_if_strategy(draw, field_roles)

    # Validate: no Point in field_roles
    if any(role is cxr.point for role in field_roles):
        msg = "field_roles cannot contain Point role (base already has it)"
        raise ValueError(msg)

    # Validate length match
    if len(field_keys) != len(field_roles):
        msg = (
            f"field_keys has {len(field_keys)} items but field_roles has "
            f"{len(field_roles)}"
        )
        raise ValueError(msg)

    # Generate field vectors
    fields: dict[str, cx.Vector] = {}
    for key, role in zip(field_keys, field_roles, strict=True):
        # Use the same chart for all field vectors (simpler and more stable)
        field_vec = draw(
            vectors(
                chart=base_chart,
                role=role,
                dtype=dtype,
                shape=shape,
                elements=elements,
            )
        )
        fields[key] = field_vec

    return cx.PointedVector(base=base, **fields)


# ==============================================================================
# Type strategy registration

# Register type strategy for Hypothesis's st.from_type()
# Note: Pass the callable, not an invoked strategy
st.register_type_strategy(cx.Vector, lambda _: vectors())
