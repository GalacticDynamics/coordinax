"""Vector."""

__all__: tuple[str, ...] = ()


from collections.abc import Callable, Mapping
from jaxtyping import Array, Bool
from typing import Any, NoReturn, cast

import equinox as eqx
import jax
import jax.tree as jtu
import plum
import quax

import quaxed.numpy as jnp
import unxt as u
import unxt.quantity as uq

import coordinax.charts as cxc
import coordinax.embeddings as cxe
import coordinax.roles as cxr
from .base import Vector
from coordinax._src.constants import LENGTH
from coordinax._src.custom_types import Shape

##############################################################################
# Primitives


@quax.register(jax.lax.broadcast_in_dim_p)
def broadcast_in_dim_p_absvec(
    operand: Vector, /, *, shape: Shape, **kwargs: Any
) -> Vector:
    """Broadcast in a dimension.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> q = cx.Vector.from_([1, 2, 3], "m")
    >>> print(q)
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 2 3]>

    >>> print(jnp.broadcast_to(q, (1, 3)))
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [[1 2 3]]>

    >>> p = cx.Vector.from_([1, 2, 3], "m/s")
    >>> print(jnp.broadcast_to(p, (1, 3)))
    <Vector: chart=Cart3D, role=PhysVel (x, y, z) [m / s]
        [[1 2 3]]>

    >>> a = cx.Vector.from_([1, 2, 3], "m/s2")
    >>> print(jnp.broadcast_to(a, (1, 3)))
    <Vector: chart=Cart3D, role=PhysAcc (x, y, z) [m / s2]
        [[1 2 3]]>

    """
    c_shape = shape[:-1]
    return Vector(
        jtu.map(lambda v: jnp.broadcast_to(v, c_shape), operand.data),
        chart=operand.chart,
        role=operand.role,
    )


@quax.register(jax.lax.convert_element_type_p)
def convert_element_type_p_absvec(operand: Vector, /, **kwargs: Any) -> Vector:
    """Convert the element type of a quantity.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import coordinax as cx

    >>> vec = cx.Vector.from_([1, 2, 3], "m")
    >>> vec["x"].dtype
    dtype('int64')

    >>> print(qlax.convert_element_type(vec, float))
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1. 2. 3.]>

    """
    convert_p = quax.quaxify(jax.lax.convert_element_type_p.bind)
    data = jtu.map(lambda v: convert_p(v, **kwargs), operand.data)
    return Vector(data, chart=operand.chart, role=operand.role)


@quax.register(jax.lax.eq_p)
def eq_p_absvecs(lhs: Vector, rhs: Vector, /) -> Bool[Array, "..."]:
    """Element-wise equality of two vectors.

    See `Vector.__eq__` for examples.

    """
    # Map the equality over the leaves, which are Quantities.
    comp_tree = jtu.map(
        jnp.equal,
        jtu.leaves(lhs.data, is_leaf=uq.is_any_quantity),
        jtu.leaves(rhs.data, is_leaf=uq.is_any_quantity),
        is_leaf=uq.is_any_quantity,
    )

    # Reduce the equality over the leaves.
    return jax.tree.reduce(jnp.logical_and, comp_tree)


@quax.register(jax.lax.neg_p)
def neg_p_absvec(operand: Vector, /) -> Vector:
    """Element-wise negation of a Vector."""
    return Vector(
        jtu.map(lambda v: -v, operand.data, is_leaf=uq.is_any_quantity),
        chart=operand.chart,
        role=operand.role,
    )


# ===============================================
# Add


# -------------------------------------------------------------------
# Internal helpers for role-aware arithmetic


def _require_at_point(
    opname: str,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    at: Vector | None,
) -> Vector:
    """Validate and return a base point for non-Euclidean operations."""
    if at is None:
        msg = (
            f"{opname} on non-Euclidean representation {type(chart).__name__} "
            "requires an `at` base-point parameter. "
            f"Use Vector.{opname.lower()}(other, at=base_point)."
        )
        raise ValueError(msg)
    at = eqx.error_if(
        at, not isinstance(at.role, cxr.Point), "`at` must be a Point vector."
    )
    return at  # noqa: RET504


def _point_in_chart(vec: Vector, chart: cxc.AbstractChart) -> Vector:  # type: ignore[type-arg]
    """Express a Point vector in the requested chart."""
    if vec.chart == chart:
        return vec
    return cast("Vector", vec.vconvert(chart))


def _at_in_chart(at: Vector, chart: cxc.AbstractChart) -> Vector:  # type: ignore[type-arg]
    """Express a base Point in the requested chart."""
    return _point_in_chart(at, chart)


def _tangent_in_chart(
    vec: Vector,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    at_in_chart: Vector,
) -> Vector:
    """Express a physical tangent vector in the requested chart at a base point."""
    if vec.chart == chart:
        return vec
    return cast("Vector", vec.vconvert(chart, at_in_chart))


def _leaf_binop(
    op: Callable,  # type: ignore[type-arg]
    a: Mapping[str, Any],
    b: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a binary op to matching leaves in two component dicts."""
    return cast(
        "dict[str, Any]",
        jtu.map(op, a, b, is_leaf=uq.is_any_quantity),
    )


def _embedded_point_tangent_update(
    op: Callable,  # type: ignore[type-arg]
    *,
    point: Vector,
    tangent: Vector,
    at: Vector,
) -> dict[str, Any]:
    """Update embedded-manifold point by tangent vector w/ ambient arithmetic."""
    chart = point.chart
    at_in_chart = _at_in_chart(at, chart)
    tangent_in_chart = _tangent_in_chart(tangent, chart, at_in_chart)

    p_amb = cxe.embed_point(chart, point.data)
    v_amb = cxe.embed_tangent(chart, tangent_in_chart.data, at=at_in_chart.data)
    p_out_amb = jtu.map(op, p_amb, v_amb, is_leaf=uq.is_any_quantity)
    return cxe.project_point(chart, p_out_amb)


def _embedded_tangent_binop(
    op: Callable,  # type: ignore[type-arg]
    *,
    lhs: Vector,
    rhs: Vector,
    at: Vector,
) -> dict[str, Any]:
    """Combine two embedded-manifold tangents via ambient op then project."""
    chart = lhs.chart
    at_in_chart = _at_in_chart(at, chart)
    rhs_in_chart = _tangent_in_chart(rhs, chart, at_in_chart)

    v_lhs_amb = cxe.embed_tangent(chart, lhs.data, at=at_in_chart.data)
    v_rhs_amb = cxe.embed_tangent(chart, rhs_in_chart.data, at=at_in_chart.data)
    v_out_amb = jtu.map(op, v_lhs_amb, v_rhs_amb, is_leaf=uq.is_any_quantity)
    return cxe.project_tangent(chart, v_out_amb, at=at_in_chart.data)


def _embedded_point_point_to_pos(
    op: Callable,  # type: ignore[type-arg]
    *,
    lhs: Vector,
    rhs: Vector,
    at: Vector,
) -> dict[str, Any]:
    """Compute Point∘Point -> PhysDisp.

    On an embedded manifold via ambient chord then tangent projection.

    """
    chart = lhs.chart
    at_in_chart = _at_in_chart(at, chart)
    rhs_in_chart = _point_in_chart(rhs, chart)

    p_lhs_amb = cxe.embed_point(chart, lhs.data)
    p_rhs_amb = cxe.embed_point(chart, rhs_in_chart.data)
    v_amb = jtu.map(op, p_lhs_amb, p_rhs_amb, is_leaf=uq.is_any_quantity)
    return cxe.project_tangent(chart, v_amb, at=at_in_chart.data)


@quax.register(jax.lax.add_p)
def add_p_absvecs(lhs: Vector, rhs: Vector, /) -> Vector:
    r"""Element-wise addition of two vectors with role semantics.

    This primitive implements vector addition following affine geometry:

    - ``PhysDisp + PhysDisp`` → ``PhysDisp``
    - ``Point + PhysDisp`` → ``Point``
    - ``Point + Point`` → **TypeError**
    - ``PhysDisp + Point`` → **TypeError**

    For non-Euclidean representations, use `Vector.add(other, at=base_point)`.

    """
    return add(lhs.role, rhs.role, lhs, rhs, at=None)


@quax.register(jax.lax.add_p)
def add_p_vec_qty(lhs: Vector, rhs: u.AbstractQuantity, /) -> Vector:
    r"""Element-wise addition of Vector + Quantity.

    Desugars to Vector + Vector by interpreting the Quantity:

    - If dimension = length: interpret as Pos (physical displacement)
    - Otherwise: interpret as Point (affine point, default)

    Then delegates to Vector + Vector semantics.

    """
    if u.dimension_of(rhs) == LENGTH:
        rhs_vec = Vector.from_(rhs, cxr.phys_disp)
    else:
        rhs_vec = Vector.from_(rhs)
    return add(lhs.role, rhs_vec.role, lhs, rhs_vec, at=None)


@quax.register(jax.lax.add_p)
def add_p_qty_vec(lhs: u.AbstractQuantity, rhs: Vector, /) -> Vector:
    r"""Element-wise addition of Quantity + Vector.

    Desugars to Vector + Vector by interpreting the Quantity:

    - If dimension = length: interpret as Pos (physical displacement)
    - Otherwise: interpret as Point (affine point, default)

    Then delegates to Vector + Vector semantics.

    """
    if u.dimension_of(lhs) == LENGTH:
        lhs_vec = Vector.from_(lhs, cxr.phys_disp)
    else:
        lhs_vec = Vector.from_(lhs)  # defaults to Point
    return add(lhs_vec.role, rhs.role, lhs_vec, rhs, at=None)


@plum.dispatch.abstract
def add(
    role_lhs: cxr.AbstractRole,
    role_rhs: cxr.AbstractRole,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Add two vectors."""
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
def add(
    role_lhs: cxr.Point,
    role_rhs: cxr.PhysDisp,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Point + Pos -> Point."""
    # Embedded / non-Euclidean requires a base point.
    if not lhs.chart.is_euclidean:
        at_vec = _require_at_point("Addition", lhs.chart, at)

        if not isinstance(lhs.chart, cxe.EmbeddedManifold):
            msg = (
                f"Addition on intrinsic non-Euclidean "
                f"manifold {type(lhs.chart).__name__} "
                "is not yet implemented. Provide an embedding (EmbeddedManifold) "
                "or implement intrinsic exponential-map semantics."
            )
            raise NotImplementedError(msg)

        out = _embedded_point_tangent_update(jnp.add, point=lhs, tangent=rhs, at=at_vec)
        return Vector(out, lhs.chart, cxr.point)

    # Euclidean: same chart fast path.
    if lhs.chart == rhs.chart:
        return Vector(_leaf_binop(jnp.add, lhs.data, rhs.data), lhs.chart, cxr.point)

    # Euclidean: convert rhs tangent to lhs chart at the base point lhs.
    rhs_in_lhs = _tangent_in_chart(rhs, lhs.chart, lhs)
    return Vector(_leaf_binop(jnp.add, lhs.data, rhs_in_lhs.data), lhs.chart, cxr.point)


@plum.dispatch
def add(
    role_lhs: cxr.PhysDisp,
    role_rhs: cxr.Point,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> NoReturn:
    """Pos + Point is not allowed."""
    msg = (
        "Cannot add Pos + Point. Use Point + Pos instead, or convert both to PhysDisp."
    )
    raise TypeError(msg)


@plum.dispatch
def add(
    role_lhs: cxr.PhysDisp,
    role_rhs: cxr.PhysDisp,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    if not lhs.chart.is_euclidean:
        at_vec = _require_at_point("Addition", lhs.chart, at)

        if isinstance(lhs.chart, cxe.EmbeddedManifold):
            out = _embedded_tangent_binop(jnp.add, lhs=lhs, rhs=rhs, at=at_vec)
            return Vector(out, lhs.chart, cxr.phys_disp)

        msg = (
            f"Addition on intrinsic non-Euclidean manifold {type(lhs.chart).__name__} "
            "is not yet implemented. Provide an embedding (EmbeddedManifold) "
            "or implement intrinsic parallel transport / exp-map semantics."
        )
        raise NotImplementedError(msg)

    # Euclidean: same chart fast path.
    if lhs.chart == rhs.chart:
        return Vector(
            _leaf_binop(jnp.add, lhs.data, rhs.data), lhs.chart, cxr.phys_disp
        )

    # Euclidean cross-chart requires a base point because the physical basis may
    # depend on position.
    at_vec = _require_at_point("Addition", lhs.chart, at)
    at_in_lhs = _at_in_chart(at_vec, lhs.chart)
    rhs_in_lhs = _tangent_in_chart(rhs, lhs.chart, at_in_lhs)
    return Vector(
        _leaf_binop(jnp.add, lhs.data, rhs_in_lhs.data), lhs.chart, cxr.phys_disp
    )


@plum.dispatch
def add(
    role_lhs: cxr.PhysVel,
    role_rhs: cxr.PhysVel,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Add PhysVel + PhysVel -> PhysVel.

    If both velocities are in the same representation, operates directly.
    Otherwise, converts to Cartesian for the operation. If `at` is provided,
    converts the result back to lhs's representation.
    """
    # Same representation - direct component-wise addition
    if lhs.chart == rhs.chart:
        data = jtu.map(jnp.add, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
        return Vector(data, lhs.chart, lhs.role)

    # Different representations - convert to Cartesian
    lhs_cart = (
        lhs.vconvert(lhs.chart.cartesian) if hasattr(lhs.chart, "cartesian") else lhs
    )
    rhs_cart = (
        rhs.vconvert(rhs.chart.cartesian) if hasattr(rhs.chart, "cartesian") else rhs
    )
    lhs_cart = eqx.error_if(
        lhs_cart,
        lhs_cart.chart != rhs_cart.chart,
        "Cannot add vectors in different non-convertible representations.",
    )
    data = jtu.map(jnp.add, lhs_cart.data, rhs_cart.data, is_leaf=uq.is_any_quantity)
    result_cart = Vector(data, lhs_cart.chart, lhs.role)

    # Convert back to original representation if base point provided
    if at is not None and lhs.chart != lhs_cart.chart:
        return cast("Vector", result_cart.vconvert(lhs.chart, at))

    return result_cart


@plum.dispatch
def add(
    role_lhs: cxr.PhysAcc,
    role_rhs: cxr.PhysAcc,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Add PhysAcc + PhysAcc -> PhysAcc.

    If both accelerations are in the same representation, operates directly.
    Otherwise, converts to Cartesian for the operation. If `at` is provided,
    converts the result back to lhs's representation.
    """
    # Same representation - direct component-wise addition
    if lhs.chart == rhs.chart:
        data = jtu.map(jnp.add, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
        return Vector(data, lhs.chart, lhs.role)

    # Different representations - convert to Cartesian
    lhs_cart = (
        lhs.vconvert(lhs.chart.cartesian) if hasattr(lhs.chart, "cartesian") else lhs
    )
    rhs_cart = (
        rhs.vconvert(rhs.chart.cartesian) if hasattr(rhs.chart, "cartesian") else rhs
    )
    lhs_cart = eqx.error_if(
        lhs_cart,
        lhs_cart.chart != rhs_cart.chart,
        "Cannot add vectors in different non-convertible representations.",
    )
    data = jtu.map(jnp.add, lhs_cart.data, rhs_cart.data, is_leaf=uq.is_any_quantity)
    result_cart = Vector(data, lhs_cart.chart, lhs.role)

    # Convert back to original representation if base point provided
    if at is not None and lhs.chart != lhs_cart.chart:
        return cast("Vector", result_cart.vconvert(lhs.chart, at))

    return result_cart


# ===============================================
# Sub


@quax.register(jax.lax.sub_p)
def sub_p_absvecs(lhs: Vector, rhs: Vector, /) -> Vector:
    r"""Element-wise subtraction of two vectors with role semantics.

    Normative behavior:
    - If `lhs.chart == rhs.chart`, subtraction is performed component-wise and
      the result role is determined by the role-dispatch table (`sub(...)`).
    - If `lhs.chart != rhs.chart`, this function does **not** pick an implicit
      intermediate chart. Instead, it delegates to `sub(...)`, which will
      either:

      * perform a role-correct conversion (for `Point - Point`, via
        `point_transform` to a common chart), or
      * raise an error indicating that `at=` is required (for tangent-like
        roles or results).

    """
    # Fast path: same chart and leaf-wise subtraction. Role semantics are still
    # enforced by delegating to `sub(...)` when roles are not trivially closed.
    # If roles are the same physical-tangent role, we can subtract leaves
    # directly and keep the role.
    if (
        (lhs.chart == rhs.chart)
        and isinstance(lhs.role, r.AbstractPhysRole)
        and type(lhs.role) is type(rhs.role)
    ):
        data = jtu.map(jnp.subtract, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
        return Vector(data, lhs.chart, lhs.role)

    # Delegate to the role-aware dispatcher. This ensures:
    # - `Point - Point -> Pos` uses the affine semantics,
    # - cross-chart tangent operations require `at=`,
    # - invalid role combinations raise.
    return sub(lhs.role, rhs.role, lhs, rhs, at=None)


@plum.dispatch.abstract
def sub(
    role_lhs: cxr.AbstractRole,
    role_rhs: cxr.AbstractRole,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Subtract two vectors."""
    raise NotImplementedError  # pragma: no cover


@plum.dispatch
def sub(
    role_lhs: cxr.Point,
    role_rhs: cxr.Point,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Point - Point -> Pos (affine difference)."""
    # Embedded / non-Euclidean requires a base point for the resulting PhysDisp.
    if not lhs.chart.is_euclidean:
        at_vec = _require_at_point("Subtraction", lhs.chart, at)

        if isinstance(lhs.chart, cxe.EmbeddedManifold):
            out = _embedded_point_point_to_pos(
                jnp.subtract, lhs=lhs, rhs=rhs, at=at_vec
            )
            return Vector(out, lhs.chart, cxr.phys_disp)

        msg = (
            "Subtraction on intrinsic non-Euclidean "
            f"manifold {type(lhs.chart).__name__} "
            "is not yet implemented. Provide an embedding (EmbeddedManifold) "
            "or implement intrinsic log-map / parallel transport semantics."
        )
        raise NotImplementedError(msg)

    # Euclidean: same chart fast path.
    if lhs.chart == rhs.chart:
        return Vector(
            _leaf_binop(jnp.subtract, lhs.data, rhs.data), lhs.chart, cxr.phys_disp
        )

    # Euclidean: convert rhs Point to lhs chart then subtract.
    rhs_in_lhs = _point_in_chart(rhs, lhs.chart)
    return Vector(
        _leaf_binop(jnp.subtract, lhs.data, rhs_in_lhs.data), lhs.chart, cxr.phys_disp
    )


@plum.dispatch
def sub(
    role_lhs: cxr.Point,
    role_rhs: cxr.PhysDisp,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    if not lhs.chart.is_euclidean:
        at_vec = _require_at_point("Subtraction", lhs.chart, at)

        if isinstance(lhs.chart, cxe.EmbeddedManifold):
            out = _embedded_point_tangent_update(
                jnp.subtract, point=lhs, tangent=rhs, at=at_vec
            )
            return Vector(out, lhs.chart, cxr.point)

        msg = (
            "Subtraction on intrinsic non-Euclidean "
            f"manifold {type(lhs.chart).__name__} is not yet implemented. "
            "Provide an embedding (EmbeddedManifold) or implement "
            "intrinsic exponential/log-map semantics."
        )
        raise NotImplementedError(msg)

    if lhs.chart == rhs.chart:
        return Vector(
            _leaf_binop(jnp.subtract, lhs.data, rhs.data), lhs.chart, cxr.point
        )

    rhs_in_lhs = _tangent_in_chart(rhs, lhs.chart, lhs)
    return Vector(
        _leaf_binop(jnp.subtract, lhs.data, rhs_in_lhs.data), lhs.chart, cxr.point
    )


@plum.dispatch
def sub(
    role_lhs: cxr.PhysDisp,
    role_rhs: cxr.Point,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> NoReturn:
    """Pos - Point is not allowed."""
    msg = (
        "Cannot subtract Point from PhysDisp. "
        "Use Point - Point - PhysDisp, or Point - Pos."
    )
    raise TypeError(msg)


@plum.dispatch
def sub(
    role_lhs: cxr.PhysDisp,
    role_rhs: cxr.PhysDisp,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    if not lhs.chart.is_euclidean:
        at_vec = _require_at_point("Subtraction", lhs.chart, at)

        if isinstance(lhs.chart, cxe.EmbeddedManifold):
            out = _embedded_tangent_binop(jnp.subtract, lhs=lhs, rhs=rhs, at=at_vec)
            return Vector(out, lhs.chart, cxr.phys_disp)

        msg = (
            f"Subtraction on intrinsic non-Euclidean "
            f"manifold {type(lhs.chart).__name__} is not yet implemented. "
            "Provide an embedding (EmbeddedManifold) or implement "
            "intrinsic parallel transport / log-map semantics."
        )
        raise NotImplementedError(msg)

    if lhs.chart == rhs.chart:
        return Vector(
            _leaf_binop(jnp.subtract, lhs.data, rhs.data), lhs.chart, cxr.phys_disp
        )

    at_vec = _require_at_point("Subtraction", lhs.chart, at)
    at_in_lhs = _at_in_chart(at_vec, lhs.chart)
    rhs_in_lhs = _tangent_in_chart(rhs, lhs.chart, at_in_lhs)
    return Vector(
        _leaf_binop(jnp.subtract, lhs.data, rhs_in_lhs.data), lhs.chart, cxr.phys_disp
    )


@plum.dispatch
def sub(
    role_lhs: cxr.PhysVel,
    role_rhs: cxr.PhysVel,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Subtract PhysVel - PhysVel -> PhysVel.

    If both velocities are in the same representation, operates directly.
    Otherwise, converts to Cartesian for the operation. If `at` is provided,
    converts the result back to lhs's representation.
    """
    # Same representation - direct component-wise subtraction
    if lhs.chart == rhs.chart:
        data = jtu.map(jnp.subtract, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
        return Vector(data, lhs.chart, lhs.role)

    # Different representations - convert to Cartesian
    lhs_cart = (
        lhs.vconvert(lhs.chart.cartesian) if hasattr(lhs.chart, "cartesian") else lhs
    )
    rhs_cart = (
        rhs.vconvert(rhs.chart.cartesian) if hasattr(rhs.chart, "cartesian") else rhs
    )
    lhs_cart = eqx.error_if(
        lhs_cart,
        lhs_cart.chart != rhs_cart.chart,
        "Cannot subtract vectors in different non-convertible representations.",
    )
    data = jtu.map(
        jnp.subtract, lhs_cart.data, rhs_cart.data, is_leaf=uq.is_any_quantity
    )
    result_cart = Vector(data, lhs_cart.chart, lhs.role)

    # Convert back to original representation if base point provided
    if at is not None and lhs.chart != lhs_cart.chart:
        return cast("Vector", result_cart.vconvert(lhs.chart, at))

    return result_cart


@plum.dispatch
def sub(
    role_lhs: cxr.PhysAcc,
    role_rhs: cxr.PhysAcc,
    lhs: Vector,
    rhs: Vector,
    /,
    *,
    at: Vector | None = None,
) -> Vector:
    """Subtract PhysAcc - PhysAcc -> PhysAcc.

    If both accelerations are in the same representation, operates directly.
    Otherwise, converts to Cartesian for the operation. If `at` is provided,
    converts the result back to lhs's representation.
    """
    # Same representation - direct component-wise subtraction
    if lhs.chart == rhs.chart:
        data = jtu.map(jnp.subtract, lhs.data, rhs.data, is_leaf=uq.is_any_quantity)
        return Vector(data, lhs.chart, lhs.role)

    # Different representations - convert to Cartesian
    lhs_cart = (
        lhs.vconvert(lhs.chart.cartesian) if hasattr(lhs.chart, "cartesian") else lhs
    )
    rhs_cart = (
        rhs.vconvert(rhs.chart.cartesian) if hasattr(rhs.chart, "cartesian") else rhs
    )
    lhs_cart = eqx.error_if(
        lhs_cart,
        lhs_cart.chart != rhs_cart.chart,
        "Cannot subtract vectors in different non-convertible representations.",
    )
    data = jtu.map(
        jnp.subtract, lhs_cart.data, rhs_cart.data, is_leaf=uq.is_any_quantity
    )
    result_cart = Vector(data, lhs_cart.chart, lhs.role)

    # Convert back to original representation if base point provided
    if at is not None and lhs.chart != lhs_cart.chart:
        return cast("Vector", result_cart.vconvert(lhs.chart, at))

    return result_cart


# ===============================================


@quax.register(jax.lax.mul_p)
def mul_p_absvecs(lhs: int | float | Array, rhs: Vector, /) -> Vector:
    """Element-wise multiplication of a scalar and a vector."""
    data = jtu.map(lambda v: jnp.multiply(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, chart=rhs.chart, role=rhs.role)


@quax.register(jax.lax.mul_p)
def mul_p_vecs(lhs: Vector, rhs: int | float | Array, /) -> Vector:
    """Element-wise multiplication of a vector and a scalar."""
    data = jtu.map(lambda v: jnp.multiply(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, chart=lhs.chart, role=lhs.role)


@quax.register(jax.lax.div_p)
def div_p_absvecs(lhs: int | float | Array, rhs: Vector, /) -> Vector:
    """Element-wise division of a scalar by a vector."""
    data = jtu.map(lambda v: jnp.divide(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, chart=rhs.chart, role=rhs.role)


@quax.register(jax.lax.div_p)
def div_p_vecs(lhs: Vector, rhs: int | float | Array, /) -> Vector:
    """Element-wise division of a vector by a scalar."""
    data = jtu.map(lambda v: jnp.divide(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return Vector(data, chart=lhs.chart, role=lhs.role)
