"""Register `quax` with `Point`, `Tangent`, and `Coordinate`."""

__all__: tuple[str, ...] = ()


from dataclasses import replace

from jaxtyping import Array, Bool
from typing import Any, cast

import jax
import jax.tree as jtu
import quax

import quaxed.numpy as jnp
import unxt.quantity as uq

import coordinax.charts as cxc
import coordinax.representations as cxr
from .bundle import Coordinate
from .custom_types import Shape
from .point import Point
from .tangent import Tangent

##############################################################################
# Primitives


@quax.register(jax.lax.broadcast_in_dim_p)
def broadcast_in_dim_p_absvec(operand: Point, /, *, shape: Shape, **kw: Any) -> Point:
    """Broadcast in a dimension.

    >>> import quaxed.numpy as jnp
    >>> import coordinax.main as cx

    >>> q = cx.Point.from_([1, 2, 3], "m")
    >>> print(q)
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    >>> print(jnp.broadcast_to(q, (1, 3)))
    <Point: chart=Cart3D (x, y, z) [m]
        [[1 2 3]]>

    """
    c_shape = shape[:-1]
    return replace(
        operand, data=jtu.map(lambda v: jnp.broadcast_to(v, c_shape), operand.data)
    )


@quax.register(jax.lax.convert_element_type_p)
def convert_element_type_p_absvec(operand: Point, /, **kw: Any) -> Point:
    """Convert the element type of a quantity.

    >>> import quaxed.lax as qlax
    >>> import coordinax.main as cx

    >>> vec = cx.Point.from_([1, 2, 3], "m")
    >>> vec["x"].dtype
    dtype('int64')

    >>> print(qlax.convert_element_type(vec, float))
    <Point: chart=Cart3D (x, y, z) [m]
        [1. 2. 3.]>

    """
    convert_p = quax.quaxify(jax.lax.convert_element_type_p.bind)
    data = jtu.map(lambda v: convert_p(v, **kw), operand.data)
    return replace(operand, data=data)


@quax.register(jax.lax.eq_p)
def eq_p_absvecs(lhs: Point, rhs: Point, /) -> Bool[Array, "..."]:
    """Element-wise equality of two vectors.

    See `Point.__eq__` for examples.

    """
    # Map the equality over the dict values, matching by key.
    comp_tree = jtu.map(
        jnp.equal,
        lhs.data,
        rhs.data,
        is_leaf=uq.is_any_quantity,
    )

    # Reduce the equality over the leaves.
    return jax.tree.reduce(jnp.logical_and, comp_tree)


@quax.register(jax.lax.neg_p)
def neg_p_absvec(operand: Point, /) -> Point:
    """Element-wise negation of a Point.

    For non-Cartesian charts, convert to Cartesian, negate, then convert back.
    This ensures correct negation semantics (e.g., negating a point in cylindrical
    coordinates should negate its Cartesian representation and convert back).
    """
    original_chart = operand.chart

    try:
        ambient_chart = original_chart.cartesian
    except cxc.NoGlobalCartesianChartError:
        ambient_chart = None

    # If there is no global Cartesian chart (or already Cartesian),
    # fall back to direct component-wise negation.
    if ambient_chart is None or ambient_chart == original_chart:
        return replace(
            operand,
            data=jtu.map(lambda v: -v, operand.data, is_leaf=uq.is_any_quantity),
        )

    # Convert to Cartesian, negate, convert back
    cart_vec = cast("Point", cxr.cconvert(operand, ambient_chart))
    neg_cart_vec = replace(
        cart_vec, data=jtu.map(lambda v: -v, cart_vec.data, is_leaf=uq.is_any_quantity)
    )
    neg_vec = cxr.cconvert(neg_cart_vec, original_chart)
    return cast("Point", neg_vec)


# ===============================================
# Add / Sub


@quax.register(jax.lax.add_p)
def add_p_absvecs(lhs: Point, rhs: Point, /, **kw: Any) -> Point:
    r"""Element-wise addition of two points.

    For non-Cartesian charts the operation converts both operands to the
    ambient Cartesian chart, adds there, and converts the result back
    to the ``lhs`` chart.  For Cartesian charts the addition is direct.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax.main as cx

    >>> v1 = cx.Point.from_([1, 2, 3], "m")
    >>> v2 = cx.Point.from_([4, 5, 6], "m")
    >>> print(v1 + v2)
    <Point: chart=Cart3D (x, y, z) [m]
        [5 7 9]>

    """
    return cast("Point", cxr.add(lhs, rhs))


@quax.register(jax.lax.sub_p)
def sub_p_absvecs(lhs: Point, rhs: Point, /, **kw: Any) -> Point:
    r"""Element-wise subtraction of two points.

    For non-Cartesian charts the operation converts both operands to the
    ambient Cartesian chart, subtracts there, and converts the result back
    to the ``lhs`` chart.  For Cartesian charts the subtraction is direct.

    The result keeps the ``lhs`` chart and representation.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax.main as cx

    Same-chart subtraction:

    >>> v1 = cx.Point.from_([4, 5, 6], "m")
    >>> v2 = cx.Point.from_([1, 2, 3], "m")
    >>> print(v1 - v2)
    <Point: chart=Cart3D (x, y, z) [m]
        [3 3 3]>

    """
    return cast("Point", cxr.subtract(lhs, rhs))


# ===============================================


@quax.register(jax.lax.mul_p)
def mul_p_absvecs(lhs: int | float | Array, rhs: Point, /, **kw: Any) -> Point:
    """Element-wise multiplication of a scalar and a point."""
    data = jtu.map(lambda v: jnp.multiply(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return replace(rhs, data=data)


@quax.register(jax.lax.mul_p)
def mul_p_vecs(lhs: Point, rhs: int | float | Array, /, **kw: Any) -> Point:
    """Element-wise multiplication of a point and a scalar."""
    data = jtu.map(lambda v: jnp.multiply(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return replace(lhs, data=data)


@quax.register(jax.lax.div_p)
def div_p_scalar_point(lhs: int | float | Array, rhs: Point, /, **kw: Any) -> Point:
    """Element-wise division of a scalar by a point."""
    data = jtu.map(lambda v: jnp.divide(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return replace(rhs, data=data)


@quax.register(jax.lax.div_p)
def div_p_vecs(lhs: Point, rhs: int | float | Array, /, **kw: Any) -> Point:
    """Element-wise division of a point by a scalar."""
    data = jtu.map(lambda v: jnp.divide(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return replace(lhs, data=data)


##############################################################################
# Tangent primitives
#
# Tangent vectors live in a genuine linear (vector) space T_p M.  All
# operations are component-wise — no Cartesian round-trip is needed or correct.
##############################################################################


@quax.register(jax.lax.broadcast_in_dim_p)
def broadcast_in_dim_p_tangent(
    operand: Tangent, /, *, shape: Shape, **kw: Any
) -> Tangent:
    """Broadcast a Tangent to a new shape.

    >>> import quaxed.numpy as jnp
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> v = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> result = jnp.broadcast_to(v, (2, 3))
    >>> result["x"].shape
    (2,)

    """
    c_shape = shape[:-1]
    return replace(
        operand,
        data=jtu.map(lambda v: jnp.broadcast_to(v, c_shape), operand.data),
    )


@quax.register(jax.lax.convert_element_type_p)
def convert_element_type_p_tangent(operand: Tangent, /, **kw: Any) -> Tangent:
    """Convert the element type of all components in a Tangent.

    >>> import quaxed.lax as qlax
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> v = cx.Tangent.from_(
    ...     {"x": u.Q(1, "m/s"), "y": u.Q(2, "m/s"), "z": u.Q(3, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> v["x"].dtype
    dtype('int64')

    >>> qlax.convert_element_type(v, float)["x"].dtype
    dtype('float64')

    """
    convert_p = quax.quaxify(jax.lax.convert_element_type_p.bind)
    data = jtu.map(lambda v: convert_p(v, **kw), operand.data)
    return replace(operand, data=data)


@quax.register(jax.lax.eq_p)
def eq_p_tangents(lhs: Tangent, rhs: Tangent, /) -> Bool[Array, "..."]:
    """Element-wise equality of two Tangent vectors."""
    comp_tree = jtu.map(
        jnp.equal,
        lhs.data,
        rhs.data,
        is_leaf=uq.is_any_quantity,
    )
    return jax.tree.reduce(jnp.logical_and, comp_tree)


@quax.register(jax.lax.neg_p)
def neg_p_tangent(operand: Tangent, /) -> Tangent:
    """Component-wise negation of a Tangent.

    Tangent spaces are vector spaces so negation is always component-wise,
    regardless of chart.

    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> v = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> (-v)["x"]
    Q(-1., 'm / s')

    """
    return replace(
        operand,
        data=jtu.map(lambda v: -v, operand.data, is_leaf=uq.is_any_quantity),
    )


@quax.register(jax.lax.add_p)
def add_p_tangents(lhs: Tangent, rhs: Tangent, /, **kw: Any) -> Tangent:
    """Component-wise addition of two Tangent vectors.

    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> v1 = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> v2 = cx.Tangent.from_(
    ...     {"x": u.Q(4.0, "m/s"), "y": u.Q(5.0, "m/s"), "z": u.Q(6.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> (v1 + v2)["x"]
    Q(5., 'm / s')

    """
    if lhs.rep != rhs.rep:
        msg = (
            f"Cannot add Tangent vectors with different representations: "
            f"{lhs.rep!r} vs {rhs.rep!r}."
        )
        raise ValueError(msg)
    return cast("Tangent", cxr.add(lhs, rhs))


@quax.register(jax.lax.sub_p)
def sub_p_tangents(lhs: Tangent, rhs: Tangent, /, **kw: Any) -> Tangent:
    """Component-wise subtraction of two Tangent vectors.

    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> v1 = cx.Tangent.from_(
    ...     {"x": u.Q(4.0, "m/s"), "y": u.Q(5.0, "m/s"), "z": u.Q(6.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> v2 = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(3.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> (v1 - v2)["x"]
    Q(3., 'm / s')

    """
    if lhs.rep != rhs.rep:
        msg = (
            f"Cannot subtract Tangent vectors with different representations: "
            f"{lhs.rep!r} vs {rhs.rep!r}."
        )
        raise ValueError(msg)
    return cast("Tangent", cxr.subtract(lhs, rhs))


@quax.register(jax.lax.mul_p)
def mul_p_scalar_tangent(
    lhs: int | float | Array, rhs: Tangent, /, **kw: Any
) -> Tangent:
    """Scalar * Tangent — scale all components."""
    data = jtu.map(lambda v: jnp.multiply(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return replace(rhs, data=data)


@quax.register(jax.lax.mul_p)
def mul_p_tangent_scalar(
    lhs: Tangent, rhs: int | float | Array, /, **kw: Any
) -> Tangent:
    """Tangent * scalar — scale all components."""
    data = jtu.map(lambda v: jnp.multiply(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return replace(lhs, data=data)


@quax.register(jax.lax.div_p)
def div_p_tangent_scalar(
    lhs: Tangent, rhs: int | float | Array, /, **kw: Any
) -> Tangent:
    """Tangent / scalar — divide all components."""
    data = jtu.map(lambda v: jnp.divide(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return replace(lhs, data=data)


##############################################################################
# Coordinate primitives
#
# Coordinate is a bundle (Point + named Tangents).  The JAX infrastructure
# primitives (broadcast_in_dim, convert_element_type) must propagate to both
# the base point and every fibre Tangent so that jit/vmap work correctly.
##############################################################################


@quax.register(jax.lax.broadcast_in_dim_p)
def broadcast_in_dim_p_coordinate(
    operand: Coordinate, /, *, shape: Shape, **kw: Any
) -> Coordinate:
    """Broadcast a Coordinate (base point + all fibre tangents) to a new shape.

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
    >>> vel = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> pv = cx.Coordinate(point=point, velocity=vel)
    >>> result = jnp.broadcast_to(pv, (2, 6))
    >>> result.point["x"].shape
    (2,)

    """
    # broadcast_in_dim is called by quax with the full batch+component shape;
    # the last axis is the component axis inside each vector — strip it.
    c_shape = shape[:-1]

    new_point = replace(
        operand.point,
        data=jtu.map(lambda v: jnp.broadcast_to(v, c_shape), operand.point.data),
    )
    new_fields = {
        name: replace(
            vec,
            data=jtu.map(lambda v: jnp.broadcast_to(v, c_shape), vec.data),
        )
        for name, vec in operand.items()
    }
    return Coordinate._create_unchecked(new_point, new_fields)


@quax.register(jax.lax.convert_element_type_p)
def convert_element_type_p_coordinate(operand: Coordinate, /, **kw: Any) -> Coordinate:
    """Convert element type for all components in a Coordinate bundle.

    >>> import quaxed.lax as qlax
    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr

    >>> point = cx.Point.from_([1, 0, 0], "m")
    >>> vel = cx.Tangent.from_(
    ...     {"x": u.Q(1, "m/s"), "y": u.Q(0, "m/s"), "z": u.Q(0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> pv = cx.Coordinate(point=point, velocity=vel)
    >>> pv.point["x"].dtype
    dtype('int64')

    >>> qlax.convert_element_type(pv, float).point["x"].value.dtype
    dtype('float64')

    """
    convert_p = quax.quaxify(jax.lax.convert_element_type_p.bind)

    new_point = replace(
        operand.point,
        data=jtu.map(lambda v: convert_p(v, **kw), operand.point.data),
    )
    new_fields = {
        name: replace(vec, data=jtu.map(lambda v: convert_p(v, **kw), vec.data))
        for name, vec in operand.items()
    }
    return Coordinate._create_unchecked(new_point, new_fields)


@quax.register(jax.lax.eq_p)
def eq_p_coordinates(lhs: Coordinate, rhs: Coordinate, /) -> Bool[Array, "..."]:
    """Element-wise equality of two Coordinate bundles.

    Returns True only if the base point and every named tangent fibre are
    component-wise equal.

    >>> import coordinax.main as cx
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import unxt as u

    >>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
    >>> vel = cx.Tangent.from_(
    ...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
    ...     cxc.cart3d, cxr.coord_vel,
    ... )
    >>> pv = cx.Coordinate(point=point, velocity=vel)
    >>> bool(pv == pv)
    True

    """
    # Ensure the same set of fibre names; if not, return False
    if set(lhs.keys()) != set(rhs.keys()):
        return jnp.asarray(False)  # noqa: FBT003
    # Check point equality
    point_eq = eq_p_absvecs(lhs.point, rhs.point)
    # Check equality for every fibre tangent, matched by name
    fibre_eqs = [eq_p_tangents(lhs[name], rhs[name]) for name in lhs]
    return jax.tree.reduce(jnp.logical_and, [point_eq, *fibre_eqs])
