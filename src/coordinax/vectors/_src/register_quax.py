"""Register `quax` with `Point`."""

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
from .point import Point
from coordinax.internal.custom_types import Shape

##############################################################################
# Primitives


@quax.register(jax.lax.broadcast_in_dim_p)
def broadcast_in_dim_p_absvec(
    operand: Point, /, *, shape: Shape, **kwargs: Any
) -> Point:
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
def convert_element_type_p_absvec(operand: Point, /, **kwargs: Any) -> Point:
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
    data = jtu.map(lambda v: convert_p(v, **kwargs), operand.data)
    return replace(operand, data=data)


@quax.register(jax.lax.eq_p)
def eq_p_absvecs(lhs: Point, rhs: Point, /) -> Bool[Array, "..."]:
    """Element-wise equality of two vectors.

    See `Point.__eq__` for examples.

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
def add_p_absvecs(lhs: Point, rhs: Point, /) -> Point:
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
def sub_p_absvecs(lhs: Point, rhs: Point, /) -> Point:
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
def mul_p_absvecs(lhs: int | float | Array, rhs: Point, /) -> Point:
    """Element-wise multiplication of a scalar and a point."""
    data = jtu.map(lambda v: jnp.multiply(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return replace(rhs, data=data)


@quax.register(jax.lax.mul_p)
def mul_p_vecs(lhs: Point, rhs: int | float | Array, /) -> Point:
    """Element-wise multiplication of a point and a scalar."""
    data = jtu.map(lambda v: jnp.multiply(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return replace(lhs, data=data)


@quax.register(jax.lax.div_p)
def div_p_absvecs(lhs: int | float | Array, rhs: Point, /) -> Point:
    """Element-wise division of a scalar by a point."""
    data = jtu.map(lambda v: jnp.divide(lhs, v), rhs.data, is_leaf=uq.is_any_quantity)
    return replace(rhs, data=data)


@quax.register(jax.lax.div_p)
def div_p_vecs(lhs: Point, rhs: int | float | Array, /) -> Point:
    """Element-wise division of a point by a scalar."""
    data = jtu.map(lambda v: jnp.divide(v, rhs), lhs.data, is_leaf=uq.is_any_quantity)
    return replace(lhs, data=data)
