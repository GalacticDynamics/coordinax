"""Representation of coordinates in different systems."""

__all__ = ["AbstractPos"]

from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from plum import convert
from quax import quaxify, register

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from unxt.quantity import AbstractQuantity

from .core import AbstractPos
from coordinax._src.vectors.api import vconvert
from coordinax._src.vectors.base import AttrFilter


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_qq(lhs: AbstractPos, rhs: AbstractPos, /) -> AbstractPos:
    # The base implementation is to convert to Cartesian and perform the
    # operation.  Cartesian coordinates do not have any branch cuts or
    # singularities or ranges that need to be handled, so this is a safe
    # default.
    cart_cls = lhs._cartesian_cls  # noqa: SLF001
    cart_cls = eqx.error_if(
        cart_cls,
        isinstance(lhs, cart_cls) and isinstance(rhs, cart_cls),
        "must register a Cartesian-specific dispatch for {cart_cls} addition",
    )
    clhs = lhs.vconvert(cart_cls)
    crhs = rhs.vconvert(cart_cls)
    return (clhs + crhs).vconvert(type(lhs))


# ------------------------------------------------
# Dot product
# TODO: see implementation in https://github.com/google/tree-math for how to do
# this more generally.


@register(jax.lax.dot_general_p)  # type: ignore[misc]
def _dot_general_pos(
    lhs: AbstractPos, rhs: AbstractPos, /, **kwargs: Any
) -> AbstractQuantity:
    """Dot product of two vectors.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.SphericalPos(
    ...     r=u.Quantity([1, 2, 3], "m"),
    ...     theta=u.Quantity([0, 0, 0], "rad"),
    ...     phi=u.Quantity([0, 0, 0], "rad"))

    >>> jnp.dot(vec, vec)
    Quantity['area'](Array([1., 4., 9.], dtype=float32), unit='m2')

    """
    return qlax.dot_general(
        lhs.vconvert(lhs._cartesian_cls),  # noqa: SLF001
        rhs.vconvert(lhs._cartesian_cls),  # noqa: SLF001
        **kwargs,
    )


# ------------------------------------------------


@register(jax.lax.div_p)  # type: ignore[misc]
def _div_pos_v(lhs: AbstractPos, rhs: ArrayLike) -> AbstractPos:
    """Divide a vector by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> jnp.divide(vec, 2).x
    Quantity['length'](Array(0.5, dtype=float32), unit='m')

    >>> (vec / 2).x
    Quantity['length'](Array(0.5, dtype=float32), unit='m')

    """
    return replace(
        lhs, **{k: jnp.divide(v, rhs) for k, v in field_items(AttrFilter, lhs)}
    )


# ------------------------------------------------


@register(jax.lax.eq_p)  # type: ignore[misc]
def _eq_pos_pos(lhs: AbstractPos, rhs: AbstractPos, /) -> ArrayLike:
    """Element-wise equality of two positions."""
    return lhs == rhs


# ------------------------------------------------


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_v_pos(lhs: ArrayLike, rhs: AbstractPos, /) -> AbstractPos:
    """Scale a position by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> print(jnp.multiply(2, vec))
    <CartesianPos3D (x[m], y[m], z[m])
        [2 4 6]>

    Most of the position classes have specific dispatches for this operation.
    So let's define a new class and try it out:

    >>> from typing import ClassVar
    >>> class MyCartesian(cx.vecs.AbstractPos):
    ...     x: u.Quantity
    ...     y: u.Quantity
    ...     z: u.Quantity
    ...     _dimensionality: ClassVar[int] = 3
    ...
    >>> MyCartesian._cartesian_cls = MyCartesian  # hack

    Add conversion to Quantity:

    >>> from plum import conversion_method
    >>> @conversion_method(MyCartesian, u.Quantity)
    ... def _to_quantity(x: MyCartesian, /) -> u.Quantity:
    ...     return jnp.stack((x.x, x.y, x.z), axis=-1)

    Add representation transformation

    >>> from plum import dispatch
    >>> @dispatch
    ... def vconvert(target: type[MyCartesian], current: MyCartesian, /) -> MyCartesian:
    ...     return current

    >>> vec = MyCartesian(x=u.Quantity([1], "m"),
    ...                   y=u.Quantity([2], "m"),
    ...                   z=u.Quantity([3], "m"))

    First hit the non-scalar error:

    >>> try: jnp.multiply(jnp.asarray([[1, 1, 1]]), vec)
    ... except Exception as e: print(e)
    must be a scalar, not <class 'jaxlib.xla_extension.ArrayImpl'>

    Then hit the Cartesian-specific dispatch error:

    >>> try: jnp.multiply(2, vec)
    ... except Exception as e: print(e)
    must register a Cartesian-specific dispatch

    Now a real example. For this we need to define the Cartesian-specific
    dispatches:

    >>> MyCartesian._cartesian_cls = cx.CartesianPos3D
    >>> @dispatch
    ... def vconvert(target: type[cx.CartesianPos3D],current: MyCartesian, /) -> cx.CartesianPos3D:
    ...     return cx.CartesianPos3D(x=current.x, y=current.y, z=current.z)
    >>> @dispatch
    ... def vconvert(target: type[MyCartesian], current: cx.CartesianPos3D, /) -> MyCartesian:
    ...     return MyCartesian(x=current.x, y=current.y, z=current.z)

    >>> print(jnp.multiply(2, vec))
    <MyCartesian (x[m], y[m], z[m])
        [[2 4 6]]>

    """  # noqa: E501
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )
    rhs = eqx.error_if(
        rhs,
        isinstance(rhs, rhs._cartesian_cls),  # noqa: SLF001
        "must register a Cartesian-specific dispatch",
    )

    rc = rhs.vconvert(rhs._cartesian_cls)  # noqa: SLF001
    nr = qlax.mul(lhs, rc)
    return nr.vconvert(type(rhs))


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_pos_v(lhs: AbstractPos, rhs: ArrayLike, /) -> AbstractPos:
    """Scale a position by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> print(jnp.multiply(vec, 2))
    <CartesianPos3D (x[m], y[m], z[m])
        [2 4 6]>

    """
    return qlax.mul(rhs, lhs)  # re-dispatch on the other side


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_pos_pos(lhs: AbstractPos, rhs: AbstractPos, /) -> u.Quantity:
    """Multiply two positions.

    This is required to take the dot product of two vectors.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D(
    ...     x=u.Quantity([1, 2, 3], "m"),
    ...     y=u.Quantity([4, 5, 6], "m"),
    ...     z=u.Quantity([7, 8, 9], "m"))

    >>> jnp.multiply(vec, vec)  # element-wise multiplication
    Quantity['area'](Array([[ 1, 16, 49],
                            [ 4, 25, 64],
                            [ 9, 36, 81]], dtype=int32), unit='m2')

    >>> jnp.linalg.vector_norm(vec, axis=-1)
    Quantity['length'](Array([ 8.124039,  9.643651, 11.224972], dtype=float32), unit='m')

    """  # noqa: E501
    lq: u.Quantity = convert(lhs.vconvert(lhs._cartesian_cls), u.Quantity)  # noqa: SLF001
    rq: u.Quantity = convert(rhs.vconvert(rhs._cartesian_cls), u.Quantity)  # noqa: SLF001
    return qlax.mul(lq, rq)  # re-dispatch to Quantities


# ------------------------------------------------


@register(jax.lax.neg_p)  # type: ignore[misc]
def _neg_pos(obj: AbstractPos, /) -> AbstractPos:
    """Negate the vector.

    The default implementation is to go through Cartesian coordinates.

    Examples
    --------
    >>> import coordinax as cx
    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> print(-vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [-1 -2 -3]>

    """
    cart = vconvert(obj._cartesian_cls, obj)  # noqa: SLF001
    negcart = jnp.negative(cart)
    return vconvert(type(obj), negcart)


# ------------------------------------------------


@register(jax.lax.reshape_p)  # type: ignore[misc]
def _reshape_pos(
    operand: AbstractPos, *, new_sizes: tuple[int, ...], **kwargs: Any
) -> AbstractPos:
    """Reshape the components of the vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import quaxed.numpy as jnp

    >>> vec = cx.CartesianPos3D(x=u.Quantity([1, 2, 3], "m"),
    ...                         y=u.Quantity([4, 5, 6], "m"),
    ...                         z=u.Quantity([7, 8, 9], "m"))
    >>> vec = jnp.reshape(vec, shape=(3, 1, 3))  # (n_components *shape)
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [[[[1 4 7]
           [2 5 8]
           [3 6 9]]]]>

    """
    # Adjust the sizes for the components
    new_sizes = (new_sizes[0] // len(operand.components), *new_sizes[1:])
    # TODO: check integer division
    # Reshape the components
    return replace(
        operand,
        **{
            k: quaxify(jax.lax.reshape_p.bind)(v, new_sizes=new_sizes, **kwargs)
            for k, v in field_items(operand)
        },
    )


# ------------------------------------------------


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_qq(lhs: AbstractPos, rhs: AbstractPos) -> AbstractPos:
    """Add another object to this vector."""
    # The base implementation is to convert to Cartesian and perform the
    # operation.  Cartesian coordinates do not have any branch cuts or
    # singularities or ranges that need to be handled, so this is a safe
    # default.
    return qlax.sub(
        lhs.vconvert(lhs._cartesian_cls),  # noqa: SLF001
        rhs.vconvert(lhs._cartesian_cls),  # noqa: SLF001
    ).vconvert(type(lhs))
