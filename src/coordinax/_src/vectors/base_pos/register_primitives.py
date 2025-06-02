"""Representation of coordinates in different systems."""

__all__ = ["AbstractPos"]

from dataclasses import replace
from typing import Any, cast

import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike
from plum import convert
from quax import quaxify, register

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from unxt.quantity import BareQuantity

from .core import AbstractPos
from coordinax._src.vectors.api import vconvert
from coordinax._src.vectors.base import AttrFilter
from coordinax._src.vectors.base.register_primitives import eq_p_absvecs


@register(jax.lax.add_p)
def add_p_poss(lhs: AbstractPos, rhs: AbstractPos, /) -> AbstractPos:
    """Add another object to this vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.CartesianPos3D.from_(u.Quantity([1, 2, 3], "kpc"))
    >>> px = x.vconvert(cxv.ProlateSpheroidalPos, Delta=u.Quantity(2.0, "kpc"))

    >>> px2 = px + px
    >>> print(px2)
    <ProlateSpheroidalPos: (mu[kpc2], nu[kpc2], phi[rad])
     Delta=Quantity(2., unit='kpc')
        [57.495  2.505  1.107]>

    >>> print(px2.vconvert(cxv.CartesianPos3D))
    <CartesianPos3D: (x, y, z) [kpc]
        [2. 4. 6.]>

    """
    # The base implementation is to convert to Cartesian and perform the
    # operation.  Cartesian coordinates do not have any branch cuts,
    # singularities, ranges, or auxiliary data that need to be handled, so this
    # is a safe default. We restore aux data from the lhs.
    cart_cls = lhs.cartesian_type
    cart_cls = eqx.error_if(
        cart_cls,
        isinstance(lhs, cart_cls) and isinstance(rhs, cart_cls),
        f"must register a Cartesian-specific dispatch for {cart_cls} addition",
    )
    add = lhs.vconvert(cart_cls) + rhs.vconvert(cart_cls)
    return cast(AbstractPos, add.vconvert(type(lhs), **lhs._auxiliary_data))


# ------------------------------------------------
# Dot product
# TODO: see implementation in https://github.com/google/tree-math for how to do
# this more generally.


@register(jax.lax.dot_general_p)
def dot_p_general_poss(
    lhs: AbstractPos, rhs: AbstractPos, /, **kwargs: Any
) -> u.AbstractQuantity:
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
    Quantity(Array([1., 4., 9.], dtype=float32), unit='m2')

    """
    cart_cls = lhs.cartesian_type
    return qlax.dot_general(lhs.vconvert(cart_cls), rhs.vconvert(cart_cls), **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------


@register(jax.lax.div_p)
def div_p_pos_arraylike(lhs: AbstractPos, rhs: ArrayLike) -> AbstractPos:
    """Divide a vector by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> jnp.divide(vec, 2).x
    Quantity(Array(0.5, dtype=float32), unit='m')

    >>> (vec / 2).x
    Quantity(Array(0.5, dtype=float32), unit='m')

    """
    return replace(
        lhs, **{k: jnp.divide(v, rhs) for k, v in field_items(AttrFilter, lhs)}
    )


# ------------------------------------------------


@register(jax.lax.eq_p)
def eq_p_poss(lhs: AbstractPos, rhs: AbstractPos, /) -> Array:
    """Element-wise equality of two positions.

    The base AbstractVector-AbstractVector equality dispatch does not allow for
    conversion of the right-hand side to the left-hand side type since e.g.
    non-Cartesian velocities require a position to transform.

    """
    # Convert to the same type (left-hand side)
    rhs = cast(AbstractPos, rhs.vconvert(type(lhs)))
    # Check if the two positions are equal. This directly calls the appropriate
    # primitive, bypassing this dispatch to avoid infinite recursion.
    return eq_p_absvecs(lhs, rhs)


# ------------------------------------------------


@register(jax.lax.mul_p)
def mul_p_arraylike_pos(lhs: ArrayLike, rhs: AbstractPos, /) -> AbstractPos:
    """Scale a position by a scalar.

    Examples
    --------
    >>> from plum import dispatch
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> print(jnp.multiply(2, vec))
    <CartesianPos3D: (x, y, z) [m]
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
    >>> @dispatch
    ... def cartesian_vector_type(obj: type[MyCartesian]) -> type[MyCartesian]: return MyCartesian

    Add conversion to Quantity:

    >>> from plum import conversion_method
    >>> @conversion_method(MyCartesian, u.Quantity)
    ... def to_quantity(x: MyCartesian, /) -> u.Quantity:
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
    must be a scalar, not <class 'jaxlib...ArrayImpl'>

    Then hit the Cartesian-specific dispatch error:

    >>> try: jnp.multiply(2, vec)
    ... except Exception as e: print(e)
    must register a Cartesian-specific dispatch

    Now a real example. For this we need to define the Cartesian-specific
    dispatches:

    >>> @dispatch
    ... def cartesian_vector_type(obj: type[MyCartesian]) -> type[cx.CartesianPos3D]:
    ...      return cx.CartesianPos3D
    >>> @dispatch
    ... def vconvert(target: type[cx.CartesianPos3D],current: MyCartesian, /) -> cx.CartesianPos3D:
    ...     return cx.CartesianPos3D(x=current.x, y=current.y, z=current.z)
    >>> @dispatch
    ... def vconvert(target: type[MyCartesian], current: cx.CartesianPos3D, /) -> MyCartesian:
    ...     return MyCartesian(x=current.x, y=current.y, z=current.z)

    >>> print(jnp.multiply(2, vec))
    <MyCartesian: (x, y, z) [m]
        [[2 4 6]]>

    """  # noqa: E501
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )
    cart_cls = rhs.cartesian_type
    rhs = eqx.error_if(
        rhs, isinstance(rhs, cart_cls), "must register a Cartesian-specific dispatch"
    )

    rc = rhs.vconvert(cart_cls)
    nr = cast(AbstractPos, qlax.mul(lhs, rc))  # type: ignore[arg-type]
    return cast(AbstractPos, nr.vconvert(type(rhs)))


@register(jax.lax.mul_p)
def mul_p_pos_arraylike(lhs: AbstractPos, rhs: ArrayLike, /) -> AbstractPos:
    """Scale a position by a scalar.

    This just re-dispatches to the other side -- for example vec * 2 becomes 2 *
    vec -- so that there's only one complex dispatch required.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> print(jnp.multiply(vec, 2))
    <CartesianPos3D: (x, y, z) [m]
        [2 4 6]>

    """
    return cast(AbstractPos, qlax.mul(rhs, lhs))  # type: ignore[arg-type]  # re-dispatch on the other side


@register(jax.lax.mul_p)
def mul_p_poss(lhs: AbstractPos, rhs: AbstractPos, /) -> BareQuantity:
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
    BareQuantity(Array([[ 1, 16, 49],
                        [ 4, 25, 64],
                        [ 9, 36, 81]], dtype=int32), unit='m2')

    >>> jnp.linalg.vector_norm(vec, axis=-1)
    BareQuantity(Array([ 8.124039,  9.643651, 11.224972], dtype=float32), unit='m')

    """
    lq: BareQuantity = convert(lhs.vconvert(lhs.cartesian_type), BareQuantity)
    rq: BareQuantity = convert(rhs.vconvert(rhs.cartesian_type), BareQuantity)
    return qlax.mul(lq, rq)  # re-dispatch to Quantities


# ------------------------------------------------


@register(jax.lax.neg_p)
def neg_p_pos(obj: AbstractPos, /) -> AbstractPos:
    """Negate the vector.

    The default implementation is to go through Cartesian coordinates.

    Examples
    --------
    >>> import coordinax as cx
    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> print(-vec)
    <CartesianPos3D: (x, y, z) [m]
        [-1 -2 -3]>

    """
    cart = vconvert(obj.cartesian_type, obj)
    negcart = jnp.negative(cart)
    return vconvert(type(obj), negcart)


# ------------------------------------------------


@register(jax.lax.reshape_p)
def reshape_p_pos(
    operand: AbstractPos, /, *, new_sizes: tuple[int, ...], **kw: Any
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
    <CartesianPos3D: (x, y, z) [m]
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
            k: quaxify(jax.lax.reshape_p.bind)(v, new_sizes=new_sizes, **kw)
            for k, v in field_items(operand)
        },
    )


# ------------------------------------------------


@register(jax.lax.sub_p)
def sub_p_poss(lhs: AbstractPos, rhs: AbstractPos, /) -> AbstractPos:
    """Subtract another object from this vector.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.CartesianPos3D.from_(u.Quantity([1, 2, 3], "kpc"))
    >>> px = x.vconvert(cxv.ProlateSpheroidalPos, Delta=u.Quantity(2.0, "kpc"))

    >>> px2 = px - px
    >>> print(px2)
    <ProlateSpheroidalPos: (mu[kpc2], nu[kpc2], phi[rad])
      Delta=Quantity(2., unit='kpc')
        [4. 0. 0.]>

    >>> print(px2.vconvert(cxv.CartesianPos3D))
    <CartesianPos3D: (x, y, z) [kpc]
        [0. 0. 0.]>

    """
    # The base implementation is to convert to Cartesian and perform the
    # operation.  Cartesian coordinates do not have any branch cuts,
    # singularities, ranges, or auxiliary data that need to be handled, so this
    # is a safe default. We restore aux data from the lhs.
    cart_cls = lhs.cartesian_type
    diff = lhs.vconvert(cart_cls) - rhs.vconvert(cart_cls)
    return cast(AbstractPos, diff.vconvert(type(lhs), **lhs._auxiliary_data))
