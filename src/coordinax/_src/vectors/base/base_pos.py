"""Representation of coordinates in different systems."""

__all__ = ["AbstractPos"]

from abc import abstractmethod
from dataclasses import replace
from functools import partial
from inspect import isabstract
from typing import Any, TypeVar

import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike, Shaped
from plum import convert
from quax import quaxify, register

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items

from .base import AbstractVector
from .flags import AttrFilter
from .mixins import AvalMixin
from .utils import ToUnitsOptions
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty
from coordinax._src.vectors.api import vconvert

PosT = TypeVar("PosT", bound="AbstractPos")

# TODO: figure out public API for this
POSITION_CLASSES: set[type["AbstractPos"]] = set()

_vec_matmul = quaxify(jax.numpy.vectorize(jax.numpy.matmul, signature="(N,N),(N)->(N)"))


class AbstractPos(AvalMixin, AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of coordinates in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered if it is not an abstract class.
        """
        # TODO: a more robust check using equinox.
        if isabstract(cls) or cls.__name__.startswith("Abstract"):
            return

        POSITION_CLASSES.add(cls)

    @classproperty
    @classmethod
    @abstractmethod
    def _cartesian_cls(cls) -> type["AbstractVector"]:
        """Return the corresponding Cartesian vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialPos._cartesian_cls
        <class 'coordinax...CartesianPos1D'>

        >>> cx.SphericalPos._cartesian_cls
        <class 'coordinax...CartesianPos3D'>

        """
        # TODO: something nicer than this for getting the corresponding class
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractVel"]:
        """Return the corresponding differential vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialPos.differential_cls.__name__
        'RadialVel'

        >>> cx.SphericalPos.differential_cls.__name__
        'SphericalVel'

        """
        raise NotImplementedError

    # ===============================================================
    # Unary operations

    __neg__ = jnp.negative

    # ===============================================================
    # Binary operations

    def __eq__(self: "AbstractPos", other: object) -> Any:
        """Element-wise equality of two positions.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx

        Showing the broadcasting, then element-wise comparison of two vectors:

        >>> vec1 = cx.CartesianPos3D.from_([[1, 2, 3], [1, 2, 4]], "m")
        >>> vec2 = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> jnp.equal(vec1, vec2)
        Array([ True, False], dtype=bool)

        Showing the change of representation:

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> vec1 = vec.vconvert(cx.SphericalPos)
        >>> vec2 = vec.vconvert(cx.vecs.MathSphericalPos)
        >>> jnp.equal(vec1, vec2)
        Array(True, dtype=bool)

        Quick run-through of each dimensionality:

        >>> vec1 = cx.vecs.CartesianPos1D.from_([1], "m")
        >>> vec2 = cx.vecs.RadialPos.from_([1], "m")
        >>> jnp.equal(vec1, vec2)
        Array(True, dtype=bool)

        >>> vec1 = cx.vecs.CartesianPos2D.from_([2, 0], "m")
        >>> vec2 = cx.vecs.PolarPos(r=u.Quantity(2, "m"), phi=u.Quantity(0, "rad"))
        >>> jnp.equal(vec1, vec2)
        Array(True, dtype=bool)

        """
        if not isinstance(other, AbstractPos):
            return NotImplemented

        rhs = other.vconvert(type(self))
        return super().__eq__(rhs)

    def __rmatmul__(self, other: Shaped[Array, "N N"]) -> Any:
        """Matrix multiplication with another object.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax as cx

        >>> q = cx.CartesianPos3D.from_([[1., 0, 0 ],
        ...                              [0 , 1, 0 ],
        ...                              [0 , 0, 1 ]], "kpc")[None]
        >>> q.shape
        (1, 3)

        >>> R_z = rot = jnp.asarray([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])

        >>> print(R_z @ q)
        <CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[[ 0.  1.  0.]
              [-1.  0.  0.]
              [ 0.  0.  1.]]]>

        """
        # TODO: figure out how to do this without converting back to arrays.
        cartvec = self.vconvert(self._cartesian_cls)
        q = convert(cartvec.uconvert(ToUnitsOptions.consistent), u.Quantity)
        newq = _vec_matmul(other, q)
        newvec = self._cartesian_cls.from_(newq)
        return newvec.vconvert(type(self))

    # ===============================================================
    # Convenience methods

    @partial(eqx.filter_jit, inline=True)
    def norm(self) -> BatchableLength:
        """Return the norm of the vector.

        Returns
        -------
        Quantity
            The norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> v = cx.vecs.CartesianPos1D.from_([-1], "km")
        >>> v.norm()
        Quantity['length'](Array(1., dtype=float32), unit='km')

        >>> v = cx.vecs.CartesianPos2D.from_([3, 4], "km")
        >>> v.norm()
        Quantity['length'](Array(5., dtype=float32), unit='km')

        >>> v = cx.vecs.PolarPos(r=u.Quantity(3, "km"), phi=u.Quantity(90, "deg"))
        >>> v.norm()
        Distance(Array(3, dtype=int32, ...), unit='km')

        >>> v = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> v.norm()
        Quantity['length'](Array(3.7416575, dtype=float32), unit='m')

        """
        return jnp.linalg.vector_norm(self, axis=-1)


# ===================================================================
# Register primitives


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
    lq = convert(lhs.vconvert(lhs._cartesian_cls), u.Quantity)  # noqa: SLF001
    rq = convert(rhs.vconvert(rhs._cartesian_cls), u.Quantity)  # noqa: SLF001
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
