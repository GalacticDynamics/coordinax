"""Representation of coordinates in different systems."""

__all__ = ["AbstractPos"]

from abc import abstractmethod
from dataclasses import replace
from functools import partial
from inspect import isabstract
from typing import Any, TypeVar
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from plum import convert
from quax import quaxify, register

import quaxed.lax as qlax
import quaxed.numpy as jnp
from dataclassish import field_items
from unxt import Quantity

from .base import AbstractVector
from .mixins import AvalMixin
from coordinax._src import typing as ct
from coordinax._src.funcs import represent_as
from coordinax._src.utils import classproperty

PosT = TypeVar("PosT", bound="AbstractPos")

# TODO: figure out public API for this
POSITION_CLASSES: set[type["AbstractPos"]] = set()


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

        >>> cx.RadialPos._cartesian_cls
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

        >>> cx.RadialPos.differential_cls.__name__
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
        >>> import coordinax as cx

        Showing the broadcasting, then element-wise comparison of two vectors:

        >>> vec1 = cx.CartesianPos3D.from_([[1, 2, 3], [1, 2, 4]], "m")
        >>> vec2 = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> jnp.equal(vec1, vec2)
        Array([ True, False], dtype=bool)

        Showing the change of representation:

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> vec1 = vec.represent_as(cx.SphericalPos)
        >>> vec2 = vec.represent_as(cx.MathSphericalPos)
        >>> jnp.equal(vec1, vec2)
        Array(True, dtype=bool)

        Quick run-through of each dimensionality:

        >>> vec1 = cx.CartesianPos1D.from_([1], "m")
        >>> vec2 = cx.RadialPos.from_([1], "m")
        >>> jnp.equal(vec1, vec2)
        Array(True, dtype=bool)

        >>> vec1 = cx.CartesianPos2D.from_([2, 0], "m")
        >>> vec2 = cx.PolarPos(r=Quantity(2, "m"), phi=Quantity(0, "rad"))
        >>> jnp.equal(vec1, vec2)
        Array(True, dtype=bool)

        """
        if not isinstance(other, AbstractPos):
            return NotImplemented

        rhs = other.represent_as(type(self))
        return super().__eq__(rhs)

    # ===============================================================
    # Convenience methods

    @override
    def represent_as(self, target: type[PosT], /, *args: Any, **kwargs: Any) -> PosT:
        """Represent the vector as another type.

        This just forwards to `coordinax.represent_as`.

        Parameters
        ----------
        target : type[`coordinax.AbstractPos`]
            The type to represent the vector as.
        *args, **kwargs : Any
            Extra arguments. These are passed to `coordinax.represent_as` and
            might be used, depending on the dispatched method.

        Returns
        -------
        `coordinax.AbstractPos`
            The vector represented as the target type.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> sph = vec.represent_as(cx.SphericalPos)
        >>> sph
        SphericalPos(
            r=Distance(value=f32[], unit=Unit("m")),
            theta=Quantity[...](value=f32[], unit=Unit("rad")),
            phi=Quantity[...](value=f32[], unit=Unit("rad")) )
        >>> sph.r
        Distance(Array(3.7416575, dtype=float32), unit='m')

        """
        return represent_as(self, target, *args, **kwargs)

    @partial(eqx.filter_jit, inline=True)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Returns
        -------
        Quantity
            The norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        >>> v = cx.CartesianPos1D.from_([-1], "kpc")
        >>> v.norm()
        Quantity['length'](Array(1., dtype=float32), unit='kpc')

        >>> v = cx.CartesianPos2D.from_([3, 4], "kpc")
        >>> v.norm()
        Quantity['length'](Array(5., dtype=float32), unit='kpc')

        >>> v = cx.PolarPos(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))
        >>> v.norm()
        Quantity['length'](Array(3., dtype=float32), unit='kpc')

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
    return qlax.add(  # re-dispatch on the Cartesian class
        lhs.represent_as(cart_cls), rhs.represent_as(cart_cls)
    ).represent_as(type(lhs))


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
    return replace(lhs, **{k: jnp.divide(v, rhs) for k, v in field_items(lhs)})


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
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import quaxed.numpy as jnp

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> jnp.multiply(2, vec)
    CartesianPos3D(
      x=Quantity[...](value=f32[], unit=Unit("m")),
      y=Quantity[...](value=f32[], unit=Unit("m")),
      z=Quantity[...](value=f32[], unit=Unit("m"))
    )

    Most of the position classes have specific dispatches for this operation.
    So let's define a new class and try it out:

    >>> from typing import ClassVar
    >>> class MyCartesian(cx.AbstractPos):
    ...     x: Quantity
    ...     y: Quantity
    ...     z: Quantity
    ...
    >>> MyCartesian._cartesian_cls = MyCartesian  # hack

    Add conversion to Quantity:

    >>> from plum import conversion_method
    >>> @conversion_method(MyCartesian, Quantity)
    ... def _to_quantity(x: MyCartesian, /) -> Quantity:
    ...     return jnp.stack((x.x, x.y, x.z), axis=-1)

    Add representation transformation

    >>> from plum import dispatch
    >>> @dispatch
    ... def represent_as(current: MyCartesian, target: type[MyCartesian], /) -> MyCartesian:
    ...     return current

    >>> vec = MyCartesian(x=Quantity([1], "m"),
    ...                   y=Quantity([2], "m"),
    ...                   z=Quantity([3], "m"))

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
    ... def represent_as(current: MyCartesian, target: type[cx.CartesianPos3D], /) -> cx.CartesianPos3D:
    ...     return cx.CartesianPos3D(x=current.x, y=current.y, z=current.z)
    >>> @dispatch
    ... def represent_as(current: cx.CartesianPos3D, target: type[MyCartesian], /) -> MyCartesian:
    ...     return MyCartesian(x=current.x, y=current.y, z=current.z)

    >>> jnp.multiply(2, vec)
    MyCartesian(
      x=Quantity[...](value=f32[1], unit=Unit("m")),
      y=Quantity[...](value=f32[1], unit=Unit("m")),
      z=Quantity[...](value=f32[1], unit=Unit("m"))
    )

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

    rc = rhs.represent_as(rhs._cartesian_cls)  # noqa: SLF001
    nr = qlax.mul(lhs, rc)
    return nr.represent_as(type(rhs))


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_pos_v(lhs: AbstractPos, rhs: ArrayLike, /) -> AbstractPos:
    """Scale a position by a scalar.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import quaxed.numpy as jnp

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> jnp.multiply(vec, 2)
    CartesianPos3D(
      x=Quantity[...](value=f32[], unit=Unit("m")),
      y=Quantity[...](value=f32[], unit=Unit("m")),
      z=Quantity[...](value=f32[], unit=Unit("m"))
    )

    """
    return qlax.mul(rhs, lhs)  # re-dispatch on the other side


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_pos_pos(lhs: AbstractPos, rhs: AbstractPos, /) -> Quantity:
    """Multiply two positions.

    This is required to take the dot product of two vectors.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D(
    ...     x=Quantity([1, 2, 3], "m"),
    ...     y=Quantity([4, 5, 6], "m"),
    ...     z=Quantity([7, 8, 9], "m"))

    >>> jnp.multiply(vec, vec)  # element-wise multiplication
    Quantity['area'](Array([[ 1., 16., 49.],
       [ 4., 25., 64.],
       [ 9., 36., 81.]], dtype=float32), unit='m2')

    >>> jnp.linalg.vector_norm(vec, axis=-1)
    Quantity['length'](Array([ 8.124039,  9.643651, 11.224972], dtype=float32), unit='m')

    """  # noqa: E501
    lq = convert(lhs.represent_as(lhs._cartesian_cls), Quantity)  # noqa: SLF001
    rq = convert(rhs.represent_as(rhs._cartesian_cls), Quantity)  # noqa: SLF001
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
    >>> -vec
    CartesianPos3D(
        x=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
        y=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")),
        z=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m"))
    )
    >>> (-vec).x
    Quantity['length'](Array(-1., dtype=float32), unit='m')

    """
    cart = represent_as(obj, obj._cartesian_cls)  # noqa: SLF001
    negcart = jnp.negative(cart)
    return represent_as(negcart, type(obj))


# ------------------------------------------------


@register(jax.lax.reshape_p)  # type: ignore[misc]
def _reshape_pos(
    operand: AbstractPos, *, new_sizes: tuple[int, ...], **kwargs: Any
) -> AbstractPos:
    """Reshape the components of the vector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import quaxed.numpy as jnp

    >>> vec = cx.CartesianPos3D(x=Quantity([1, 2, 3], "m"),
    ...                         y=Quantity([4, 5, 6], "m"),
    ...                         z=Quantity([7, 8, 9], "m"))
    >>> jnp.reshape(vec, shape=(3, 1, 3))  # (n_components *shape)
    CartesianPos3D(
      x=Quantity[PhysicalType('length')](value=f32[1,1,3], unit=Unit("m")),
      y=Quantity[PhysicalType('length')](value=f32[1,1,3], unit=Unit("m")),
      z=Quantity[PhysicalType('length')](value=f32[1,1,3], unit=Unit("m"))
    )

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
        lhs.represent_as(lhs._cartesian_cls),  # noqa: SLF001
        rhs.represent_as(lhs._cartesian_cls),  # noqa: SLF001
    ).represent_as(type(lhs))
