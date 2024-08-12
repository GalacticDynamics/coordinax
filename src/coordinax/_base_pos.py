"""Representation of coordinates in different systems."""

__all__ = ["AbstractPosition"]

import operator
import warnings
from abc import abstractmethod
from dataclasses import replace
from functools import partial
from inspect import isabstract
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from plum import convert, dispatch
from quax import quaxify, register

import quaxed.array_api as xp
import quaxed.lax as qlax
from dataclassish import field_items
from unxt import Quantity

import coordinax._typing as ct
from ._base import AbstractVector
from ._utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Self

VT = TypeVar("VT", bound="AbstractPosition")

VECTOR_CLASSES: set[type["AbstractPosition"]] = set()


class AbstractPosition(AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of coordinates in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered if it is not an abstract class.
        """
        # TODO: a more robust check using equinox.
        if isabstract(cls) or cls.__name__.startswith("Abstract"):
            return

        VECTOR_CLASSES.add(cls)

    @classproperty
    @classmethod
    @abstractmethod
    def _cartesian_cls(cls) -> type["AbstractVector"]:
        """Return the corresponding Cartesian vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.RadialPosition._cartesian_cls
        <class 'coordinax...CartesianPosition1D'>

        >>> cx.SphericalPosition._cartesian_cls
        <class 'coordinax...CartesianPosition3D'>

        """
        # TODO: something nicer than this for getting the corresponding class
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractVelocity"]:
        """Return the corresponding differential vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.RadialPosition.differential_cls.__name__
        'RadialVelocity'

        >>> cx.SphericalPosition.differential_cls.__name__
        'SphericalVelocity'

        """
        raise NotImplementedError

    # ===============================================================
    # Quax

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        >>> vec = cx.CartesianPosition3D.constructor([1, 2, 3], "m")
        >>> vec.aval()
        ConcreteArray([1. 2. 3.], dtype=float32)

        >>> vec = cx.CartesianPosition3D.constructor([[1, 2, 3], [4, 5, 6]], "m")
        >>> vec.aval()
        ConcreteArray([[1. 2. 3.]
                       [4. 5. 6.]], dtype=float32)

        """
        return jax.core.get_aval(
            convert(self.represent_as(self._cartesian_cls), Quantity).value
        )

    # ===============================================================
    # Array

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        The default implementation is to go through Cartesian coordinates.
        """
        cart = self.represent_as(self._cartesian_cls)
        return (-cart).represent_as(type(self))

    # -----------------------------------------------------
    # Binary arithmetic operations

    @AbstractVector.__add__.dispatch  # type: ignore[misc]
    def __add__(
        self: "AbstractPosition", other: "AbstractPosition"
    ) -> "AbstractPosition":  # TODO: use Self
        """Add another object to this vector."""
        # The base implementation is to convert to Cartesian and perform the
        # operation.  Cartesian coordinates do not have any branch cuts or
        # singularities or ranges that need to be handled, so this is a safe
        # default.
        return operator.add(
            self.represent_as(self._cartesian_cls),
            other.represent_as(self._cartesian_cls),
        ).represent_as(type(self))

    @AbstractVector.__sub__.dispatch  # type: ignore[misc]
    def __sub__(
        self: "AbstractPosition", other: "AbstractPosition"
    ) -> "AbstractPosition":  # TODO: use Self
        """Add another object to this vector."""
        # The base implementation is to convert to Cartesian and perform the
        # operation.  Cartesian coordinates do not have any branch cuts or
        # singularities or ranges that need to be handled, so this is a safe
        # default.
        return operator.sub(
            self.represent_as(self._cartesian_cls),
            other.represent_as(self._cartesian_cls),
        ).represent_as(type(self))

    @AbstractVector.__mul__.dispatch  # type: ignore[misc]
    def __mul__(self: "AbstractPosition", other: ArrayLike) -> "AbstractPosition":
        """Multiply the vector by a scalar.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        >>> vec = cx.CartesianPosition3D.constructor([1, 2, 3], "m")
        >>> (vec * 2).x
        Quantity['length'](Array(2., dtype=float32), unit='m')

        """
        return qlax.mul(self, other)

    @AbstractVector.__rmul__.dispatch  # type: ignore[misc]
    def __rmul__(self: "AbstractPosition", other: ArrayLike) -> "AbstractPosition":
        """Multiply the vector by a scalar.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        >>> vec = cx.CartesianPosition3D.constructor([1, 2, 3], "m")
        >>> (2 * vec).x
        Quantity['length'](Array(2., dtype=float32), unit='m')

        """
        return qlax.mul(other, self)

    @AbstractVector.__truediv__.dispatch  # type: ignore[misc]
    def __truediv__(
        self: "AbstractPosition", other: ArrayLike
    ) -> "AbstractPosition":  # TODO: use Self
        return replace(self, **{k: v / other for k, v in field_items(self)})

    # ===============================================================
    # Convenience methods

    @partial(jax.jit, static_argnums=1)
    def represent_as(self, target: type[VT], /, *args: Any, **kwargs: Any) -> VT:
        """Represent the vector as another type.

        Parameters
        ----------
        target : type[AbstractPosition]
            The type to represent the vector as.
        *args : Any
            Extra arguments. Raises a warning if any are given.
        **kwargs : Any
            Extra keyword arguments.

        Returns
        -------
        AbstractPosition
            The vector represented as the target type.

        Warns
        -----
        UserWarning
            If extra arguments are given.

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> import coordinax as cx

        We can represent a vector as another type:

        >>> x, y, z = Quantity(1, "meter"), Quantity(2, "meter"), Quantity(3, "meter")
        >>> vec = cx.CartesianPosition3D(x=x, y=y, z=z)
        >>> sph = vec.represent_as(cx.SphericalPosition)
        >>> sph
        SphericalPosition(
            r=Distance(value=f32[], unit=Unit("m")),
            theta=Quantity[...](value=f32[], unit=Unit("rad")),
            phi=Quantity[...](value=f32[], unit=Unit("rad")) )
        >>> sph.r
        Distance(Array(3.7416575, dtype=float32), unit='m')

        """
        if any(args):
            warnings.warn("Extra arguments are ignored.", UserWarning, stacklevel=2)

        from ._transform import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, **kwargs)

    @partial(jax.jit, inline=True)
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

        >>> v = cx.CartesianPosition1D.constructor([-1], "kpc")
        >>> v.norm()
        Quantity['length'](Array(1., dtype=float32), unit='kpc')

        >>> v = cx.CartesianPosition2D.constructor([3, 4], "kpc")
        >>> v.norm()
        Quantity['length'](Array(5., dtype=float32), unit='kpc')

        >>> v = cx.PolarPosition(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))
        >>> v.norm()
        Quantity['length'](Array(3., dtype=float32), unit='kpc')

        >>> v = cx.CartesianPosition3D.constructor([1, 2, 3], "m")
        >>> v.norm()
        Quantity['length'](Array(3.7416575, dtype=float32), unit='m')

        """
        return xp.linalg.vector_norm(self, axis=-1)


# ===================================================================
# Register dispatches


# from coordinax._funcs
@dispatch  # type: ignore[misc]
@partial(jax.jit, inline=True)
def normalize_vector(x: AbstractPosition, /) -> AbstractPosition:
    """Return the norm of the vector.

    Returns
    -------
    Quantity
        The norm of the vector.

    """
    # TODO: the issue is units! what should the units be?
    raise NotImplementedError  # pragma: no cover


# ------------------------------------------------


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_p_vq(lhs: ArrayLike, rhs: AbstractPosition, /) -> AbstractPosition:
    """Scale a position by a scalar.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import quaxed.array_api as jnp

    >>> vec = cx.CartesianPosition3D.constructor([1, 2, 3], "m")
    >>> jnp.multiply(2, vec)
    CartesianPosition3D(
      x=Quantity[...](value=f32[], unit=Unit("m")),
      y=Quantity[...](value=f32[], unit=Unit("m")),
      z=Quantity[...](value=f32[], unit=Unit("m"))
    )

    Most of the position classes have specific dispatches for this operation.
    So let's define a new class and try it out:

    >>> from typing import ClassVar
    >>> from coordinax._utils import classproperty
    >>> class MyCartesian(cx.AbstractPosition):
    ...     x: Quantity
    ...     y: Quantity
    ...     z: Quantity
    ...
    >>> MyCartesian._cartesian_cls = MyCartesian  # hack

    Add conversion to Quantity:

    >>> from plum import conversion_method
    >>> @conversion_method(MyCartesian, Quantity)
    ... def _to_quantity(x: MyCartesian, /) -> Quantity:
    ...     return xp.stack((x.x, x.y, x.z), axis=-1)

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

    >>> MyCartesian._cartesian_cls = cx.CartesianPosition3D
    >>> @dispatch
    ... def represent_as(current: MyCartesian, target: type[cx.CartesianPosition3D], /) -> cx.CartesianPosition3D:
    ...     return cx.CartesianPosition3D(x=current.x, y=current.y, z=current.z)
    >>> @dispatch
    ... def represent_as(current: cx.CartesianPosition3D, target: type[MyCartesian], /) -> MyCartesian:
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
def _mul_p_qv(lhs: AbstractPosition, rhs: ArrayLike, /) -> AbstractPosition:
    """Scale a position by a scalar.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import quaxed.numpy as jnp

    >>> vec = cx.CartesianPosition3D.constructor([1, 2, 3], "m")
    >>> jnp.multiply(vec, 2)
    CartesianPosition3D(
      x=Quantity[...](value=f32[], unit=Unit("m")),
      y=Quantity[...](value=f32[], unit=Unit("m")),
      z=Quantity[...](value=f32[], unit=Unit("m"))
    )

    """
    return qlax.mul(rhs, lhs)  # re-dispatch on the other side


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_p_qq(lhs: AbstractPosition, rhs: AbstractPosition, /) -> Quantity:
    """Multiply two positions.

    This is required to take the dot product of two vectors.

    Examples
    --------
    >>> import quaxed.array_api as jnp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPosition3D(
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


@register(jax.lax.reshape_p)  # type: ignore[misc]
def _reshape_p(
    operand: AbstractPosition, *, new_sizes: tuple[int, ...], **kwargs: Any
) -> AbstractPosition:
    """Reshape the components of the vector.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import quaxed.numpy as jnp

    >>> vec = cx.CartesianPosition3D(x=Quantity([1, 2, 3], "m"),
    ...                              y=Quantity([4, 5, 6], "m"),
    ...                              z=Quantity([7, 8, 9], "m"))
    >>> jnp.reshape(vec, shape=(3, 1, 3))  # (n_components *shape)
    CartesianPosition3D(
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
