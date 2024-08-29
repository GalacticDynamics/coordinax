"""Representation of coordinates in different systems."""

__all__ = ["AbstractAcceleration"]

from abc import abstractmethod
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import jax
from quax import register

import quaxed.array_api as xp
from dataclassish import field_items
from quaxed import lax as qlax
from unxt import Quantity

from .base import AbstractVector
from .base_pos import AbstractPosition
from .base_vel import AbstractVelocity
from .utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Self

AccT = TypeVar("AccT", bound="AbstractAcceleration")

ACCELERATION_CLASSES: set[type["AbstractAcceleration"]] = set()


class AbstractAcceleration(AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of vector differentials in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered.
        """
        ACCELERATION_CLASSES.add(cls)

    @classproperty
    @classmethod
    @abstractmethod
    def _cartesian_cls(cls) -> type["AbstractVector"]:
        """Return the corresponding Cartesian vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.CartesianAcceleration3D._cartesian_cls
        <class 'coordinax...CartesianAcceleration3D'>

        >>> cx.SphericalAcceleration._cartesian_cls
        <class 'coordinax...CartesianAcceleration3D'>

        """
        # TODO: something nicer than this for getting the corresponding class
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVelocity]:
        """Return the corresponding vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.RadialAcceleration.integral_cls.__name__
        'RadialVelocity'

        >>> cx.SphericalAcceleration.integral_cls.__name__
        'SphericalVelocity'

        """
        raise NotImplementedError

    # ===============================================================
    # Quax

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        raise NotImplementedError

    # ===============================================================
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        >>> d2r = cx.RadialAcceleration.constructor([1], "m/s2")
        >>> -d2r
        RadialAcceleration( d2_r=Quantity[...](value=i32[], unit=Unit("m / s2")) )

        >>> d2p = cx.PolarAcceleration(Quantity(1, "m/s2"), Quantity(1, "mas/yr2"))
        >>> negd2p = -d2p
        >>> negd2p.d2_r
        Quantity['acceleration'](Array(-1., dtype=float32), unit='m / s2')
        >>> negd2p.d2_phi
        Quantity['angular acceleration'](Array(-1., dtype=float32), unit='mas / yr2')

        """
        return replace(self, **{k: -v for k, v in field_items(self)})

    # ===============================================================
    # Convenience methods

    def represent_as(self, target: type[AccT], /, *args: Any, **kwargs: Any) -> AccT:
        """Represent the vector as another type.

        This just forwards to `coordinax.represent_as`.

        Parameters
        ----------
        target : type[`coordinax.AbstractVelocity`]
            The type to represent the vector as.
        *args, **kwargs : Any
            Extra arguments. These are passed to `coordinax.represent_as` and
            might be used, depending on the dispatched method. Generally the
            first argument is the velocity (`coordinax.AbstractVelocity`)
            followed by the position (`coordinax.AbstractPosition`) at which the
            acceleration is defined. In general this is a required argument,
            though it is not for Cartesian-to-Cartesian transforms -- see
            https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for
            more information.

        Returns
        -------
        `coordinax.AbstractAcceleration`
            The vector represented as the target type.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.CartesianPosition3D.constructor([1, 2, 3], "m")
        >>> p = cx.CartesianVelocity3D.constructor([4, 5, 6], "m/s")
        >>> a = cx.CartesianAcceleration3D.constructor([7, 8, 9], "m/s2")
        >>> sph = a.represent_as(cx.SphericalAcceleration, p, q)
        >>> sph
        SphericalAcceleration(
            d2_r=Quantity[...](value=f32[], unit=Unit("m / s2")),
            d2_theta=Quantity[...]( value=f32[], unit=Unit("rad / s2") ),
            d2_phi=Quantity[...]( value=f32[], unit=Unit("rad / s2") )
        )
        >>> sph.d2_r
        Quantity['acceleration'](Array(13.363062, dtype=float32), unit='m / s2')

        """
        from coordinax import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, *args, **kwargs)

    @partial(jax.jit, inline=True)
    def norm(
        self, velocity: AbstractVelocity, position: AbstractPosition, /
    ) -> Quantity["speed"]:
        """Return the norm of the vector."""
        return self.represent_as(self._cartesian_cls, velocity, position).norm()


# ===============================================================


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_acc_time(lhs: AbstractAcceleration, rhs: Quantity["time"]) -> AbstractVelocity:
    """Multiply the vector by a :class:`unxt.Quantity`.

    Examples
    --------
    >>> from quaxed import lax
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> d2r = cx.RadialAcceleration(Quantity(1, "m/s2"))
    >>> vec = lax.mul(d2r, Quantity(2, "s"))
    >>> vec
    RadialVelocity( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )
    >>> vec.d_r
    Quantity['speed'](Array(2, dtype=int32, ...), unit='m / s')

    >>> (d2r * Quantity(2, "s")).d_r
    Quantity['speed'](Array(2, dtype=int32, ...), unit='m / s')

    """
    # TODO: better access to corresponding fields
    return lhs.integral_cls.constructor(
        {k.replace("2", ""): xp.multiply(v, rhs) for k, v in field_items(lhs)}
    )


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_time_acc(lhs: Quantity["time"], rhs: AbstractAcceleration) -> AbstractVelocity:
    """Multiply a scalar by an acceleration.

    Examples
    --------
    >>> from quaxed import lax
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> d2r = cx.RadialAcceleration(Quantity(1, "m/s2"))
    >>> vec = lax.mul(Quantity(2, "s"), d2r)
    >>> vec
    RadialVelocity( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )
    >>> vec.d_r
    Quantity['speed'](Array(2, dtype=int32, ...), unit='m / s')

    """
    return qlax.mul(rhs, lhs)  # pylint: disable=arguments-out-of-order


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_acc_time2(lhs: AbstractAcceleration, rhs: Quantity["s2"]) -> AbstractPosition:
    """Multiply an acceleration by a scalar.

    Examples
    --------
    >>> from quaxed import lax
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> d2r = cx.RadialAcceleration(Quantity(1, "m/s2"))
    >>> vec = lax.mul(d2r, Quantity(2, "s2"))
    >>> vec
    RadialPosition(r=Distance(value=f32[], unit=Unit("m")))
    >>> vec.r
    Distance(Array(2., dtype=float32), unit='m')

    >>> (d2r * Quantity(2, "s2")).r
    Distance(Array(2., dtype=float32), unit='m')

    """
    # TODO: better access to corresponding fields
    return lhs.integral_cls.integral_cls.constructor(
        {k.replace("d2_", ""): v * rhs for k, v in field_items(lhs)}
    )


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_time2_acc(lhs: Quantity["s2"], rhs: AbstractAcceleration) -> AbstractPosition:
    """Multiply a scalar by an acceleration.

    Examples
    --------
    >>> from quaxed import lax
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> d2r = cx.RadialAcceleration(Quantity(1, "m/s2"))
    >>> vec = lax.mul(Quantity(2, "s2"), d2r)
    >>> vec
    RadialPosition(r=Distance(value=f32[], unit=Unit("m")))
    >>> vec.r
    Distance(Array(2., dtype=float32), unit='m')

    """
    return qlax.mul(rhs, lhs)  # pylint: disable=arguments-out-of-order
