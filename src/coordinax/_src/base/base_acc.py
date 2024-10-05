"""Representation of coordinates in different systems."""

__all__ = ["AbstractAcc"]

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar
from typing_extensions import override

import equinox as eqx
import jax
from quax import register

import quaxed.numpy as jnp
from dataclassish import field_items
from quaxed import lax as qlax
from unxt import Quantity

from .base import AbstractVector
from .base_pos import AbstractPos
from .base_vel import AbstractVel
from coordinax._src.funcs import represent_as
from coordinax._src.utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Self

AccT = TypeVar("AccT", bound="AbstractAcc")

ACCELERATION_CLASSES: set[type["AbstractAcc"]] = set()


class AbstractAcc(AbstractVector):  # pylint: disable=abstract-method
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

        >>> cx.CartesianAcc3D._cartesian_cls
        <class 'coordinax...CartesianAcc3D'>

        >>> cx.SphericalAcc._cartesian_cls
        <class 'coordinax...CartesianAcc3D'>

        """
        # TODO: something nicer than this for getting the corresponding class
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVel]:
        """Return the corresponding vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.RadialAcc.integral_cls.__name__
        'RadialVel'

        >>> cx.SphericalAcc.integral_cls.__name__
        'SphericalVel'

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

        >>> d2r = cx.RadialAcc.from_([1], "m/s2")
        >>> -d2r
        RadialAcc( d2_r=Quantity[...](value=i32[], unit=Unit("m / s2")) )

        >>> d2p = cx.PolarAcc(Quantity(1, "m/s2"), Quantity(1, "mas/yr2"))
        >>> negd2p = -d2p
        >>> negd2p.d2_r
        Quantity['acceleration'](Array(-1., dtype=float32), unit='m / s2')
        >>> negd2p.d2_phi
        Quantity['angular acceleration'](Array(-1., dtype=float32), unit='mas / yr2')

        """
        return jax.tree.map(jnp.negative, self)

    # ===============================================================
    # Convenience methods

    @override
    def represent_as(self, target: type[AccT], /, *args: Any, **kwargs: Any) -> AccT:
        """Represent the vector as another type.

        This just forwards to `coordinax.represent_as`.

        Parameters
        ----------
        target : type[`coordinax.AbstractVel`]
            The type to represent the vector as.
        *args, **kwargs : Any
            Extra arguments. These are passed to `coordinax.represent_as` and
            might be used, depending on the dispatched method. Generally the
            first argument is the velocity (`coordinax.AbstractVel`)
            followed by the position (`coordinax.AbstractPos`) at which the
            acceleration is defined. In general this is a required argument,
            though it is not for Cartesian-to-Cartesian transforms -- see
            https://en.wikipedia.org/wiki/Tensors_in_curvilinear_coordinates for
            more information.

        Returns
        -------
        `coordinax.AbstractAcc`
            The vector represented as the target type.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> p = cx.CartesianVel3D.from_([4, 5, 6], "m/s")
        >>> a = cx.CartesianAcc3D.from_([7, 8, 9], "m/s2")
        >>> sph = a.represent_as(cx.SphericalAcc, p, q)
        >>> sph
        SphericalAcc(
            d2_r=Quantity[...](value=f32[], unit=Unit("m / s2")),
            d2_theta=Quantity[...]( value=f32[], unit=Unit("rad / s2") ),
            d2_phi=Quantity[...]( value=f32[], unit=Unit("rad / s2") )
        )
        >>> sph.d2_r
        Quantity['acceleration'](Array(13.363062, dtype=float32), unit='m / s2')

        """
        return represent_as(self, target, *args, **kwargs)

    @partial(eqx.filter_jit, inline=True)
    def norm(
        self, velocity: AbstractVel, position: AbstractPos, /
    ) -> Quantity["speed"]:
        """Return the norm of the vector."""
        return self.represent_as(self._cartesian_cls, velocity, position).norm()


# ===============================================================


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_acc_time(lhs: AbstractAcc, rhs: Quantity["time"]) -> AbstractVel:
    """Multiply the vector by a :class:`unxt.Quantity`.

    Examples
    --------
    >>> from quaxed import lax
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> d2r = cx.RadialAcc(Quantity(1, "m/s2"))
    >>> vec = lax.mul(d2r, Quantity(2, "s"))
    >>> vec
    RadialVel( d_r=Quantity[...]( value=...i32[], unit=Unit("m / s") ) )
    >>> vec.d_r
    Quantity['speed'](Array(2, dtype=int32, ...), unit='m / s')

    >>> (d2r * Quantity(2, "s")).d_r
    Quantity['speed'](Array(2, dtype=int32, ...), unit='m / s')

    """
    # TODO: better access to corresponding fields
    return lhs.integral_cls.from_(
        {k.replace("2", ""): jnp.multiply(v, rhs) for k, v in field_items(lhs)}
    )


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_time_acc(lhs: Quantity["time"], rhs: AbstractAcc) -> AbstractVel:
    """Multiply a scalar by an acceleration.

    Examples
    --------
    >>> from quaxed import lax
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> d2r = cx.RadialAcc(Quantity(1, "m/s2"))
    >>> vec = lax.mul(Quantity(2, "s"), d2r)
    >>> vec
    RadialVel( d_r=Quantity[...]( value=...i32[], unit=Unit("m / s") ) )
    >>> vec.d_r
    Quantity['speed'](Array(2, dtype=int32, ...), unit='m / s')

    """
    return qlax.mul(rhs, lhs)  # pylint: disable=arguments-out-of-order


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_acc_time2(lhs: AbstractAcc, rhs: Quantity["s2"]) -> AbstractPos:
    """Multiply an acceleration by a scalar.

    Examples
    --------
    >>> from quaxed import lax
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> d2r = cx.RadialAcc(Quantity(1, "m/s2"))
    >>> vec = lax.mul(d2r, Quantity(2, "s2"))
    >>> vec
    RadialPos(r=Distance(value=f32[], unit=Unit("m")))
    >>> vec.r
    Distance(Array(2., dtype=float32), unit='m')

    >>> (d2r * Quantity(2, "s2")).r
    Distance(Array(2., dtype=float32), unit='m')

    """
    # TODO: better access to corresponding fields
    return lhs.integral_cls.integral_cls.from_(
        {k.replace("d2_", ""): v * rhs for k, v in field_items(lhs)}
    )


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_time2_acc(lhs: Quantity["s2"], rhs: AbstractAcc) -> AbstractPos:
    """Multiply a scalar by an acceleration.

    Examples
    --------
    >>> from quaxed import lax
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> d2r = cx.RadialAcc(Quantity(1, "m/s2"))
    >>> vec = lax.mul(Quantity(2, "s2"), d2r)
    >>> vec
    RadialPos(r=Distance(value=f32[], unit=Unit("m")))
    >>> vec.r
    Distance(Array(2., dtype=float32), unit='m')

    """
    return qlax.mul(rhs, lhs)  # pylint: disable=arguments-out-of-order
