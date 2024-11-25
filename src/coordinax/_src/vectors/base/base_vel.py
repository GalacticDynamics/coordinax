"""Representation of velocities in different systems."""

__all__ = ["AbstractVel"]

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import jax
from quax import register

import quaxed.numpy as jnp
from dataclassish import field_items
from unxt import Quantity

from .base import AbstractVector
from .base_pos import AbstractPos
from coordinax._src.utils import classproperty

if TYPE_CHECKING:
    from typing_extensions import Self

VelT = TypeVar("VelT", bound="AbstractVel")

DIFFERENTIAL_CLASSES: set[type["AbstractVel"]] = set()


class AbstractVel(AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of vector differentials in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered.
        """
        DIFFERENTIAL_CLASSES.add(cls)

    @classproperty
    @classmethod
    @abstractmethod
    def _cartesian_cls(cls) -> type["AbstractVector"]:
        """Return the corresponding Cartesian vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.RadialVel._cartesian_cls
        <class 'coordinax...CartesianVel1D'>

        >>> cx.SphericalVel._cartesian_cls
        <class 'coordinax...CartesianVel3D'>

        """
        # TODO: something nicer than this for getting the corresponding class
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type["AbstractPos"]:
        """Return the corresponding vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.RadialVel.integral_cls.__name__
        'RadialPos'

        >>> cx.SphericalVel.integral_cls.__name__
        'SphericalPos'

        """
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractAcc"]:
        """Return the corresponding differential vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.RadialVel.differential_cls.__name__
        'RadialAcc'

        >>> cx.SphericalVel.differential_cls.__name__
        'SphericalAcc'

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

        >>> dr = cx.RadialVel.from_([1], "m/s")
        >>> -dr
        RadialVel( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

        >>> dp = cx.PolarVel(Quantity(1, "m/s"), Quantity(1, "mas/yr"))
        >>> neg_dp = -dp
        >>> neg_dp.d_r
        Quantity['speed'](Array(-1., dtype=float32), unit='m / s')
        >>> neg_dp.d_phi
        Quantity['angular frequency'](Array(-1., dtype=float32), unit='mas / yr')

        """
        return jax.tree.map(jnp.negative, self)

    # ===============================================================
    # Convenience methods

    @partial(eqx.filter_jit, inline=True)
    def norm(self, position: AbstractPos, /) -> Quantity["speed"]:
        """Return the norm of the vector."""
        return self.represent_as(self._cartesian_cls, position).norm()


# ---------------------------------------------------------


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_vel_q(self: AbstractVel, other: Quantity["time"]) -> AbstractPos:
    """Multiply the vector by a time :class:`unxt.Quantity` to get a position.

    Examples
    --------
    >>> from quaxed import lax
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> dr = cx.RadialVel(Quantity(1, "m/s"))
    >>> vec = dr * Quantity(2, "s")
    >>> vec
    RadialPos(r=Distance(value=f32[], unit=Unit("m")))
    >>> vec.r
    Distance(Array(2., dtype=float32), unit='m')

    >>> lax.mul(dr, Quantity(2, "s")).r
    Distance(Array(2., dtype=float32), unit='m')

    """
    return self.integral_cls.from_({k[2:]: v * other for k, v in field_items(self)})
