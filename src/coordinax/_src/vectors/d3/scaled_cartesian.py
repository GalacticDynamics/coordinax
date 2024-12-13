"""Built-in vector classes."""

__all__ = [
    "ScaledCartesianPos3D",
    "ScaledCartesianVel3D",
    "ScaledCartesianAcc3D",
]

from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx
import jax
from plum import dispatch

from unxt import Quantity
from unxt.quantity import AbstractQuantity

import coordinax._src.typing as ct
from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from .cartesian import CartesianPos3D
from .generic import CartesianGeneric3D
from coordinax._src.distances import (
    AbstractDistance,
    BatchableDistance,
    Distance,
)
from coordinax._src.utils import classproperty

#####################################################################
# Position


@final
class ScaledCartesianPos3D(AbstractPos3D):
    """Cartesian vector representation."""

    r: BatchableDistance = eqx.field(
        converter=Unless(AbstractDistance, partial(Distance.from_, dtype=float))
    )

    s: Quantity["dimensionless"] = eqx.field(converter=Quantity[""].from_)
    r"""Scaled x coordinate :math:`x \in (-\infty,+\infty)`."""

    t: Quantity["dimensionless"] = eqx.field(converter=Quantity[""].from_)
    r"""Scaled y coordinate :math:`y \in (-\infty,+\infty)`."""

    u: Quantity["dimensionless"] = eqx.field(converter=Quantity[""].from_)
    r"""Scaled z coordinate :math:`z \in (-\infty,+\infty)`."""

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["ScaledCartesianVel3D"]:
        """Return the differential of the class."""
        return ScaledCartesianVel3D

    @partial(jax.jit)
    def norm(self) -> BatchableDistance:
        return self.r


# =====================================================
# Constructors


@ScaledCartesianPos3D.from_.dispatch  # type: ignore[attr-defined, misc]
def from_(
    cls: type[ScaledCartesianPos3D],
    obj: AbstractQuantity,  # TODO: Shaped[AbstractQuantity, "*batch 3"]
    /,
) -> ScaledCartesianPos3D:
    """Construct a scaled 3D Cartesian position.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.ScaledCartesianPos3D.from_(Quantity([1, 2, 3], "m"))
    >>> vec
    CartesianPos3D(
      r=Distance(value=f32[], unit=Unit("m")),
      s=Quantity[...](value=f32[], unit=Unit("")),
      t=Quantity[...](value=f32[], unit=Unit("")),
      u=Quantity[...](value=f32[], unit=Unit(""))
    )

    """
    cart = CartesianPos3D.from_(obj)
    return cart.vconvert(cls)


# =====================================================
# Functions


# from coordinax.vectors.funcs
@dispatch  # type: ignore[misc]
@partial(eqx.filter_jit, inline=True)
def normalize_vector(obj: ScaledCartesianPos3D, /) -> ScaledCartesianPos3D:
    """Return the norm of the vector.

    This has length 1.

    .. note::

        The unit vector is dimensionless, even if the input vector has units.
        This is because the unit vector is a ratio of two quantities: each
        component and the norm of the vector.

    Returns
    -------
    CartesianGeneric3D
        The norm of the vector.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.ScaledCartesianPos3D.from_([1, 2, 3], "km")
    >>> cx.vecs.normalize_vector(q)
    CartesianGeneric3D(
      x=Quantity[...]( value=f32[], unit=Unit(dimensionless) ),
      y=Quantity[...]( value=f32[], unit=Unit(dimensionless) ),
      z=Quantity[...]( value=f32[], unit=Unit(dimensionless) )
    )

    """
    return CartesianGeneric3D(x=obj.x / obj.r, y=obj.y / obj.r, z=obj.z / obj.r)


#####################################################################
# Velocity


@final
class ScaledCartesianVel3D(AbstractVel3D):
    """Scaled 3D Cartesian velocity."""

    d_r: ct.BatchableSpeed = eqx.field(converter=Quantity["speed"].from_)
    r"""speed :math:`dx/dt \in [-\infty, \infty]."""

    d_s: ct.BatchableSpeed = eqx.field(converter=Quantity["frequency"].from_)
    r"""s speed :math:`dx/dt \in [-\infty, \infty] [frequency]."""

    d_t: ct.BatchableSpeed = eqx.field(converter=Quantity["frequency"].from_)
    r"""s speed :math:`dx/dt \in [-\infty, \infty] [frequency]."""

    d_u: ct.BatchableSpeed = eqx.field(converter=Quantity["frequency"].from_)
    r"""Z speed :math:`dz/dt \in [-\infty, \infty] [frequency]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[ScaledCartesianPos3D]:
        return ScaledCartesianPos3D

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["ScaledCartesianAcc3D"]:
        return ScaledCartesianAcc3D

    @partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractPos3D | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> c = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
        >>> c.norm()
        Quantity['speed'](Array(3.7416575, dtype=float32), unit='km / s')

        """
        return self.d_r


#####################################################################
# Acceleration


@final
class ScaledCartesianAcc3D(AbstractAcc3D):
    """Scaled 3D Cartesian acceleration."""

    d2_r: ct.BatchableAcc = eqx.field(converter=Quantity["acceleration"].from_)
    r"""scaled radial acceleration :math:`d^2x/dt^2 \in [-\infty, \infty]."""

    d2_s: ct.BatchableAcc = eqx.field(converter=Quantity["frequency drift"].from_)
    r"""s acceleration :math:`d^2s/dt^2 \in [-\infty, \infty] [time^-2]."""

    d2_t: ct.BatchableAcc = eqx.field(converter=Quantity["frequency drift"].from_)
    r"""t acceleration :math:`d^2t/dt^2 \in [-\infty, \infty] [time^-2]."""

    d2_u: ct.BatchableAcc = eqx.field(converter=Quantity["frequency drift"].from_)
    r"""u acceleration :math:`d^2u/dt^2 \in [-\infty, \infty] [time^-2]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[ScaledCartesianVel3D]:
        return ScaledCartesianVel3D

    # -----------------------------------------------------
    # Methods

    @override
    @partial(jax.jit, inline=True)
    def norm(
        self, _: AbstractVel3D | None = None, __: AbstractPos3D | None = None, /
    ) -> ct.BatchableAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> c = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
        >>> c.norm()
        Quantity['acceleration'](Array(3.7416575, dtype=float32), unit='km / s2')

        """
        return self.d2_r
