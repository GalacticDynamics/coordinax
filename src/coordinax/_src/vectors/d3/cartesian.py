"""Built-in vector classes."""

__all__ = [
    "CartesianAcc3D",
    "CartesianPos3D",
    "CartesianVel3D",
]

from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u

import coordinax._src.typing as ct
from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty
from coordinax._src.vectors.base.cartesian import AbstractCartesian

#####################################################################
# Position


@final
class CartesianPos3D(AbstractCartesian, AbstractPos3D):
    """Cartesian 3D Position.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_(u.Quantity([1, 2, 3], "m"))
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [1 2 3]>

    """

    x: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    z: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Z coordinate :math:`z \in (-\infty,+\infty)`."""

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVel3D"]:  # type: ignore[override]
        """Return the differential class.

        Examples
        --------
        >>> import coordinax as cx
        >>> print(cx.vecs.CartesianPos3D.differential_cls)
        <class 'coordinax...CartesianVel3D'>

        """
        return CartesianVel3D


#####################################################################
# Velocity


@final
class CartesianVel3D(AbstractCartesian, AbstractVel3D):
    """Cartesian 3D Velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.CartesianVel3D.from_([1, 2, 3], "m/s")
    >>> print(vec)
    <CartesianVel3D (x[m / s], y[m / s], z[m / s])
        [1 2 3]>

    """

    x: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""X speed :math:`dx/dt \in [-\infty, \infty]."""

    y: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Y speed :math:`dy/dt \in [-\infty, \infty]."""

    z: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Z speed :math:`dz/dt \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPos3D]:  # type: ignore[override]
        return CartesianPos3D

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAcc3D"]:  # type: ignore[override]
        """Return the differential class.

        Examples
        --------
        >>> import coordinax as cx
        >>> print(cx.vecs.CartesianVel3D.differential_cls)
        <class 'coordinax...CartesianAcc3D'>

        """
        return CartesianAcc3D

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
        return jnp.sqrt(self.x**2 + self.y**2 + self.z**2)


#####################################################################
# Acceleration


@final
class CartesianAcc3D(AbstractCartesian, AbstractAcc3D):
    """Cartesian differential representation."""

    x: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""X acceleration :math:`d^2x/dt^2 \in [-\infty, \infty]."""

    y: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Y acceleration :math:`d^2y/dt^2 \in [-\infty, \infty]."""

    z: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Z acceleration :math:`d^2z/dt^2 \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVel3D]:  # type: ignore[override]
        return CartesianVel3D

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
        return jnp.sqrt(self.x**2 + self.y**2 + self.z**2)
