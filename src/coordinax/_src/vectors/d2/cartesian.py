"""Built-in vector classes."""

__all__ = [
    "CartesianAcc2D",
    "CartesianPos2D",
    "CartesianVel2D",
]

from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u

import coordinax._src.typing as ct
from .base import AbstractAcc2D, AbstractPos2D, AbstractVel2D
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty


@final
class CartesianPos2D(AbstractPos2D):
    """Cartesian 2D Position.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianPos2D.from_([1, 2], "m")
    >>> print(vec)
    <CartesianPos2D (x[m], y[m])
        [1 2]>

    """

    x: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVel2D"]:  # type: ignore[override]
        return CartesianVel2D


#####################################################################


@final
class CartesianVel2D(AbstractVel2D):
    """Cartesian 2D Velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianVel2D.from_([1, 2], "m/s")
    >>> print(vec)
    <CartesianVel2D (d_x[m / s], d_y[m / s])
        [1 2]>

    """

    d_x: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""X coordinate differential :math:`\dot{x} \in (-\infty,+\infty)`."""

    d_y: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Y coordinate differential :math:`\dot{y} \in (-\infty,+\infty)`."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPos2D]:  # type: ignore[override]
        return CartesianPos2D

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAcc2D"]:  # type: ignore[override]
        """Return the differential class.

        Examples
        --------
        >>> import coordinax as cx
        >>> print(cx.vecs.CartesianVel2D.differential_cls)
        <class 'coordinax...CartesianAcc2D'>

        """
        return CartesianAcc2D

    @override
    def norm(self, _: AbstractPos2D | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> v = cx.vecs.CartesianVel2D.from_([3, 4], "km/s")
        >>> v.norm()
        Quantity['speed'](Array(5., dtype=float32), unit='km / s')

        """
        return jnp.sqrt(self.d_x**2 + self.d_y**2)


#####################################################################


@final
class CartesianAcc2D(AbstractAcc2D):
    """Cartesian Acceleration 3D.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianAcc2D.from_([1, 2], "m/s2")
    >>> print(vec)
    <CartesianAcc2D (d2_x[m / s2], d2_y[m / s2])
        [1 2]>

    """

    d2_x: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""X coordinate acceleration :math:`\frac{d^2 x}{dt^2} \in (-\infty,+\infty)`."""

    d2_y: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Y coordinate acceleration :math:`\frac{d^2 y}{dt^2} \in (-\infty,+\infty)`."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVel2D]:  # type: ignore[override]
        return CartesianVel2D

    # -----------------------------------------------------

    @override
    @partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractVel2D | None = None, /) -> ct.BatchableAcc:  # type: ignore[override]
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> v = cx.vecs.CartesianAcc2D.from_([3, 4], "km/s2")
        >>> v.norm()
        Quantity['acceleration'](Array(5., dtype=float32), unit='km / s2')

        """
        return jnp.sqrt(self.d2_x**2 + self.d2_y**2)
