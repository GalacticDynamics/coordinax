"""Built-in vector classes."""

__all__ = [
    "CartesianAcc2D",
    "CartesianPos2D",
    "CartesianVel2D",
]

import functools as ft
from typing import final
from typing_extensions import override

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u

import coordinax._src.custom_types as ct
from .base import AbstractAcc2D, AbstractPos2D, AbstractVel2D
from coordinax._src.distances import BBtLength
from coordinax._src.vectors.base.cartesian import AbstractCartesian


@final
class CartesianPos2D(AbstractCartesian, AbstractPos2D):
    """Cartesian 2D Position.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianPos2D.from_([1, 2], "m")
    >>> print(vec)
    <CartesianPos2D: (x, y) [m]
        [1 2]>

    """

    x: BBtLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: BBtLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""


@final
class CartesianVel2D(AbstractCartesian, AbstractVel2D):
    """Cartesian 2D Velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianVel2D.from_([1, 2], "m/s")
    >>> print(vec)
    <CartesianVel2D: (x, y) [m / s]
        [1 2]>

    """

    x: ct.BBtSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""X coordinate differential :math:`\dot{x} \in (-\infty,+\infty)`."""

    y: ct.BBtSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Y coordinate differential :math:`\dot{y} \in (-\infty,+\infty)`."""

    @override
    def norm(self, _: AbstractPos2D | None = None, /) -> ct.BBtSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> v = cx.vecs.CartesianVel2D.from_([3, 4], "km/s")
        >>> v.norm()
        Quantity(Array(5., dtype=float32), unit='km / s')

        """
        return jnp.sqrt(self.x**2 + self.y**2)


@final
class CartesianAcc2D(AbstractCartesian, AbstractAcc2D):
    """Cartesian Acceleration 3D.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianAcc2D.from_([1, 2], "m/s2")
    >>> print(vec)
    <CartesianAcc2D: (x, y) [m / s2]
        [1 2]>

    """

    x: ct.BBtAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""X coordinate acceleration :math:`\frac{d^2 x}{dt^2} \in (-\infty,+\infty)`."""

    y: ct.BBtAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Y coordinate acceleration :math:`\frac{d^2 y}{dt^2} \in (-\infty,+\infty)`."""

    @override
    @ft.partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractVel2D | None = None, /) -> ct.BBtAcc:  # type: ignore[override]
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> v = cx.vecs.CartesianAcc2D.from_([3, 4], "km/s2")
        >>> v.norm()
        Quantity(Array(5., dtype=float32), unit='km / s2')

        """
        return jnp.sqrt(self.x**2 + self.y**2)
