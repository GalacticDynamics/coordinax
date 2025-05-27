"""Built-in vector classes."""

__all__ = [
    "CartesianAcc3D",
    "CartesianPos3D",
    "CartesianVel3D",
]

import functools as ft
from typing import final
from typing_extensions import override

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u

import coordinax._src.custom_types as ct
from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from coordinax._src.distances import BBtLength
from coordinax._src.vectors.base.cartesian import AbstractCartesian


@final
class CartesianPos3D(AbstractCartesian, AbstractPos3D):
    """Cartesian 3D Position.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_(u.Quantity([1, 2, 3], "m"))
    >>> print(vec)
    <CartesianPos3D: (x, y, z) [m]
        [1 2 3]>

    """

    x: BBtLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: BBtLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    z: BBtLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Z coordinate :math:`z \in (-\infty,+\infty)`."""


@final
class CartesianVel3D(AbstractCartesian, AbstractVel3D):
    """Cartesian 3D Velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.CartesianVel3D.from_([1, 2, 3], "m/s")
    >>> print(vec)
    <CartesianVel3D: (x, y, z) [m / s]
        [1 2 3]>

    """

    x: ct.BBtSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""X speed :math:`dx/dt \in [-\infty, \infty]."""

    y: ct.BBtSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Y speed :math:`dy/dt \in [-\infty, \infty]."""

    z: ct.BBtSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Z speed :math:`dz/dt \in [-\infty, \infty]."""

    @ft.partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractPos3D | None = None, /) -> ct.BBtSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> c = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
        >>> c.norm()
        Quantity(Array(3.7416575, dtype=float32), unit='km / s')

        """
        return jnp.sqrt(self.x**2 + self.y**2 + self.z**2)


@final
class CartesianAcc3D(AbstractCartesian, AbstractAcc3D):
    """Cartesian differential representation."""

    x: ct.BBtAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""X acceleration :math:`d^2x/dt^2 \in [-\infty, \infty]."""

    y: ct.BBtAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Y acceleration :math:`d^2y/dt^2 \in [-\infty, \infty]."""

    z: ct.BBtAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Z acceleration :math:`d^2z/dt^2 \in [-\infty, \infty]."""

    @override
    @ft.partial(jax.jit, inline=True)
    def norm(
        self, _: AbstractVel3D | None = None, __: AbstractPos3D | None = None, /
    ) -> ct.BBtAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> c = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
        >>> c.norm()
        Quantity(Array(3.7416575, dtype=float32), unit='km / s2')

        """
        return jnp.sqrt(self.x**2 + self.y**2 + self.z**2)
