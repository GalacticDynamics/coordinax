"""Carteisan vector."""

__all__ = ["CartesianAcc1D", "CartesianPos1D", "CartesianVel1D"]

from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u

import coordinax._src.typing as ct
from .base import AbstractAcc1D, AbstractPos1D, AbstractVel1D
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty


@final
class CartesianPos1D(AbstractPos1D):
    """Cartesian vector representation.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianPos1D.from_([2], "m")
    >>> vec
    CartesianPos1D(x=Quantity[...](value=i32[], unit=Unit("m")))

    Vectors support the basic math operations:

    >>> (vec + vec).x
    Quantity['length'](Array(4, dtype=int32), unit='m')

    >>> (vec - vec).x
    Quantity['length'](Array(0, dtype=int32), unit='m')

    >>> (3 * vec).x
    Quantity['length'](Array(6, dtype=int32), unit='m')

    """

    x: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVel1D"]:  # type: ignore[override]
        return CartesianVel1D


#####################################################################


@final
class CartesianVel1D(AbstractVel1D):
    """Cartesian differential representation."""

    d_x: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""X differential :math:`dx/dt \in (-\infty,+\infty`)`."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPos1D]:  # type: ignore[override]
        return CartesianPos1D

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAcc1D"]:  # type: ignore[override]
        return CartesianAcc1D

    @override
    @partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractPos1D | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.vecs.CartesianVel1D.from_([-1], "km/s")
        >>> q.norm()
        Quantity['speed'](Array(1, dtype=int32), unit='km / s')

        """
        return jnp.abs(self.d_x)


#####################################################################


@final
class CartesianAcc1D(AbstractAcc1D):
    """Cartesian differential representation."""

    d2_x: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""X differential :math:`d^2x/dt^2 \in (-\infty,+\infty`)`."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVel1D]:  # type: ignore[override]
        return CartesianVel1D

    # -----------------------------------------------------
    # Methods

    @override
    @partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractPos1D | None = None, /) -> ct.BatchableAcc:  # type: ignore[override]
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.vecs.CartesianAcc1D.from_([-1], "km/s2")
        >>> q.norm()
        Quantity['acceleration'](Array(1, dtype=int32), unit='km / s2')

        """
        return jnp.abs(self.d2_x)
