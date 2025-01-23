"""Radial 1D vectors."""

__all__ = ["RadialAcc", "RadialPos", "RadialVel"]

from typing import final
from typing_extensions import override

import equinox as eqx

import unxt as u
from dataclassish.converters import Unless

import coordinax._src.typing as ct
from .base import AbstractAcc1D, AbstractPos1D, AbstractVel1D
from coordinax._src.distances import AbstractDistance, BatchableDistance, Distance
from coordinax._src.utils import classproperty
from coordinax._src.vectors.checks import check_r_non_negative


@final
class RadialPos(AbstractPos1D):
    """Radial vector representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.RadialPos(u.Quantity([2], "m"))
    >>> print(vec)
    <RadialPos (r[m])
        [[2]]>

    """

    r: BatchableDistance = eqx.field(converter=Unless(AbstractDistance, Distance.from_))
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        check_r_non_negative(self.r)

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["RadialVel"]:  # type: ignore[override]
        return RadialVel


#####################################################################


@final
class RadialVel(AbstractVel1D):
    """Radial velocity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.RadialVel(u.Quantity([2], "m/s"))
    >>> print(vec)
    <RadialVel (d_r[m / s])
        [[2]]>

    """

    d_r: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Radial speed :math:`dr/dt \in (-\infty,+\infty)`."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[RadialPos]:  # type: ignore[override]
        return RadialPos

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["RadialAcc"]:  # type: ignore[override]
        return RadialAcc


#####################################################################


@final
class RadialAcc(AbstractAcc1D):
    """Radial Acceleration.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.RadialAcc(u.Quantity([2], "m/s2"))
    >>> print(vec)
    <RadialAcc (d2_r[m / s2])
        [[2]]>

    """

    d2_r: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Radial acceleration :math:`d^2r/dt^2 \in (-\infty,+\infty)`."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[RadialVel]:  # type: ignore[override]
        return RadialVel
