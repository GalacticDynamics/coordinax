"""Carteisan vector."""

__all__ = ["RadialPos", "RadialVel", "RadialAcc"]

from functools import partial
from typing import final

import equinox as eqx
import jax
from plum import convert

from dataclassish.converters import Unless
from unxt import Quantity

import coordinax._src.typing as ct
from .base import AbstractAcc1D, AbstractPos1D, AbstractVel1D
from coordinax._src.checks import check_r_non_negative
from coordinax._src.distance import AbstractDistance, Distance
from coordinax._src.utils import classproperty


@final
class RadialPos(AbstractPos1D):
    """Radial vector representation."""

    r: ct.BatchableDistance = eqx.field(
        converter=Unless(AbstractDistance, partial(Distance.from_, dtype=float))
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        check_r_non_negative(self.r)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["RadialVel"]:
        return RadialVel


@final
class RadialVel(AbstractVel1D):
    """Radial differential representation."""

    d_r: ct.BatchableSpeed = eqx.field(converter=Quantity["speed"].from_)
    r"""Radial speed :math:`dr/dt \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[RadialPos]:
        return RadialPos

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["RadialAcc"]:
        return RadialAcc

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        # TODO: change to UncheckedQuantity
        return jax.core.get_aval(convert(self, Quantity).value)


@final
class RadialAcc(AbstractAcc1D):
    """Radial differential representation."""

    d2_r: ct.BatchableAcc = eqx.field(converter=Quantity["acceleration"].from_)
    r"""Radial acceleration :math:`d^2r/dt^2 \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[RadialVel]:
        return RadialVel

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        # TODO: change to UncheckedQuantity
        return jax.core.get_aval(convert(self, Quantity).value)
