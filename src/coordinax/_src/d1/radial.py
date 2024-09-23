"""Carteisan vector."""

__all__ = ["RadialPosition", "RadialVelocity", "RadialAcceleration"]

from functools import partial
from typing import final

import equinox as eqx
import jax
from plum import convert

from dataclassish.converters import Unless
from unxt import AbstractDistance, Distance, Quantity

import coordinax._src.typing as ct
from .base import AbstractAcceleration1D, AbstractPosition1D, AbstractVelocity1D
from coordinax._src.checks import check_r_non_negative
from coordinax._src.utils import classproperty


@final
class RadialPosition(AbstractPosition1D):
    """Radial vector representation."""

    r: ct.BatchableDistance = eqx.field(
        converter=Unless(AbstractDistance, partial(Distance.constructor, dtype=float))
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        check_r_non_negative(self.r)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["RadialVelocity"]:
        return RadialVelocity


@final
class RadialVelocity(AbstractVelocity1D):
    """Radial differential representation."""

    d_r: ct.BatchableSpeed = eqx.field(converter=Quantity["speed"].constructor)
    r"""Radial speed :math:`dr/dt \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[RadialPosition]:
        return RadialPosition

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["RadialAcceleration"]:
        return RadialAcceleration

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        # TODO: change to UncheckedQuantity
        return jax.core.get_aval(convert(self, Quantity).value)


@final
class RadialAcceleration(AbstractAcceleration1D):
    """Radial differential representation."""

    d2_r: ct.BatchableAcc = eqx.field(converter=Quantity["acceleration"].constructor)
    r"""Radial acceleration :math:`d^2r/dt^2 \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[RadialVelocity]:
        return RadialVelocity

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        # TODO: change to UncheckedQuantity
        return jax.core.get_aval(convert(self, Quantity).value)
