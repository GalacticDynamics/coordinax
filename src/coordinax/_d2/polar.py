"""Built-in vector classes."""

__all__ = [
    "PolarPosition",
    "PolarVelocity",
    "PolarAcceleration",
]

from functools import partial
from typing import final

import equinox as eqx
import jax

from unxt import AbstractDistance, Distance, Quantity

import coordinax._typing as ct
from .base import AbstractAcceleration2D, AbstractPosition2D, AbstractVelocity2D
from coordinax._checks import check_azimuth_range, check_r_non_negative
from coordinax._converters import converter_azimuth_to_range
from coordinax._utils import classproperty


@final
class PolarPosition(AbstractPosition2D):
    r"""Polar vector representation.

    Parameters
    ----------
    r : BatchableDistance
        Radial distance :math:`r \in [0,+\infty)`.
    phi : BatchableAngle
        Polar angle :math:`\phi \in [0,2\pi)`.  We use the symbol `phi` to
        adhere to the ISO standard 31-11.

    """

    r: ct.BatchableDistance = eqx.field(
        converter=lambda x: x
        if isinstance(x, AbstractDistance)
        else Distance.constructor(x, dtype=float)
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    phi: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_azimuth_to_range(
            Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Polar angle :math:`\phi \in [0,2\pi)`."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        check_r_non_negative(self.r)
        check_azimuth_range(self.phi)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["PolarVelocity"]:
        return PolarVelocity

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> q = cx.PolarPosition(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))
        >>> q.norm()
        Distance(Array(3., dtype=float32), unit='kpc')

        """
        return self.r


@final
class PolarVelocity(AbstractVelocity2D):
    """Polar differential representation."""

    d_r: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Radial speed :math:`dr/dt \in [-\infty,+\infty]`."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Polar angular speed :math:`d\phi/dt \in [-\infty,+\infty]`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[PolarPosition]:
        return PolarPosition


@final
class PolarAcceleration(AbstractAcceleration2D):
    """Polar acceleration representation."""

    d2_r: ct.BatchableAcc = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty,+\infty]`."""

    d2_phi: ct.BatchableAngularAcc = eqx.field(
        converter=partial(Quantity["angular acceleration"].constructor, dtype=float)
    )
    r"""Polar angular acceleration :math:`d^2\phi/dt^2 \in [-\infty,+\infty]`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[PolarVelocity]:
        return PolarVelocity
