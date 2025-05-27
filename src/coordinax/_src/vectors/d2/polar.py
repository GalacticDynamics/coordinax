"""Built-in vector classes."""

__all__ = ["PolarAcc", "PolarPos", "PolarVel"]

from typing import final
from typing_extensions import override

import equinox as eqx

import unxt as u
from dataclassish.converters import Unless

import coordinax._src.custom_types as ct
from .base import AbstractAcc2D, AbstractPos2D, AbstractVel2D
from coordinax._src.angles import Angle, BatchableAngle
from coordinax._src.distances import AbstractDistance, BatchableDistance, Distance
from coordinax._src.vectors.checks import check_r_non_negative
from coordinax._src.vectors.converters import converter_azimuth_to_range


@final
class PolarPos(AbstractPos2D):
    r"""Polar vector representation."""

    r: BatchableDistance = eqx.field(converter=Unless(AbstractDistance, Distance.from_))
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    phi: BatchableAngle = eqx.field(
        converter=Unless(Angle, lambda x: converter_azimuth_to_range(Angle.from_(x)))
    )
    r"""Polar angle, generally :math:`\phi \in [0,2\pi)`.

    We use the symbol `phi` to adhere to the ISO standard 31-11.
    """

    def __check_init__(self) -> None:
        """Check the initialization."""
        check_r_non_negative(self.r)

    @override
    def norm(self) -> BatchableDistance:
        """Return the norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> vec = cx.vecs.PolarPos(r=u.Quantity(1, "m"),
        ...                        phi=u.Quantity(90, "deg"))
        >>> vec.norm()
        Distance(Array(1, dtype=int32, ...), unit='m')

        """
        return self.r


@final
class PolarVel(AbstractVel2D):
    """Polar Velocity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vev = cx.vecs.PolarVel(r=u.Quantity(1, "m/s"), phi=u.Quantity(90, "deg/s"))
    >>> print(vev)
    <PolarVel: (r[m / s], phi[deg / s])
        [ 1 90]>

    """

    r: ct.BBtSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Radial speed :math:`dr/dt \in [-\infty,+\infty]`."""

    phi: ct.BBtAngularSpeed = eqx.field(converter=u.Quantity["angular speed"].from_)
    r"""Polar angular speed :math:`d\phi/dt \in [-\infty,+\infty]`."""


#####################################################################


@final
class PolarAcc(AbstractAcc2D):
    """Polar acceleration.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> acc = cx.vecs.PolarAcc(r=u.Quantity(1, "m/s2"),
    ...                        phi=u.Quantity(3, "deg/s2"))
    >>> print(acc)
    <PolarAcc: (r[m / s2], phi[deg / s2])
        [1 3]>

    """

    r: ct.BBtAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty,+\infty]`."""

    phi: ct.BBtAngularAcc = eqx.field(
        converter=u.Quantity["angular acceleration"].from_
    )
    r"""Polar angular acceleration :math:`d^2\phi/dt^2 \in [-\infty,+\infty]`."""
