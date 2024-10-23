"""Built-in vector classes."""

__all__ = [
    "PolarPos",
    "PolarVel",
    "PolarAcc",
]

from functools import partial
from typing import final

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from quax import register

from dataclassish import replace
from dataclassish.converters import Unless
from unxt import Quantity

import coordinax._src.typing as ct
from .base import AbstractAcc2D, AbstractPos2D, AbstractVel2D
from coordinax._src.checks import check_azimuth_range, check_r_non_negative
from coordinax._src.converters import converter_azimuth_to_range
from coordinax._src.distance import AbstractDistance, Distance
from coordinax._src.utils import classproperty


@final
class PolarPos(AbstractPos2D):
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
        converter=Unless(AbstractDistance, partial(Distance.from_, dtype=float))
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    phi: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_azimuth_to_range(
            Quantity["angle"].from_(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Polar angle :math:`\phi \in [0,2\pi)`."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        check_r_non_negative(self.r)
        check_azimuth_range(self.phi)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["PolarVel"]:
        return PolarVel


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_p_vpolar(lhs: ArrayLike, rhs: PolarPos, /) -> PolarPos:
    """Scale the polar position by a scalar.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import quaxed.numpy as jnp

    >>> v = cx.PolarPos(r=Quantity(1, "m"), phi=Quantity(90, "deg"))

    >>> jnp.linalg.vector_norm(v, axis=-1)
    Quantity['length'](Array(1., dtype=float32), unit='m')

    >>> nv = jnp.multiply(2, v)
    >>> nv
    PolarPos(
      r=Distance(value=f32[], unit=Unit("m")),
      phi=Quantity[...](value=f32[], unit=Unit("deg"))
    )
    >>> nv.r
    Distance(Array(2., dtype=float32), unit='m')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )
    # Scale the radial distance
    return replace(rhs, r=lhs * rhs.r)


#####################################################################


@final
class PolarVel(AbstractVel2D):
    """Polar differential representation."""

    d_r: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].from_, dtype=float)
    )
    r"""Radial speed :math:`dr/dt \in [-\infty,+\infty]`."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].from_, dtype=float)
    )
    r"""Polar angular speed :math:`d\phi/dt \in [-\infty,+\infty]`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[PolarPos]:
        return PolarPos

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["PolarAcc"]:
        return PolarAcc


@final
class PolarAcc(AbstractAcc2D):
    """Polar acceleration representation."""

    d2_r: ct.BatchableAcc = eqx.field(
        converter=partial(Quantity["acceleration"].from_, dtype=float)
    )
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty,+\infty]`."""

    d2_phi: ct.BatchableAngularAcc = eqx.field(
        converter=partial(Quantity["angular acceleration"].from_, dtype=float)
    )
    r"""Polar angular acceleration :math:`d^2\phi/dt^2 \in [-\infty,+\infty]`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[PolarVel]:
        return PolarVel
