"""Built-in vector classes."""

__all__ = ["PolarAcc", "PolarPos", "PolarVel"]

from functools import partial
from typing import final

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from quax import register

import unxt as u
from dataclassish import replace
from dataclassish.converters import Unless

import coordinax._src.typing as ct
from .base import AbstractAcc2D, AbstractPos2D, AbstractVel2D
from coordinax._src.angles import Angle, BatchableAngle
from coordinax._src.distances import AbstractDistance, BatchableDistance, Distance
from coordinax._src.utils import classproperty
from coordinax._src.vectors.checks import check_r_non_negative
from coordinax._src.vectors.converters import converter_azimuth_to_range


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

    r: BatchableDistance = eqx.field(
        converter=Unless(AbstractDistance, partial(Distance.from_, dtype=float))
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    phi: BatchableAngle = eqx.field(
        converter=Unless(
            Angle, lambda x: converter_azimuth_to_range(Angle.from_(x, dtype=float))
        )
    )
    r"""Polar angle, generally :math:`\phi \in [0,2\pi)`."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        check_r_non_negative(self.r)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["PolarVel"]:
        return PolarVel


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_p_vpolar(lhs: ArrayLike, rhs: PolarPos, /) -> PolarPos:
    """Scale the polar position by a scalar.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import quaxed.numpy as jnp

    >>> v = cx.vecs.PolarPos(r=u.Quantity(1, "m"), phi=u.Quantity(90, "deg"))

    >>> jnp.linalg.vector_norm(v, axis=-1)
    Quantity['length'](Array(1., dtype=float32), unit='m')

    >>> nv = jnp.multiply(2, v)
    >>> nv
    PolarPos(
      r=Distance(value=f32[], unit=Unit("m")),
      phi=Angle(value=f32[], unit=Unit("deg"))
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
        converter=partial(u.Quantity["speed"].from_, dtype=float)
    )
    r"""Radial speed :math:`dr/dt \in [-\infty,+\infty]`."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(u.Quantity["angular speed"].from_, dtype=float)
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
        converter=partial(u.Quantity["acceleration"].from_, dtype=float)
    )
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty,+\infty]`."""

    d2_phi: ct.BatchableAngularAcc = eqx.field(
        converter=partial(u.Quantity["angular acceleration"].from_, dtype=float)
    )
    r"""Polar angular acceleration :math:`d^2\phi/dt^2 \in [-\infty,+\infty]`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[PolarVel]:
        return PolarVel
