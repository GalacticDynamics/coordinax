"""Built-in vector classes."""

__all__ = [
    "MathSphericalAcc",
    "MathSphericalPos",
    "MathSphericalVel",
]

from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from plum import dispatch
from quax import register

import quaxed.numpy as jnp
from dataclassish import replace
from dataclassish.converters import Unless
from unxt.quantity import AbstractQuantity, Quantity

import coordinax._src.typing as ct
from .base_spherical import (
    AbstractSphericalAcc,
    AbstractSphericalPos,
    AbstractSphericalVel,
    _180d,
    _360d,
)
from coordinax._src.angles import Angle, BatchableAngle
from coordinax._src.distances import AbstractDistance, BatchableDistance, Distance
from coordinax._src.utils import classproperty
from coordinax._src.vectors import checks
from coordinax._src.vectors.converters import converter_azimuth_to_range


@final
class MathSphericalPos(AbstractSphericalPos):
    """Spherical vector representation.

    .. note::

        This class follows the Mathematics conventions.

    Parameters
    ----------
    r : `coordinax.Distance`
        Radial distance r (slant distance to origin),
    theta : `coordinax.angle.Angle`
        Azimuthal angle [0, 360) [deg] where 0 is the x-axis.
    phi : `coordinax.angle.Angle`
        Polar angle [0, 180] [deg] where 0 is the z-axis.

    """

    r: BatchableDistance = eqx.field(converter=Unless(AbstractDistance, Distance.from_))
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    theta: BatchableAngle = eqx.field(
        converter=Unless(Angle, lambda x: converter_azimuth_to_range(Angle.from_(x)))
    )
    r"""Azimuthal angle, generally :math:`\theta \in [0,360)`."""

    phi: BatchableAngle = eqx.field(converter=Angle.from_)
    r"""Inclination angle :math:`\phi \in [0,180]`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        checks.check_r_non_negative(self.r)
        checks.check_polar_range(self.phi)

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["MathSphericalVel"]:
        return MathSphericalVel

    @partial(eqx.filter_jit, inline=True)
    def norm(self) -> BatchableDistance:
        """Return the norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> s = cx.vecs.MathSphericalPos(r=u.Quantity(3, "km"),
        ...                              theta=u.Quantity(90, "deg"),
        ...                              phi=u.Quantity(0, "deg"))
        >>> s.norm()
        Distance(Array(3, dtype=int32, ...), unit='km')

        """
        return self.r


@dispatch  # type: ignore[misc]
def vector(
    cls: type[MathSphericalPos],
    *,
    r: AbstractQuantity,
    theta: AbstractQuantity,
    phi: AbstractQuantity,
) -> MathSphericalPos:
    """Construct MathSphericalPos, allowing for out-of-range values.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Let's start with a valid input:

    >>> vec = cx.vecs.MathSphericalPos.from_(r=u.Quantity(3, "km"),
    ...                                      theta=u.Quantity(90, "deg"),
    ...                                      phi=u.Quantity(0, "deg"))
    >>> print(vec)
    <MathSphericalPos (r[km], theta[deg], phi[deg])
        [ 3 90  0]>

    The radial distance can be negative, which wraps the azimuthal angle by 180
    degrees and flips the polar angle:

    >>> vec = cx.vecs.MathSphericalPos.from_(r=u.Quantity(-3, "km"),
    ...                                      theta=u.Quantity(100, "deg"),
    ...                                      phi=u.Quantity(45, "deg"))
    >>> print(vec)
    <MathSphericalPos (r[km], theta[deg], phi[deg])
        [  3 280 135]>

    The polar angle can be outside the [0, 180] deg range, causing the azimuthal
    angle to be shifted by 180 degrees:

    >>> vec = cx.vecs.MathSphericalPos.from_(r=u.Quantity(3, "km"),
    ...                                      theta=u.Quantity(0, "deg"),
    ...                                      phi=u.Quantity(190, "deg"))
    >>> print(vec)
    <MathSphericalPos (r[km], theta[deg], phi[deg])
        [  3 180 170]>

    The azimuth can be outside the [0, 360) deg range. This is wrapped to the
    [0, 360) deg range (actually the base constructor does this):

    >>> vec = cx.vecs.MathSphericalPos.from_(r=u.Quantity(3, "km"),
    ...                                      theta=u.Quantity(365, "deg"),
    ...                                      phi=u.Quantity(90, "deg"))
    >>> vec.theta
    Angle(Array(5, dtype=int32, ...), unit='deg')

    """
    # 1) Convert the inputs
    fields = MathSphericalPos.__dataclass_fields__
    r = fields["r"].metadata["converter"](r)
    theta = fields["theta"].metadata["converter"](theta)
    phi = fields["phi"].metadata["converter"](phi)

    # 2) handle negative distances
    r_pred = r < jnp.zeros_like(r)
    r = jnp.where(r_pred, -r, r)
    theta = jnp.where(r_pred, theta + _180d, theta)
    phi = jnp.where(r_pred, _180d - phi, phi)

    # 3) Handle polar angle outside of [0, 180] degrees
    phi = jnp.mod(phi, _360d)  # wrap to [0, 360) deg
    phi_pred = phi < _180d
    phi = jnp.where(phi_pred, phi, _360d - phi)
    theta = jnp.where(phi_pred, theta, theta + _180d)

    # 4) Construct. This also handles the azimuthal angle wrapping
    return cls(r=r, theta=theta, phi=phi)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_p_vmsph(lhs: ArrayLike, rhs: MathSphericalPos, /) -> MathSphericalPos:
    """Scale the polar position by a scalar.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import quaxed.numpy as jnp

    >>> v = cx.vecs.MathSphericalPos(r=u.Quantity(3, "km"),
    ...                              theta=u.Quantity(90, "deg"),
    ...                              phi=u.Quantity(0, "deg"))

    >>> jnp.linalg.vector_norm(v, axis=-1)
    Quantity['length'](Array(3., dtype=float32), unit='km')

    >>> nv = jnp.multiply(2, v)
    >>> print(nv)
    <MathSphericalPos (r[km], theta[deg], phi[deg])
        [ 6. 90.  0.]>

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )
    # Scale the radial distance
    return replace(rhs, r=lhs * rhs.r)


##############################################################################


@final
class MathSphericalVel(AbstractSphericalVel):
    """Spherical differential representation."""

    d_r: ct.BatchableSpeed = eqx.field(converter=Quantity["speed"].from_)
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    d_theta: ct.BatchableAngularSpeed = eqx.field(
        converter=Quantity["angular speed"].from_
    )
    r"""Azimuthal speed :math:`d\theta/dt \in [-\infty, \infty]."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=Quantity["angular speed"].from_
    )
    r"""Inclination speed :math:`d\phi/dt \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[MathSphericalPos]:
        return MathSphericalPos

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["MathSphericalAcc"]:
        return MathSphericalAcc


##############################################################################


@final
class MathSphericalAcc(AbstractSphericalAcc):
    """Spherical acceleration representation."""

    d2_r: ct.BatchableAcc = eqx.field(converter=Quantity["acceleration"].from_)
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty, \infty]."""

    d2_theta: ct.BatchableAngularAcc = eqx.field(
        converter=Quantity["angular acceleration"].from_
    )
    r"""Azimuthal acceleration :math:`d^2\theta/dt^2 \in [-\infty, \infty]."""

    d2_phi: ct.BatchableAngularAcc = eqx.field(
        converter=Quantity["angular acceleration"].from_
    )
    r"""Inclination acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[MathSphericalVel]:
        return MathSphericalVel
