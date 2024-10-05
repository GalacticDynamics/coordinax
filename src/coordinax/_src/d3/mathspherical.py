"""Built-in vector classes."""

__all__ = [
    "MathSphericalPosition",
    "MathSphericalVelocity",
    "MathSphericalAcceleration",
]

from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from quax import register

import quaxed.lax as qlax
import quaxed.numpy as jnp
from dataclassish import replace
from dataclassish.converters import Unless
from unxt import AbstractDistance, AbstractQuantity, Distance, Quantity

import coordinax._src.typing as ct
from .base_spherical import (
    AbstractSphericalAcceleration,
    AbstractSphericalPosition,
    AbstractSphericalVelocity,
    _180d,
    _360d,
)
from coordinax._src.checks import (
    check_azimuth_range,
    check_polar_range,
    check_r_non_negative,
)
from coordinax._src.converters import converter_azimuth_to_range
from coordinax._src.utils import classproperty


@final
class MathSphericalPosition(AbstractSphericalPosition):
    """Spherical vector representation.

    .. note::

        This class follows the Mathematics conventions.

    Parameters
    ----------
    r : Distance
        Radial distance r (slant distance to origin),
    theta : Quantity['angle']
        Azimuthal angle [0, 360) [deg] where 0 is the x-axis.
    phi : Quantity['angle']
        Polar angle [0, 180] [deg] where 0 is the z-axis.

    """

    r: ct.BatchableDistance = eqx.field(
        converter=Unless(AbstractDistance, partial(Distance.constructor, dtype=float))
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    theta: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_azimuth_to_range(
            Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Azimuthal angle :math:`\theta \in [0,360)`."""

    phi: ct.BatchableAngle = eqx.field(
        converter=partial(Quantity["angle"].constructor, dtype=float)
    )
    r"""Inclination angle :math:`\phi \in [0,180]`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        check_r_non_negative(self.r)
        check_azimuth_range(self.theta)
        check_polar_range(self.phi)

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["MathSphericalVelocity"]:
        return MathSphericalVelocity

    @partial(eqx.filter_jit, inline=True)
    def norm(self) -> ct.BatchableDistance:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> s = cx.MathSphericalPosition(r=Quantity(3, "kpc"),
        ...                              theta=Quantity(90, "deg"),
        ...                              phi=Quantity(0, "deg"))
        >>> s.norm()
        Distance(Array(3., dtype=float32), unit='kpc')

        """
        return self.r


@MathSphericalPosition.constructor._f.register  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: type[MathSphericalPosition],
    *,
    r: AbstractQuantity,
    theta: AbstractQuantity,
    phi: AbstractQuantity,
) -> MathSphericalPosition:
    """Construct MathSphericalPosition, allowing for out-of-range values.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    Let's start with a valid input:

    >>> cx.MathSphericalPosition.constructor(r=Quantity(3, "kpc"),
    ...                                      theta=Quantity(90, "deg"),
    ...                                      phi=Quantity(0, "deg"))
    MathSphericalPosition(
      r=Distance(value=f32[], unit=Unit("kpc")),
      theta=Quantity[...](value=f32[], unit=Unit("deg")),
      phi=Quantity[...](value=f32[], unit=Unit("deg"))
    )

    The radial distance can be negative, which wraps the azimuthal angle by 180
    degrees and flips the polar angle:

    >>> vec = cx.MathSphericalPosition.constructor(r=Quantity(-3, "kpc"),
    ...                                            theta=Quantity(100, "deg"),
    ...                                            phi=Quantity(45, "deg"))
    >>> vec.r
    Distance(Array(3., dtype=float32), unit='kpc')
    >>> vec.theta
    Quantity['angle'](Array(280., dtype=float32), unit='deg')
    >>> vec.phi
    Quantity[...](Array(135., dtype=float32), unit='deg')

    The polar angle can be outside the [0, 180] deg range, causing the azimuthal
    angle to be shifted by 180 degrees:

    >>> vec = cx.MathSphericalPosition.constructor(r=Quantity(3, "kpc"),
    ...                                            theta=Quantity(0, "deg"),
    ...                                            phi=Quantity(190, "deg"))
    >>> vec.r
    Distance(Array(3., dtype=float32), unit='kpc')
    >>> vec.theta
    Quantity['angle'](Array(180., dtype=float32), unit='deg')
    >>> vec.phi
    Quantity['angle'](Array(170., dtype=float32), unit='deg')

    The azimuth can be outside the [0, 360) deg range. This is wrapped to the
    [0, 360) deg range (actually the base constructor does this):

    >>> vec = cx.MathSphericalPosition.constructor(r=Quantity(3, "kpc"),
    ...                                            theta=Quantity(365, "deg"),
    ...                                            phi=Quantity(90, "deg"))
    >>> vec.theta
    Quantity['angle'](Array(5., dtype=float32), unit='deg')

    """
    # 1) Convert the inputs
    fields = MathSphericalPosition.__dataclass_fields__
    r = fields["r"].metadata["converter"](r)
    theta = fields["theta"].metadata["converter"](theta)
    phi = fields["phi"].metadata["converter"](phi)

    # 2) handle negative distances
    r_pred = r < jnp.zeros_like(r)
    r = qlax.select(r_pred, -r, r)
    theta = qlax.select(r_pred, theta + _180d, theta)
    phi = qlax.select(r_pred, _180d - phi, phi)

    # 3) Handle polar angle outside of [0, 180] degrees
    phi = jnp.mod(phi, _360d)  # wrap to [0, 360) deg
    phi_pred = phi < _180d
    phi = qlax.select(phi_pred, phi, _360d - phi)
    theta = qlax.select(phi_pred, theta, theta + _180d)

    # 4) Construct. This also handles the azimuthal angle wrapping
    return cls(r=r, theta=theta, phi=phi)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_p_vmsph(
    lhs: ArrayLike, rhs: MathSphericalPosition, /
) -> MathSphericalPosition:
    """Scale the polar position by a scalar.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import quaxed.numpy as jnp

    >>> v = cx.MathSphericalPosition(r=Quantity(3, "kpc"),
    ...                              theta=Quantity(90, "deg"),
    ...                              phi=Quantity(0, "deg"))

    >>> jnp.linalg.vector_norm(v, axis=-1)
    Quantity['length'](Array(3., dtype=float32), unit='kpc')

    >>> nv = jnp.multiply(2, v)
    >>> nv
    MathSphericalPosition(
      r=Distance(value=f32[], unit=Unit("kpc")),
      theta=Quantity[...](value=f32[], unit=Unit("deg")),
      phi=Quantity[...](value=f32[], unit=Unit("deg"))
    )
    >>> nv.r
    Distance(Array(6., dtype=float32), unit='kpc')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )
    # Scale the radial distance
    return replace(rhs, r=lhs * rhs.r)


##############################################################################


@final
class MathSphericalVelocity(AbstractSphericalVelocity):
    """Spherical differential representation."""

    d_r: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Radial speed :math:`dr/dt \in [-\infty, \infty]."""

    d_theta: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Azimuthal speed :math:`d\theta/dt \in [-\infty, \infty]."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Inclination speed :math:`d\phi/dt \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[MathSphericalPosition]:
        return MathSphericalPosition

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["MathSphericalAcceleration"]:
        return MathSphericalAcceleration


##############################################################################


@final
class MathSphericalAcceleration(AbstractSphericalAcceleration):
    """Spherical acceleration representation."""

    d2_r: ct.BatchableAcc = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""Radial acceleration :math:`d^2r/dt^2 \in [-\infty, \infty]."""

    d2_theta: ct.BatchableAngularAcc = eqx.field(
        converter=partial(Quantity["angular acceleration"].constructor, dtype=float)
    )
    r"""Azimuthal acceleration :math:`d^2\theta/dt^2 \in [-\infty, \infty]."""

    d2_phi: ct.BatchableAngularAcc = eqx.field(
        converter=partial(Quantity["angular acceleration"].constructor, dtype=float)
    )
    r"""Inclination acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[MathSphericalVelocity]:
        return MathSphericalVelocity
