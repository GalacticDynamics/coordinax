"""Built-in vector classes."""

__all__ = [
    "TwoSpherePosition",
    "TwoSphereVelocity",
    "TwoSphereAcceleration",
]

from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx

from unxt import Quantity

import coordinax._src.typing as ct
from .base import AbstractAcceleration2D, AbstractPosition2D, AbstractVelocity2D
from coordinax._src.checks import check_azimuth_range, check_polar_range
from coordinax._src.converters import converter_azimuth_to_range
from coordinax._src.utils import classproperty


# TODO: should this be named TwoSphericalPosition or S(phere)(ical)2Position
@final
class TwoSpherePosition(AbstractPosition2D):
    r"""Position on the 2-Sphere.

    The space of coordinates on the unit sphere is called the 2-sphere or $S^2$.
    It is a two-dimensional surface embedded in three-dimensional space, defined
    by the set of all points at a unit distance from a central point.
    Mathematically, this is:

    $$ S^2 = \{ \mathbf{x} \in \mathbb{R}^3 |  \|\mathbf{x}\| = 1 \}. $$

    .. note::

        This class follows the Physics conventions (ISO 80000-2:2019).

    Parameters
    ----------
    theta : Quantity['angle']
        Polar angle [0, 180] [deg] where 0 is the z-axis.
    phi : Quantity['angle']
        Azimuthal angle [0, 360) [deg] where 0 is the x-axis.

    See Also
    --------
    `coordinax.SphericalPosition`
        The counterpart in $R^3$, adding the polar distance coordinate $r$.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    We can construct a 2-spherical coordinate:

    >>> s2 = cx.TwoSpherePosition(theta=Quantity(0, "deg"), phi=Quantity(180, "deg"))

    This coordinate has corresponding velocity class:

    >>> s2.differential_cls
    <class 'coordinax._src.d2.spherical.TwoSphereVelocity'>

    """

    theta: ct.BatchableAngle = eqx.field(
        converter=partial(Quantity["angle"].constructor, dtype=float)
    )
    r"""Inclination angle :math:`\theta \in [0,180]`."""

    phi: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_azimuth_to_range(
            Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        check_polar_range(self.theta)
        check_azimuth_range(self.phi)

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["TwoSphereVelocity"]:
        return TwoSphereVelocity


#####################################################################


@final
class TwoSphereVelocity(AbstractVelocity2D):
    r"""Velocity on the 2-Sphere.

    The space of coordinates on the unit sphere is called the 2-sphere or $S^2$.
    It is a two-dimensional surface embedded in three-dimensional space, defined
    by the set of all points at a unit distance from a central point.
    Mathematically, this is:

    $$ S^2 = \{ \mathbf{x} \in \mathbb{R}^3 |  \|\mathbf{x}\| = 1 \}. $$

    .. note::

        This class follows the Physics conventions (ISO 80000-2:2019).

    Parameters
    ----------
    d_theta : Quantity['angular speed']
        Inclination speed $`d\theta/dt \in [-\infty, \infty]$.
    d_phi : Quantity['angular speed']
        Azimuthal speed $d\phi/dt \in [-\infty, \infty]$.

    See Also
    --------
    `coordinax.SphericalVelocity`
        The counterpart in $R^3$, adding the polar distance coordinate $d_r$.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    We can construct a 2-spherical velocity:

    >>> s2 = cx.TwoSphereVelocity(d_theta=Quantity(0, "deg/s"),
    ...                           d_phi=Quantity(2, "deg/s"))

    This coordinate has corresponding position and acceleration class:

    >>> s2.integral_cls
    <class 'coordinax._src.d2.spherical.TwoSpherePosition'>

    >>> s2.differential_cls
    <class 'coordinax._src.d2.spherical.TwoSphereAcceleration'>

    """

    d_theta: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Inclination speed :math:`d\theta/dt \in [-\infty, \infty]."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[TwoSpherePosition]:
        return TwoSpherePosition

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["TwoSphereAcceleration"]:
        return TwoSphereAcceleration


@final
class TwoSphereAcceleration(AbstractAcceleration2D):
    r"""Velocity on the 2-Sphere.

    The space of coordinates on the unit sphere is called the 2-sphere or $S^2$.
    It is a two-dimensional surface embedded in three-dimensional space, defined
    by the set of all points at a unit distance from a central point.
    Mathematically, this is:

    $$ S^2 = \{ \mathbf{x} \in \mathbb{R}^3 |  \|\mathbf{x}\| = 1 \}. $$

    .. note::

        This class follows the Physics conventions (ISO 80000-2:2019).

    Parameters
    ----------
    d2_theta : Quantity['angular acceleration']
        Inclination acceleration $`d^2\theta/dt^2 \in [-\infty, \infty]$.
    d2_phi : Quantity['angular acceleration']
        Azimuthal acceleration $d^2\phi/dt^2 \in [-\infty, \infty]$.

    See Also
    --------
    `coordinax.SphericalAcceleration`
        The counterpart in $R^3$, adding the polar distance coordinate $d_r$.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    We can construct a 2-spherical acceleration:

    >>> s2 = cx.TwoSphereAcceleration(d2_theta=Quantity(0, "deg/s2"),
    ...                               d2_phi=Quantity(2, "deg/s2"))

    This coordinate has corresponding velocity class:

    >>> s2.integral_cls
    <class 'coordinax._src.d2.spherical.TwoSphereVelocity'>

    """

    d2_theta: ct.BatchableAngularAcc = eqx.field(
        converter=partial(Quantity["angular acceleration"].constructor, dtype=float)
    )
    r"""Inclination acceleration :math:`d^2\theta/dt^2 \in [-\infty, \infty]."""

    d2_phi: ct.BatchableAngularAcc = eqx.field(
        converter=partial(Quantity["angular acceleration"].constructor, dtype=float)
    )
    r"""Azimuthal acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[TwoSphereVelocity]:
        return TwoSphereVelocity
