"""Built-in vector classes."""

__all__ = ["TwoSphereAcc", "TwoSpherePos", "TwoSphereVel"]

from typing import final

import equinox as eqx

import unxt as u
from dataclassish.converters import Unless

import coordinax._src.typing as ct
from .base import AbstractAcc2D, AbstractPos2D, AbstractVel2D
from coordinax._src.angles import Angle, BatchableAngle
from coordinax._src.vectors.checks import check_polar_range
from coordinax._src.vectors.converters import converter_azimuth_to_range


@final
class TwoSpherePos(AbstractPos2D):
    r"""Pos on the 2-Sphere.

    The space of coordinates on the unit sphere is called the 2-sphere or $S^2$.
    It is a two-dimensional surface embedded in three-dimensional space, defined
    by the set of all points at a unit distance from a central point.
    Mathematically, this is:

    $$ S^2 = \{ \mathbf{x} \in \mathbb{R}^3 |  \|\mathbf{x}\| = 1 \}. $$

    .. note::

        This class follows the Physics conventions (ISO 80000-2:2019).

    Parameters
    ----------
    theta : `coordinax.angle.Angle`
        Polar angle [0, 180] [deg] where 0 is the z-axis.
    phi : `coordinax.angle.Angle`
        Azimuthal angle [0, 360) [deg] where 0 is the x-axis.

    See Also
    --------
    `coordinax.vecs.SphericalPos`
        The counterpart in $R^3$, adding the polar distance coordinate $r$.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can construct a 2-spherical coordinate:

    >>> s2 = cx.vecs.TwoSpherePos(theta=u.Quantity(0, "deg"),
    ...                           phi=u.Quantity(180, "deg"))

    This coordinate has corresponding velocity class:

    >>> s2.time_derivative_cls
    <class 'coordinax...TwoSphereVel'>

    """

    theta: BatchableAngle = eqx.field(converter=Angle.from_)
    r"""Inclination angle :math:`\theta \in [0,180]`."""

    phi: BatchableAngle = eqx.field(
        converter=Unless(Angle, lambda x: converter_azimuth_to_range(Angle.from_(x)))
    )
    r"""Azimuthal angle :math:`\phi \in [0,360)`."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        check_polar_range(self.theta)


@final
class TwoSphereVel(AbstractVel2D):
    r"""Vel on the 2-Sphere.

    The space of coordinates on the unit sphere is called the 2-sphere or $S^2$.
    It is a two-dimensional surface embedded in three-dimensional space, defined
    by the set of all points at a unit distance from a central point.
    Mathematically, this is:

    $$ S^2 = \{ \mathbf{x} \in \mathbb{R}^3 |  \|\mathbf{x}\| = 1 \}. $$

    .. note::

        This class follows the Physics conventions (ISO 80000-2:2019).

    Parameters
    ----------
    theta : Quantity['angular speed']
        Inclination speed $`d\theta/dt \in [-\infty, \infty]$.
    phi : Quantity['angular speed']
        Azimuthal speed $d\phi/dt \in [-\infty, \infty]$.

    See Also
    --------
    `coordinax.vecs.SphericalVel`
        The counterpart in $R^3$, adding the polar distance coordinate $d_r$.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can construct a 2-spherical velocity:

    >>> s2 = cx.vecs.TwoSphereVel(theta=u.Quantity(0, "deg/s"),
    ...                           phi=u.Quantity(2, "deg/s"))

    This coordinate has corresponding position and acceleration class:

    >>> s2.time_antiderivative_cls
    <class 'coordinax...TwoSpherePos'>

    >>> s2.time_derivative_cls
    <class 'coordinax...TwoSphereAcc'>

    """

    theta: ct.BatchableAngularSpeed = eqx.field(
        converter=u.Quantity["angular speed"].from_
    )
    r"""Inclination speed :math:`d\theta/dt \in [-\infty, \infty]."""

    phi: ct.BatchableAngularSpeed = eqx.field(
        converter=u.Quantity["angular speed"].from_
    )
    r"""Azimuthal speed :math:`d\phi/dt \in [-\infty, \infty]."""


@final
class TwoSphereAcc(AbstractAcc2D):
    r"""Vel on the 2-Sphere.

    The space of coordinates on the unit sphere is called the 2-sphere or $S^2$.
    It is a two-dimensional surface embedded in three-dimensional space, defined
    by the set of all points at a unit distance from a central point.
    Mathematically, this is:

    $$ S^2 = \{ \mathbf{x} \in \mathbb{R}^3 |  \|\mathbf{x}\| = 1 \}. $$

    .. note::

        This class follows the Physics conventions (ISO 80000-2:2019).

    Parameters
    ----------
    theta : Quantity['angular acceleration']
        Inclination acceleration $`d^2\theta/dt^2 \in [-\infty, \infty]$.
    phi : Quantity['angular acceleration']
        Azimuthal acceleration $d^2\phi/dt^2 \in [-\infty, \infty]$.

    See Also
    --------
    `coordinax.vecs.SphericalAcc`
        The counterpart in $R^3$, adding the polar distance coordinate $d_r$.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can construct a 2-spherical acceleration:

    >>> s2 = cx.vecs.TwoSphereAcc(theta=u.Quantity(0, "deg/s2"),
    ...                           phi=u.Quantity(2, "deg/s2"))

    This coordinate has corresponding velocity class:

    >>> s2.time_antiderivative_cls
    <class 'coordinax...TwoSphereVel'>

    """

    theta: ct.BatchableAngularAcc = eqx.field(
        converter=u.Quantity["angular acceleration"].from_
    )
    r"""Inclination acceleration :math:`d^2\theta/dt^2 \in [-\infty, \infty]."""

    phi: ct.BatchableAngularAcc = eqx.field(
        converter=u.Quantity["angular acceleration"].from_
    )
    r"""Azimuthal acceleration :math:`d^2\phi/dt^2 \in [-\infty, \infty]."""
