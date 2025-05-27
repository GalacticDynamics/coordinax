"""Built-in vector classes."""

__all__ = ["ProlateSpheroidalAcc", "ProlateSpheroidalPos", "ProlateSpheroidalVel"]

from dataclasses import KW_ONLY
from typing import final

import equinox as eqx
from jaxtyping import Shaped

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless

import coordinax._src.custom_types as ct
from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from coordinax._src.angles import Angle, BatchableAngleQ
from coordinax._src.vectors import checks
from coordinax._src.vectors.base import VectorAttribute
from coordinax._src.vectors.converters import converter_azimuth_to_range


@final
class ProlateSpheroidalPos(AbstractPos3D):
    r"""Prolate spheroidal coordinates as defined by Dejonghe & de Zeeuw 1988.

    Note that valid coordinates have: $- mu >= \Delta^2 - |nu| <= \Delta^2 -
    \Delta > 0$

    Parameters
    ----------
    mu
        The spheroidal mu coordinate. This is called `lambda` by Dejonghe & de
        Zeeuw. Surfaces of constant mu are ellipsoids with foci at the origin
        and at (`Delta`, 0, 0).
    nu
        The spheroidal nu coordinate. Surfaces of constant nu are hyperboloids
        of two sheets.
    phi
        Azimuthal angle [0, 360) [deg] where 0 is the x-axis.
    Delta
        The focal length of the coordinate system. Must be > 0.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> vec = cxv.ProlateSpheroidalPos(
    ...     mu=u.Quantity(3.0, "km2"),
    ...     nu=u.Quantity(0.5, "km2"),
    ...     phi=u.Quantity(0.25, "rad"),
    ...     Delta=u.Quantity(1.5, "km"),
    ... )
    >>> print(vec)
    <ProlateSpheroidalPos: (mu[km2], nu[km2], phi[rad])
     Delta=Quantity(1.5, unit='km')
        [3.   0.5  0.25]>

    This fails with a zero or negative Delta:

    >>> try: vec = cxv.ProlateSpheroidalPos(
    ...     mu=u.Quantity(3.0, "km2"),
    ...     nu=u.Quantity(0.5, "km2"),
    ...     phi=u.Quantity(0.25, "rad"),
    ...     Delta=u.Quantity(0.0, "km"),
    ... )
    ... except Exception as e: pass

    Or with invalid mu and nu:

    >>> try: vec = cxv.ProlateSpheroidalPos(
    ...     mu=u.Quantity(0.5, "km2"),
    ...     nu=u.Quantity(0.5, "km2"),
    ...     phi=u.Quantity(0.25, "rad"),
    ...     Delta=u.Quantity(1.5, "km"),
    ... )
    ... except Exception as e: pass

    We can convert to other coordinates:

    >>> sph = vec.vconvert(cxv.SphericalPos)
    >>> print(sph)
    <SphericalPos: (r[km], theta[rad], phi[rad])
        [1.118 0.752 0.25 ]>

    However, this is generally a one-way conversion, as the focal length
    parameter `Delta` is not retained through the conversion. To convert back to
    prolate spheroidal coordinates, we need to provide the focal length again:

    >>> vec2 = sph.vconvert(cxv.ProlateSpheroidalPos, Delta=u.Quantity(1.5, "km"))
    >>> print(vec2.round(3))
    <ProlateSpheroidalPos: (mu[km2], nu[km2], phi[rad])
     Delta=Quantity(1.5, unit='km')
        [3.   0.5  0.25]>

    >>> print((vec2 - vec).vconvert(cxv.CartesianPos3D))
    <CartesianPos3D: (x, y, z) [km]
        [0. 0. 0.]>

    """

    mu: ct.BBtArea = eqx.field(converter=u.Quantity["area"].from_)
    r"""Spheroidal mu coordinate :math:`\mu \in [0,+\infty)` (called :math:`\lambda` in
     some Galactic dynamics contexts)."""

    nu: ct.BBtArea = eqx.field(converter=u.Quantity["area"].from_)
    r"""Spheroidal nu coordinate :math:`\lambda \in [-\infty,+\infty)`."""

    phi: BatchableAngleQ = eqx.field(
        converter=Unless(Angle, lambda x: converter_azimuth_to_range(Angle.from_(x)))
    )
    r"""Azimuthal angle, generally :math:`\phi \in [0,360)`."""

    _: KW_ONLY
    Delta: Shaped[u.Quantity["length"], ""] = VectorAttribute()
    """Focal length of the coordinate system."""

    def __check_init__(self) -> None:
        """Check the validity of the initialization."""
        checks.check_non_negative_non_zero(self.Delta, name="Delta")
        checks.check_greater_than_equal(
            self.mu, self.Delta**2, name="mu", comparison_name="Delta^2"
        )
        checks.check_less_than_equal(
            jnp.abs(self.nu), self.Delta**2, name="nu", comparison_name="Delta^2"
        )


@final
class ProlateSpheroidalVel(AbstractVel3D):
    r"""Prolate spheroidal differential representation.

    Parameters
    ----------
    mu
        Prolate spheroidal mu speed $d\mu/dt \in [-\infty, \infty]$.
    nu
        Prolate spheroidal nu speed $d\nu/dt \in [-\infty, \infty]$.
    phi
        Azimuthal speed $d\phi/dt \in [-\infty, \infty]$.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.CartesianPos3D.from_(u.Quantity([1, 2, 3], "kpc"))
    >>> v = cxv.CartesianVel3D.from_(u.Quantity([4, 5, 6], "km/s"))

    >>> px = x.vconvert(cxv.ProlateSpheroidalPos, Delta=u.Quantity(4, "kpc"))
    >>> pv = v.vconvert(cxv.ProlateSpheroidalVel, px)

    >>> print(pv.vconvert(cxv.CartesianVel3D, px))
    <CartesianVel3D: (x, y, z) [km / s]
        [4. 5. 6.]>

    >>> print(pv.vconvert(cxv.CartesianVel3D, x, Delta=u.Quantity(4, "kpc")))
    <CartesianVel3D: (x, y, z) [km / s]
        [4. 5. 6.]>

    """

    mu: ct.BBtKinematicFlux = eqx.field(converter=u.Quantity["diffusivity"].from_)
    r"""Prolate spheroidal mu speed $d\mu/dt \in [-\infty, \infty]$."""

    nu: ct.BBtKinematicFlux = eqx.field(converter=u.Quantity["diffusivity"].from_)
    r"""Prolate spheroidal nu speed $d\nu/dt \in [-\infty, \infty]$."""

    phi: ct.BBtAngularSpeed = eqx.field(converter=u.Quantity["angular speed"].from_)
    r"""Azimuthal speed $d\phi/dt \in [-\infty, \infty]$."""


@final
class ProlateSpheroidalAcc(AbstractAcc3D):
    r"""Prolate spheroidal acceleration representation.

    Parameters
    ----------
    mu
        Prolate spheroidal mu speed $d\mu/dt \in [-\infty, \infty]$.
    nu
        Prolate spheroidal nu speed $d\nu/dt \in [-\infty, \infty]$.
    phi
        Azimuthal speed $d\phi/dt \in [-\infty, \infty]$.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.CartesianPos3D.from_(u.Quantity([1, 2, 3], "kpc"))
    >>> v = cxv.CartesianVel3D.from_(u.Quantity([4, 5, 6], "km/s"))
    >>> a = cxv.CartesianAcc3D.from_(u.Quantity([4, 5, 6], "km/s2"))

    >>> px = x.vconvert(cxv.ProlateSpheroidalPos, Delta=u.Quantity(4, "kpc"))
    >>> pa = a.vconvert(cxv.ProlateSpheroidalAcc, v, px)

    >>> print(pa.vconvert(cxv.CartesianAcc3D, v, px))
    <CartesianAcc3D: (x, y, z) [km / s2]
        [4. 5. 6.]>

    >>> print(pa.vconvert(cxv.CartesianAcc3D, v, x, Delta=u.Quantity(4, "kpc")))
    <CartesianAcc3D: (x, y, z) [km / s2]
        [4. 5. 6.]>

    """

    mu: ct.BBtSpecificEnergy = eqx.field(converter=u.Quantity["specific energy"].from_)
    r"""Prolate spheroidal mu acceleration $d^2\mu/dt^2 \in [-\infty, \infty]$."""

    nu: ct.BBtSpecificEnergy = eqx.field(converter=u.Quantity["specific energy"].from_)
    r"""Prolate spheroidal nu acceleration $d^2\nu/dt^2 \in [-\infty, \infty]$."""

    phi: ct.BBtAngularAcc = eqx.field(
        converter=u.Quantity["angular acceleration"].from_
    )
    r"""Azimuthal acceleration $d^2\phi/dt^2 \in [-\infty, \infty]$."""
