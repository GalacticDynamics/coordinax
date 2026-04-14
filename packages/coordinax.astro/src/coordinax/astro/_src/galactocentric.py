"""Astronomy reference frames."""

__all__ = ("Galactocentric",)


from jaxtyping import Shaped
from typing import Any, ClassVar, TypeAlias, final

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.vectors as cxv
from .base_frame import AbstractSpaceFrame
from coordinax.distances import Distance

ScalarAngle: TypeAlias = Shaped[u.Q["angle"] | u.Angle, ""]  # ty: ignore[unresolved-reference]


galcen_default = cxv.Point.from_(
    {
        "lon": u.Angle(jnp.array(266.4051), "deg"),
        "lat": u.Angle(jnp.array(-28.936175), "degree"),
        "distance": Distance(jnp.array(8.122), "kpc"),
    },
    cxc.lonlat_sph3d,
    cxr.point,
)
# galcen_v_sun_default = cxv.Point.from_([12.9, 245.6, 7.78], "km/s")


@final
class Galactocentric(AbstractSpaceFrame):
    """Reference frame centered at the Galactic center.

    Based on the Astropy implementation of the Galactocentric frame.

    Examples
    --------
    >>> import coordinax.astro as cxastro
    >>> frame = cxastro.Galactocentric()
    >>> frame
    Galactocentric()

    """

    #: RA, Dec, and distance of the Galactic center from an ICRS origin.
    #: ra, dec: https://ui.adsabs.harvard.edu/abs/2004ApJ...616..872R
    #: distance: https://ui.adsabs.harvard.edu/abs/2018A%26A...615L..15G
    galcen: cxv.Point[cxc.LonLatSpherical3D, Any] = eqx.field(
        converter=cxv.Point[cxc.LonLatSpherical3D, Any].from_,
        default=galcen_default,
    )

    #: Rotation angle of the Galactic center from the ICRS x-axis.
    roll: ScalarAngle = eqx.field(
        converter=Unless(u.Angle, u.Angle.from_),
        default=u.Angle(jnp.array(0), "deg"),
    )

    #: Distance from the Sun to the Galactic midplane.
    #: https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1417B
    z_sun: u.Q["length"] = eqx.field(  # ty: ignore[unresolved-reference]
        converter=Unless(u.Q["length"], u.Q["length"].from_),
        default=u.Q(jnp.array(20.8), "pc"),
    )

    # #: Velocity of the Sun in the Galactic center frame.
    # #: https://ui.adsabs.harvard.edu/abs/2018RNAAS...2..210D
    # #: https://ui.adsabs.harvard.edu/abs/2018A%26A...615L..15G
    # #: https://ui.adsabs.harvard.edu/abs/2004ApJ...616..872R
    # galcen_v_sun: cxv.Vector[cxc.Cart3D, cxr.TangentGeometry] = eqx.field(
    #     converter=cxv.Vector[cxc.Cart3D, cxr.TangentGeometry].from_,
    #     default=galcen_v_sun_default,
    # )

    # --------

    #: The angle between the Galactic center and the ICRS x-axis.
    roll0: ClassVar[ScalarAngle] = eqx.field(
        default=u.Angle(jnp.array(58.5986320306), "deg"),
        converter=Unless(u.Angle, u.Angle.from_),
    )
