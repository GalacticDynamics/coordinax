"""Astronomy reference frames."""

__all__ = ("Galactocentric",)


from jaxtyping import Array, Shaped
from typing import ClassVar, TypeAlias, final

import equinox as eqx

import unxt as u
from dataclassish.converters import Unless

import coordinax.charts as cxc
import coordinax.roles as cxr
from .base import AbstractSpaceFrame
from coordinax._src.distances import Distance
from coordinax._src.objects.vector import Vector

ScalarAngle: TypeAlias = Shaped[u.Q["angle"] | u.Angle, ""]
RotationMatrix: TypeAlias = Shaped[Array, "3 3"]
LengthVector: TypeAlias = Shaped[u.Q["length"], "3"] | Shaped[Distance, "3"]
VelocityVector: TypeAlias = Shaped[u.Q["speed"], "3"]


@final
class Galactocentric(AbstractSpaceFrame):
    """Reference frame centered at the Galactic center.

    Based on the Astropy implementation of the Galactocentric frame.

    Examples
    --------
    >>> import coordinax as cx
    >>> frame = cx.frames.Galactocentric()
    >>> frame
    Galactocentric(
      galcen=LonLatSpherical3D( ... ),
      roll=Quantity(weak_i32[], unit='deg'),
      z_sun=Quantity(weak_f32[], unit='pc'),
      galcen_v_sun=CartVel3D( ... )
    )

    """

    #: RA, Dec, and distance of the Galactic center from an ICRS origin.
    #: ra, dec: https://ui.adsabs.harvard.edu/abs/2004ApJ...616..872R
    #: distance: https://ui.adsabs.harvard.edu/abs/2018A%26A...615L..15G
    galcen: Vector[cxc.LonLatSpherical3D, cxr.Point] = eqx.field(
        converter=Vector[cxc.LonLatSpherical3D, cxr.Point].from_,
        default_factory=lambda: Vector(
            {
                "lon": u.Angle(266.4051, "deg"),
                "lat": u.Angle(-28.936175, "degree"),
                "distance": Distance(8.122, "kpc"),
            },
            chart=cxc.lonlatsph3d,
            role=cxr.point,
        ),
    )

    #: Rotation angle of the Galactic center from the ICRS x-axis.
    roll: ScalarAngle = eqx.field(
        converter=Unless(u.Angle, u.Q["angle"].from_),
        default=u.Q(0, "deg"),
    )

    #: Distance from the Sun to the Galactic center.
    #: https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1417B
    z_sun: u.Q["length"] = eqx.field(
        converter=u.Q["length"].from_, default=u.Q(20.8, "pc")
    )

    #: Velocity of the Sun in the Galactic center frame.
    #: https://ui.adsabs.harvard.edu/abs/2018RNAAS...2..210D
    #: https://ui.adsabs.harvard.edu/abs/2018A%26A...615L..15G
    #: https://ui.adsabs.harvard.edu/abs/2004ApJ...616..872R
    galcen_v_sun: Vector[cxc.Cart3D, cxr.Vel] = eqx.field(
        converter=Vector[cxc.Cart3D, cxr.Vel].from_,
        default_factory=lambda: Vector.from_([12.9, 245.6, 7.78], "km/s"),
    )

    # --------

    #: The angle between the Galactic center and the ICRS x-axis.
    roll0: ClassVar[ScalarAngle] = eqx.field(
        default=u.Q(58.5986320306, "degree"),
        converter=Unless(u.Angle, u.Q["angle"].from_),
    )
