"""Astronomy reference frames."""

__all__ = ["Galactocentric"]


from typing import ClassVar, TypeAlias, final

import equinox as eqx
from jaxtyping import Array, Shaped

import unxt as u

from coordinax._src.angles import Angle
from coordinax._src.distances import Distance
from coordinax._src.frames import AbstractReferenceFrame
from coordinax._src.vectors.d3 import CartesianVel3D, LonLatSphericalPos

ScalarAngle: TypeAlias = Shaped[u.Quantity["angle"] | Angle, ""]
RotationMatrix: TypeAlias = Shaped[Array, "3 3"]
LengthVector: TypeAlias = Shaped[u.Quantity["length"], "3"] | Shaped[Distance, "3"]
VelocityVector: TypeAlias = Shaped[u.Quantity["speed"], "3"]


@final
class Galactocentric(AbstractReferenceFrame):
    """Reference frame centered at the Galactic center.

    Based on the Astropy implementation of the Galactocentric frame.

    Examples
    --------
    >>> import coordinax as cx
    >>> frame = cx.frames.Galactocentric()
    >>> frame
    Galactocentric(
      galcen=LonLatSphericalPos( ... ),
      roll=Quantity[...](value=weak_i32[], unit=Unit("deg")),
      z_sun=Quantity[...](value=weak_f32[], unit=Unit("pc")),
      galcen_v_sun=CartesianVel3D( ... )
    )

    """

    #: RA, Dec, and distance of the Galactic center from an ICRS origin.
    #: ra, dec: https://ui.adsabs.harvard.edu/abs/2004ApJ...616..872R
    #: distance: https://ui.adsabs.harvard.edu/abs/2018A%26A...615L..15G
    galcen: LonLatSphericalPos = eqx.field(
        converter=LonLatSphericalPos.from_,
        default_factory=lambda: LonLatSphericalPos(
            lon=Angle(266.4051, "deg"),
            lat=Angle(-28.936175, "degree"),
            distance=Distance(8.122, "kpc"),
        ),
    )

    #: Rotation angle of the Galactic center from the ICRS x-axis.
    roll: u.Quantity["angle"] = eqx.field(
        converter=u.Quantity["angle"].from_, default=u.Quantity(0, "deg")
    )

    #: Distance from the Sun to the Galactic center.
    #: https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1417B
    z_sun: u.Quantity["length"] = eqx.field(
        converter=u.Quantity["length"].from_, default=u.Quantity(20.8, "pc")
    )

    #: Velocity of the Sun in the Galactic center frame.
    #: https://ui.adsabs.harvard.edu/abs/2018RNAAS...2..210D
    #: https://ui.adsabs.harvard.edu/abs/2018A%26A...615L..15G
    #: https://ui.adsabs.harvard.edu/abs/2004ApJ...616..872R
    galcen_v_sun: CartesianVel3D = eqx.field(
        converter=CartesianVel3D.from_,
        default_factory=lambda: CartesianVel3D.from_([12.9, 245.6, 7.78], "km/s"),
    )

    # --------

    #: The angle between the Galactic center and the ICRS x-axis.
    roll0: ClassVar[ScalarAngle] = u.Quantity(58.5986320306, "degree")
