"""Astronomy reference frames."""

__all__: tuple[str, ...] = ()


from jaxtyping import Array, Shaped
from typing import TypeAlias

import plum

import quaxed.numpy as jnp
import unxt as u

import coordinax.api as cxapi
import coordinax.ops as cxop
from .base import AbstractSpaceFrame
from .galactocentric import Galactocentric
from .icrs import ICRS
from coordinax.distances import Distance

ScalarAngle: TypeAlias = Shaped[u.Q["angle"] | u.Angle, ""]  # type: ignore[type-arg]
RotationMatrix: TypeAlias = Shaped[Array, "3 3"]
LengthVector: TypeAlias = Shaped[u.Q["length"], "3"] | Shaped[Distance, "3"]  # type: ignore[type-arg]
VelocityVector: TypeAlias = Shaped[u.Q["speed"], "3"]  # type: ignore[type-arg]


# ---------------------------------------------------------------
# Base Space-Frame Transformation

_icrs_frame = ICRS()  # type: ignore[no-untyped-call]


@plum.dispatch
def frame_transform_op(
    from_frame: AbstractSpaceFrame, to_frame: AbstractSpaceFrame, /
) -> cxop.Pipe:
    """Compute frame transformations with ICRS as the intermediary.

    Examples
    --------
    >>> import plum
    >>> import unxt as u
    >>> import coordinax as cx

    >>> class MySpaceFrame(cxf.AbstractSpaceFrame):
    ...     pass

    >>> @plum.dispatch
    ... def frame_transform_op(from_frame: MySpaceFrame, to_frame: ICRS, /) -> cxop.AbstractOperator:
    ...     return cxop.Rotate.from_euler("z", u.Q(10, "deg"))

    We can transform from `MySpaceFrame` to a Galacocentric frame, even though
    we don't have a direct transformation defined:

    >>> my_frame = MySpaceFrame()
    >>> gcf_frame = cxf.Galactocentric()

    >>> op = cxf.frame_transform_op(my_frame, gcf_frame)
    >>> op
    Pipe((
        Rotate(rotation=f32[3,3]),
        ...
    ))

    """  # noqa: E501
    fromframe_to_icrs = frame_transform_op(from_frame, _icrs_frame)
    icrs_to_toframe = frame_transform_op(_icrs_frame, to_frame)
    pipe = fromframe_to_icrs | icrs_to_toframe
    return cxapi.simplify(pipe)


# ---------------------------------------------------------------


@plum.dispatch
def frame_transform_op(from_frame: ICRS, to_frame: ICRS, /) -> cxop.Identity:
    """Return an identity operator for the ICRS->ICRS transformation.

    Examples
    --------
    >>> import coordinax.frames as cxf
    >>> icrs_frame = cxf.ICRS()
    >>> frame_op = cxf.frame_transform_op(icrs_frame, icrs_frame)
    >>> frame_op
    Identity()

    """
    return cxop.Identity()


# ---------------------------------------------------------------


@plum.dispatch
def frame_transform_op(
    from_frame: Galactocentric, to_frame: Galactocentric, /
) -> cxop.Pipe:
    """Return a sequence of operators for the Galactocentric frame self transformation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.frames as cxf

    >>> gcf_frame = cxf.Galactocentric()
    >>> frame_op = cxf.frame_transform_op(gcf_frame, gcf_frame)
    >>> frame_op
    Pipe(Identity())

    >>> gcf_frame2 = cxf.Galactocentric(roll=u.Q(10, "deg"))
    >>> frame_op2 = cxf.frame_transform_op(gcf_frame, gcf_frame2)
    >>> frame_op2
    Pipe((
        Boost(CartVel3D( ... )),
        Rotate(rotation=f32[3,3]),
        Translate(Cart3D( ... )),
        Rotate(rotation=f32[3,3]),
        Rotate(rotation=f32[3,3]),
        Translate(Cart3D( ... )),
        Rotate(rotation=f32[3,3]),
        Boost(CartVel3D( ... ))
    ))

    """
    if from_frame == to_frame:
        return cxop.Pipe((cxop.Identity(),))

    # TODO: not go through ICRS for the self-transformation
    return cxapi.simplify(
        frame_transform_op(from_frame, ICRS()) | frame_transform_op(ICRS(), to_frame)
    )


# ---------------------------------------------------------------


@plum.dispatch
def frame_transform_op(from_frame: ICRS, to_frame: Galactocentric, /) -> cxop.Pipe:
    r"""Return an ICRS to Galactocentric frame transformation operator.

    This transformation applies a series of Galilean transformations to convert
    coordinates from the ICRS frame to a Galactocentric frame. The transformation
    accounts for:

    1. Rotation to align with the Galactic coordinate system
    2. Translation to the Galactic center
    3. Tilt correction for the Sun's height above the Galactic plane
    4. Velocity boost to the Galactocentric rest frame


    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Create the frames:

    >>> icrs_frame = cxf.ICRS()
    >>> gcf_frame = cxf.Galactocentric()

    Define the transformation operator:

    >>> frame_op = cxf.frame_transform_op(icrs_frame, gcf_frame)
    >>> frame_op
    Pipe((
        Rotate(rotation=f32[3,3]),
        Translate(Cart3D( ... )),
        Rotate(rotation=f32[3,3]),
        Boost(CartVel3D( ... ))
    ))

    Transform a position at the origin of ICRS to Galactocentric:

    >>> q = cx.Vector.from_([0, 0, 0], "pc")
    >>> print(frame_op(q))
    <CartesianPos3D: (x, y, z) [pc]
        [-8121.972     0.       20.8  ]>

    The result shows the Sun's position in Galactocentric coordinates: the Sun
    is about 8.1 kpc from the Galactic center along the x-axis and about 21 pc
    above the Galactic plane.

    Transform both position and velocity:

    >>> q = cx.Vector.from_([0, 0, 0], "pc")
    >>> p = cx.Vector.from_([0, 0, 0], "km/s")
    >>> newq, newp = frame_op(q, p)
    >>> print(newq, newp, sep="\n")
    <Cart3D: (x, y, z) [pc]
        [-8121.972     0.       20.8  ]>
    <CartVel3D: (x, y, z) [km / s]
        [ 12.9  245.6    7.78]>

    The velocity shows the Sun's motion in the Galactocentric frame.

    Works with unxt Quantities too:

    >>> q = u.Q([0, 0, 0], "pc")
    >>> frame_op(q)
    Quantity(Array([-8121.972, 0. , 20.8 ], dtype=float32), unit='pc')

    >>> p = u.Q([0., 0, 0], "km/s")
    >>> newq, newp = frame_op(q, p)
    >>> newq, newp
    (Quantity(Array([-8121.972,     0.   ,    20.8  ], dtype=float32), unit='pc'),
     Quantity(Array([ 12.9 , 245.6 ,   7.78], dtype=float32), unit='km / s'))

    Transform a star position in spherical coordinates:

    >>> star_q = cx.Vector.from_(
    ...     {"lon": u.Q(279.23, "deg"), "lat": u.Q(38.78, "deg"),
    ...      "distance": u.Q(25, "pc")}, cxc.lonlatsph3d)
    >>> star_p = cx.Vector.from_(
    ...     {"lon_coslat": u.Q(200, "mas/yr"), "lat": u.Q(-286, "mas/yr"),
    ...      "distance": u.Q(-13.9, "km/s")}, cxc.loncoslatsph3d)
    >>> gcf_q, gcf_p = frame_op(star_q, star_p)
    >>> gcf_q = gcf_q.vconvert(cxc.cart3d)
    >>> gcf_p = gcf_p.vconvert(cxc.cart3d, gcf_q)
    >>> print(gcf_q)
    <CartesianPos3D: (x, y, z) [pc]
        [-8112.897    21.799    29.015]>
    >>> print(gcf_p.uconvert({u.dimension("speed"): "km/s"}))
    <CartesianVel3D: (x, y, z) [km / s]
        [ 34.067 234.615 -28.759]>

    Notes
    -----
    The transformation is composed of:

    - R: Combined rotation matrix (roll x latitude x longitude)
    - offset_q: Translation by the Galactic center distance
    - H: Rotation to account for Sun's height above plane
    - offset_v: Velocity boost to Galactocentric rest frame

    The default Galactocentric frame uses parameters from the Astropy default
    values (as of v4.0), which are based on various literature sources.

    """
    # rotation matrix to align x(ICRS) with the vector to the Galactic center
    galcen = to_frame.galcen
    rot_lat = cxop.Rotate.from_euler("y", galcen["lat"])
    rot_lon = cxop.Rotate.from_euler("z", -galcen["lon"])
    # extra roll away from the Galactic x-z plane
    roll = cxop.Rotate.from_euler("x", to_frame.roll - to_frame.roll0)
    # construct transformation matrix
    R = (roll @ rot_lat @ rot_lon).simplify()

    # Translation by Sun-Galactic center distance around x' and rotate about y'
    # to account for tilt due to Sun's height above the plane
    z_d = u.ustrip("", to_frame.z_sun / galcen["distance"])  # [radian]
    H = cxop.Rotate.from_euler("y", u.Q(jnp.asin(z_d), "rad"))

    # Post-rotation spatial offset to Galactic center.
    offset_q = cxop.Translate.from_(-galcen["distance"] * jnp.asarray([1, 0, 0]))

    # Post-rotation velocity offset
    galcen_v_sun = to_frame.galcen_v_sun
    offset_v = cxop.Boost(galcen_v_sun.data, chart=galcen_v_sun.chart)

    # Total Operator
    return R | offset_q | H | offset_v


# ---------------------------------------------------------------


@plum.dispatch
def frame_transform_op(from_frame: Galactocentric, to_frame: ICRS, /) -> cxop.Pipe:
    r"""Return a Galactocentric to ICRS frame transformation operator.

    This transformation inverts the ICRS→Galactocentric transformation,
    converting coordinates from a Galactocentric frame back to ICRS.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Create the frames:

    >>> icrs_frame = cxf.ICRS()
    >>> gcf_frame = cxf.Galactocentric()

    Define the transformation operator:

    >>> frame_op = cxf.frame_transform_op(gcf_frame, icrs_frame)

    Transform from Galactocentric origin to ICRS:

    >>> q = cx.Vector.from_([0, 0, 0], "pc")
    >>> print(frame_op(q).round(0))
    <CartesianPos3D: (x, y, z) [pc]
        [ -446. -7094. -3930.]>

    This shows the Galactic center's position in ICRS coordinates from the
    Sun's perspective.

    Transform both position and velocity:

    >>> q = cx.Vector.from_([0, 0, 0], "pc")
    >>> p = cx.Vector.from_([0, 0, 0], "km/s")
    >>> newq, newp = frame_op(q, p)
    >>> print(newq, newp, sep="\n")
    <Cart3D: (x, y, z) [pc]
        [ -445.689 -7094.056 -3929.708]>
    <CartVel3D: (x, y, z) [km / s]
        [-113.868  122.047 -180.79 ]>

    Works with unxt Quantities:

    >>> q = u.Q([0, 0, 0], "pc")
    >>> frame_op(q).round(0)
    Quantity(Array([ -446., -7094., -3930.], dtype=float32), unit='pc')

    >>> p = u.Q([0., 0, 0], "km/s")
    >>> newq, newp = frame_op(q, p)
    >>> newq.round(0), newp.round(0)
    (Quantity(Array([ -446., -7094., -3930.], dtype=float32), unit='pc'),
     Quantity(Array([-114.,  122., -181.], dtype=float32), unit='km / s'))

    Transform a star in Galactocentric coordinates back to ICRS:

    >>> star_q = cx.Vector.from_([-8112.9, 21.8, 29.0], "pc")
    >>> star_p = cx.Vector.from_([34.07, 234.62, -28.76], "km/s")
    >>> icrs_q, icrs_p = frame_op(star_q, star_p)
    >>> icrs_q = icrs_q.vconvert(cxc.lonlatsph3d)
    >>> icrs_p = icrs_p.vconvert(cxc.loncoslatsph3d, icrs_q)
    >>> print(icrs_q.uconvert({u.dimension("angle"): "deg", u.dimension("length"): "pc"}))
    <LonLatSphericalPos: (lon[deg], lat[deg], distance[pc])
        [279.272  38.775  24.996]>
    >>> print(icrs_p.uconvert({u.dimension("angular speed"): "mas / yr", u.dimension("speed"): "km/s"}))
    <LonCosLatSphericalVel: (lon_coslat[mas / yr], lat[mas / yr], distance[km / s])
        [ 199.983 -286.16   -13.879]>

    Notes
    -----
    This transformation is implemented by computing the inverse of the
    ICRS→Galactocentric transformation. The operator pipeline is simplified
    automatically for computational efficiency.

    """  # noqa: E501
    icrs2gcf = cxapi.frame_transform_op(to_frame, from_frame)  # pylint: disable=W1114
    return icrs2gcf.inverse.simplify()
