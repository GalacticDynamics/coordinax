"""Astronomy reference frames."""
# ruff:noqa: N806

__all__: list[str] = []


from typing import TypeAlias

from jaxtyping import Array, Shaped
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u

from .galactocentric import Galactocentric
from .icrs import ICRS
from coordinax._src.angles import Angle
from coordinax._src.distances import Distance
from coordinax._src.operators import (
    GalileanRotation,
    GalileanSpatialTranslation,
    Identity,
    Pipe,
    VelocityBoost,
    simplify_op,
)

ScalarAngle: TypeAlias = Shaped[u.Quantity["angle"] | Angle, ""]
RotationMatrix: TypeAlias = Shaped[Array, "3 3"]
LengthVector: TypeAlias = Shaped[u.Quantity["length"], "3"] | Shaped[Distance, "3"]
VelocityVector: TypeAlias = Shaped[u.Quantity["speed"], "3"]


@dispatch
def frame_transform_op(from_frame: ICRS, to_frame: ICRS, /) -> Identity:
    """Return an identity operator for the ICRS->ICRS transformation.

    Examples
    --------
    >>> import coordinax.frames as cxf
    >>> icrs_frame = cxf.ICRS()
    >>> frame_op = cxf.frame_transform_op(icrs_frame, icrs_frame)
    >>> frame_op
    Identity()

    """
    return Identity()


# ---------------------------------------------------------------


@dispatch
def frame_transform_op(from_frame: Galactocentric, to_frame: Galactocentric, /) -> Pipe:
    """Return a sequence of operators for the Galactocentric frame self transformation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.frames as cxf

    >>> gcf_frame = cxf.Galactocentric()
    >>> frame_op = cxf.frame_transform_op(gcf_frame, gcf_frame)
    >>> frame_op
    Pipe((Identity(),))

    >>> gcf_frame2 = cxf.Galactocentric(roll=u.Quantity(10, "deg"))
    >>> frame_op2 = cxf.frame_transform_op(gcf_frame, gcf_frame2)
    >>> frame_op2
    Pipe((
        VelocityBoost(CartesianVel3D( ... )),
        GalileanRotation(rotation=f32[3,3]),
        GalileanSpatialTranslation(CartesianPos3D( ... )),
        GalileanRotation(rotation=f32[3,3]),
        GalileanRotation(rotation=f32[3,3]),
        GalileanSpatialTranslation(CartesianPos3D( ... )),
        GalileanRotation(rotation=f32[3,3]),
        VelocityBoost(CartesianVel3D( ... ))
    ))

    """
    if from_frame == to_frame:
        return Pipe((Identity(),))

    # TODO: not go through ICRS for the self-transformation
    return simplify_op(
        frame_transform_op(from_frame, ICRS()) | frame_transform_op(ICRS(), to_frame)
    )


# ---------------------------------------------------------------


@dispatch
def frame_transform_op(from_frame: ICRS, to_frame: Galactocentric, /) -> Pipe:
    r"""Return an ICRS to Galactocentric frame transformation operator.

    Examples
    --------
    For this example we compare against the Astropy implementation of the
    frame transformation:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import astropy.coordinates as apyc

    The location of Vega in ICRS coordinates:

    >>> vega = apyc.SkyCoord(
    ...     ra=279.23473479 * u.unit("deg"), dec=38.78368896 * u.unit("deg"),
    ...     distance=25 * u.unit("pc"),
    ...     pm_ra_cosdec=200 * u.unit("mas / yr"), pm_dec=-286 * u.unit("mas / yr"),
    ...     radial_velocity=-13.9 * u.unit("km / s"))
    >>> print(vega)
    <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, pc)
        (279.23473479, 38.78368896, 25.)
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        (200., -286., -13.9)>

    Transforming to a Galactocentric frame:

    >>> apy_gcf = apyc.Galactocentric()
    >>> apy_gcf
    <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
        (266.4051, -28.936175)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg)>

    >>> vega.transform_to(apy_gcf)
    <SkyCoord (Galactocentric: ...): (x, y, z) in pc
        (-8112.89970167, 21.79911216, 29.01384942)
     (v_x, v_y, v_z) in km / s
        (34.06711868, 234.61647066, -28.75976702)>

    Now we can use `coordinax` to perform the same transformation!

    Converting the Astropy objects to `coordinax`:

    >>> vega_q = cx.vecs.LonLatSphericalPos.from_(vega.icrs.data)
    >>> vega_p = cx.vecs.LonCosLatSphericalVel.from_(vega.icrs.data.differentials["s"])

    >>> icrs_frame = cx.frames.ICRS()
    >>> gcf_frame = cx.frames.Galactocentric.from_(apy_gcf)

    Define the transformation operator:

    >>> frame_op = cx.frames.frame_transform_op(icrs_frame, gcf_frame)
    >>> frame_op
    Pipe((
        GalileanRotation(rotation=f32[3,3]),
        GalileanSpatialTranslation(CartesianPos3D( ... )),
        GalileanRotation(rotation=f32[3,3]),
        VelocityBoost(CartesianVel3D( ... ))
    ))

    Apply the transformation:

    >>> vega_gcf_q, vega_gcf_p = frame_op(vega_q, vega_p)
    >>> vega_gcf_q = vega_gcf_q.vconvert(cx.vecs.CartesianPos3D)
    >>> vega_gcf_p = vega_gcf_p.vconvert(cx.vecs.CartesianVel3D, vega_gcf_q)
    >>> print(vega_gcf_q)
    <CartesianPos3D (x[pc], y[pc], z[pc])
        [-8112.898    21.799    29.01...]>
    >>> print(vega_gcf_p.uconvert({u.dimension("speed"): "km/s"}))
    <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
        [ 34.067 234.616 -28.76 ]>

    It matches!

    Let's do it again for a few different input types:

    >>> q = u.Quantity([0, 0, 0], "pc")
    >>> frame_op(q)
    Quantity[...](Array([-8121.972, 0. , 20.8 ], dtype=float32), unit='pc')

    >>> p = u.Quantity([0., 0, 0], "km/s")

    >>> newq, newp = frame_op(q, p)
    >>> print(newq, newp, sep="\n")
    Quantity['length'](Array([-8121.972, 0. , 20.8  ], dtype=float32), unit='pc')
    Quantity['speed'](Array([ 12.9 , 245.6 , 7.78], dtype=float32), unit='km / s')

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "pc")
    >>> p = cx.CartesianVel3D.from_([0, 0, 0], "km/s")

    >>> newq, newp = frame_op(q, p)
    >>> print(newq, newp, sep="\n")
    <CartesianPos3D (x[pc], y[pc], z[pc])
        [-8121.972     0.       20.8  ]>
    <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
        [ 12.9  245.6    7.78]>

    """  # noqa: E501
    # rotation matrix to align x(ICRS) with the vector to the Galactic center
    rot_lat = GalileanRotation.from_euler("y", to_frame.galcen.lat)
    rot_lon = GalileanRotation.from_euler("z", -to_frame.galcen.lon)
    # extra roll away from the Galactic x-z plane
    roll = GalileanRotation.from_euler("x", to_frame.roll - to_frame.roll0)
    # construct transformation matrix
    R = (roll @ rot_lat @ rot_lon).simplify()

    # Translation by Sun-Galactic center distance around x' and rotate about y'
    # to account for tilt due to Sun's height above the plane
    z_d = u.ustrip("", to_frame.z_sun / to_frame.galcen.distance)  # [radian]
    H = GalileanRotation.from_euler("y", u.Quantity(jnp.asin(z_d), "rad"))

    # Post-rotation spatial offset to Galactic center.
    offset_q = GalileanSpatialTranslation(
        -to_frame.galcen.distance * jnp.asarray([1, 0, 0])
    )

    # Post-rotation velocity offset
    offset_v = VelocityBoost(to_frame.galcen_v_sun)

    # Total Operator
    return R | offset_q | H | offset_v


# ---------------------------------------------------------------


@dispatch
def frame_transform_op(from_frame: Galactocentric, to_frame: ICRS, /) -> Pipe:
    r"""Return a Galactocentric to ICRS frame transformation operator.

    Examples
    --------
    For this example we compare against the Astropy implementation of the
    frame transformation:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import astropy.coordinates as apyc

    The location of Vega in Galactocentric coordinates:

    >>> apy_gcf = apyc.Galactocentric()
    >>> apy_gcf
    <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
        (266.4051, -28.936175)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg)>

    >>> vega = apyc.SkyCoord(
    ...     ra=279.23473479 * u.unit("deg"), dec=38.78368896 * u.unit("deg"),
    ...     distance=25 * u.unit("pc"),
    ...     pm_ra_cosdec=200 * u.unit("mas / yr"), pm_dec=-286 * u.unit("mas / yr"),
    ...     radial_velocity=-13.9 * u.unit("km / s")
    ... ).transform_to(apy_gcf)
    >>> print(vega)
    <SkyCoord (Galactocentric: ...): (x, y, z) in pc
        (-8112.89970167, 21.79911216, 29.01384942)
     (v_x, v_y, v_z) in km / s
        (34.06711868, 234.61647066, -28.75976702)>

    Transforming to an ICRS frame:

    >>> vega.transform_to(apyc.ICRS())
    <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, pc)
        (279.23473479, 38.78368896, 25.)
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        (200., -286., -13.9)>

    Now we can use `coordinax` to perform the same transformation!

    Converting the Astropy objects to `coordinax`:

    >>> vega_q = cx.CartesianPos3D.from_(vega.galactocentric.data)
    >>> vega_p = cx.CartesianVel3D.from_(vega.galactocentric.data.differentials["s"])

    >>> icrs_frame = cx.frames.ICRS()
    >>> gcf_frame = cx.frames.Galactocentric.from_(apy_gcf)

    Define the transformation operator:

    >>> frame_op = cx.frames.frame_transform_op(gcf_frame, icrs_frame)

    Apply the transformation:

    >>> vega_icrs_q, vega_icrs_p = frame_op(vega_q, vega_p)

    >>> vega_icrs_q = vega_icrs_q.vconvert(cx.vecs.LonLatSphericalPos)
    >>> vega_icrs_p = vega_icrs_p.vconvert(cx.vecs.LonCosLatSphericalVel, vega_icrs_q)
    >>> print(vega_icrs_q.uconvert({u.dimension("angle"): "deg", u.dimension("length"): "pc"}))
    <LonLatSphericalPos (lon[deg], lat[deg], distance[pc])
        [279.235  38.784  25.   ]>
    >>> print(vega_icrs_p.uconvert({u.dimension("angular speed"): "mas / yr", u.dimension("speed"): "km/s"}))
    <LonCosLatSphericalVel (d_lon_coslat[mas / yr], d_lat[mas / yr], d_distance[km / s])
        [ 200.001 -286.  -13.9  ]>

    It matches!

    Let's do it again for a few different input types:

    >>> q = u.Quantity([0, 0, 0], "pc")
    >>> frame_op(q).round(0)
    Quantity['length'](Array([ -446., -7094., -3930.], dtype=float32), unit='pc')

    >>> p = u.Quantity([0., 0, 0], "km/s")

    >>> newq, newp = frame_op(q, p)
    >>> print(newq.round(0), newp.round(0), sep="\n")
    Quantity['length'](Array([ -446., -7094., -3930.], dtype=float32), unit='pc')
    Quantity['speed'](Array([-114., 122., -181.], dtype=float32), unit='km / s')

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "pc")
    >>> p = cx.CartesianVel3D.from_([0, 0, 0], "km/s")

    >>> newq, newp = frame_op(q, p)
    >>> print(newq, newp, sep="\n")
    <CartesianPos3D (x[pc], y[pc], z[pc])
        [ -445.689 -7094.056 -3929.708]>
    <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
        [-113.868  122.047 -180.79 ]>

    """  # noqa: E501
    icrs2gcf = frame_transform_op(to_frame, from_frame)  # pylint: disable=W1114
    return icrs2gcf.inverse.simplify()
