"""Astronomy reference frames."""

__all__: tuple[str, ...] = ()


from typing import cast

import plum

import quaxed.numpy as jnp
import unxt as u

import coordinax.frames as cxf
import coordinax.transforms as cxfm
from .base_frame import AbstractSpaceFrame
from .galactocentric import Galactocentric
from .icrs import ICRS, icrs

# ---------------------------------------------------------------
# Base Space-Frame Transformation


@plum.dispatch
def frame_transition(
    from_frame: AbstractSpaceFrame, to_frame: AbstractSpaceFrame, /
) -> cxfm.Composed:
    """Compute frame transformations with ICRS as the intermediary.

    Examples
    --------
    >>> import plum
    >>> import unxt as u
    >>> import coordinax.frames as cxf
    >>> import coordinax.astro as cxastro

    >>> class MySpaceFrame(cxastro.AbstractSpaceFrame):
    ...     pass

    >>> @plum.dispatch
    ... def frame_transition(from_frame: MySpaceFrame, to_frame: ICRS, /) -> cxfm.AbstractTransform:
    ...     return cxfm.Rotate.from_euler("z", u.Q(10, "deg"))

    We can transform from `MySpaceFrame` to a Galactocentric frame, even though
    we don't have a direct transformation defined:

    >>> my_frame = MySpaceFrame()
    >>> gcf_frame = cxastro.Galactocentric()

    >>> op = cxf.frame_transition(my_frame, gcf_frame)
    >>> op
    Composed((
      Rotate(f64[3,3](jax)),
      Rotate(f64[3,3](jax)),
      Translate( {...}, chart=Cart3D(M=Rn(3)) ),
      Rotate(f64[3,3](jax))
    ))

    """  # noqa: E501
    fromframe_to_icrs = frame_transition(from_frame, icrs)
    icrs_to_toframe = frame_transition(icrs, to_frame)
    pipe = fromframe_to_icrs | icrs_to_toframe
    return cast("cxfm.Composed", cxfm.simplify(pipe))


# ---------------------------------------------------------------


@plum.dispatch
def frame_transition(from_frame: ICRS, to_frame: ICRS, /) -> cxfm.Identity:
    """Return an identity operator for the ICRS->ICRS transformation.

    Examples
    --------
    >>> import coordinax.frames as cxf
    >>> import coordinax.astro as cxastro

    >>> icrs_frame = cxastro.ICRS()
    >>> frame_op = cxf.frame_transition(icrs_frame, icrs_frame)
    >>> frame_op
    Identity()

    """
    return cxfm.identity


# ---------------------------------------------------------------


@plum.dispatch
def frame_transition(
    from_frame: Galactocentric, to_frame: Galactocentric, /
) -> cxfm.Composed:
    """Return a sequence of operators for the Galactocentric frame self transformation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.frames as cxf
    >>> import coordinax.astro as cxastro

    >>> gcf_frame = cxastro.Galactocentric()
    >>> frame_op = cxf.frame_transition(gcf_frame, gcf_frame)
    >>> frame_op
    Composed(Identity())

    >>> gcf_frame2 = cxastro.Galactocentric(roll=u.Q(10, "deg"))
    >>> frame_op2 = cxf.frame_transition(gcf_frame, gcf_frame2)
    >>> frame_op2
    Composed((
      Rotate(f64[3,3](jax)),
      Translate( {...}, chart=Cart3D(M=Rn(3)) ),
      Rotate(f64[3,3](jax)),
      Rotate(f64[3,3](jax)),
      Translate( {...}, chart=Cart3D(M=Rn(3)) ),
      Rotate(f64[3,3](jax))
    ))

    """
    if from_frame == to_frame:
        return cxfm.Composed((cxfm.identity,))

    # TODO: not go through ICRS for the self-transformation
    return cxfm.simplify(  # ty: ignore[invalid-return-type]
        frame_transition(from_frame, icrs) | frame_transition(icrs, to_frame)
    )


# ---------------------------------------------------------------


@plum.dispatch
def frame_transition(from_frame: ICRS, to_frame: Galactocentric, /) -> cxfm.Composed:
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
    >>> import coordinax.main as cx
    >>> import coordinax.astro as cxastro

    Create the frames:

    >>> icrs_frame = cxastro.ICRS()
    >>> gcf_frame = cxastro.Galactocentric()

    Define the transformation operator:

    >>> frame_op = cx.frame_transition(icrs_frame, gcf_frame)
    >>> frame_op
    Composed((
      Rotate(f64[3,3](jax)),
      Translate( {...}, chart=Cart3D(M=Rn(3)) ),
      Rotate(f64[3,3](jax))
    ))

    Transform a position at the origin of ICRS to Galactocentric:

    >>> q = cx.Point.from_([0, 0, 0], "pc")
    >>> print(frame_op(q))
    <Point: chart=Cart3D (x, y, z) [pc]
        [-8121.973     0.       20.8  ]>

    The result shows the Sun's position in Galactocentric coordinates: the Sun
    is about 8.1 kpc from the Galactic center along the x-axis and about 21 pc
    above the Galactic plane.

    Works with unxt Quantities too:

    >>> q = u.Q([0, 0, 0], "pc")
    >>> frame_op(q)
    Q([-8121.97336612,     0.        ,    20.8       ], 'pc')

    Transform a star position in spherical coordinates:

    >>> star_q = cx.Point.from_(
    ...     {"lon": u.Q(279.23, "deg"), "lat": u.Q(38.78, "deg"),
    ...      "distance": u.Q(25, "pc")}, cx.lonlat_sph3d)
    >>> gcf_q = frame_op(star_q)
    >>> gcf_q = gcf_q.cconvert(cx.cart3d)
    >>> print(gcf_q)
    <Point: chart=Cart3D (x, y, z) [pc]
        [-8112.898    21.798    29.015]>

    Notes
    -----
    The transformation is composed of:

    - R: Combined rotation matrix (longitude x latitude x roll)
    - offset_q: Translation by the Galactic center distance
    - H: Rotation to account for Sun's height above plane
    - offset_v: Velocity boost to Galactocentric rest frame

    The default Galactocentric frame uses parameters from the Astropy default
    values (as of v4.0), which are based on various literature sources.

    """
    # rotation matrix to align x(ICRS) with the vector to the Galactic center
    galcen = to_frame.galcen
    rot_lat = cxfm.Rotate.from_euler("y", galcen["lat"])
    rot_lon = cxfm.Rotate.from_euler("z", -galcen["lon"])
    # extra roll away from the Galactic x-z plane
    roll = cxfm.Rotate.from_euler("x", to_frame.roll - to_frame.roll0)
    # construct transformation matrix
    R = (rot_lon @ rot_lat @ roll).simplify()

    # Translation by Sun-Galactic center distance around x' and rotate about y'
    # to account for tilt due to Sun's height above the plane
    z_d = u.ustrip("", to_frame.z_sun / galcen["distance"])  # [radian]
    H = cxfm.Rotate.from_euler("y", u.Q(jnp.asin(z_d), "rad"))  # ty: ignore[no-matching-overload]

    # Post-rotation spatial offset to Galactic center.
    offset_q = cxfm.Translate.from_(-galcen["distance"] * jnp.asarray([1, 0, 0]))

    # TODO: re-add when Boost is a class
    # # Post-rotation velocity offset
    # galcen_v_sun = to_frame.galcen_v_sun
    # offset_v = cxf.Boost(galcen_v_sun.data, chart=galcen_v_sun.chart)

    # # Total Operator
    # return R | offset_q | H | offset_v
    # Total Operator
    return R | offset_q | H


# ---------------------------------------------------------------


@plum.dispatch
def frame_transition(from_frame: Galactocentric, to_frame: ICRS, /) -> cxfm.Composed:
    r"""Return a Galactocentric to ICRS frame transformation operator.

    This transformation inverts the ICRS→Galactocentric transformation,
    converting coordinates from a Galactocentric frame back to ICRS.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> import coordinax.astro as cxastro

    Create the frames:

    >>> icrs_frame = cxastro.ICRS()
    >>> gcf_frame = cxastro.Galactocentric()

    Define the transformation operator:

    >>> frame_op = cx.frame_transition(gcf_frame, icrs_frame)

    Transform from Galactocentric origin to ICRS:

    >>> q = cx.Point.from_([0, 0, 0], "pc")
    >>> print(frame_op(q).round(0))
    <Point: chart=Cart3D (x, y, z) [pc]
        [ -446. -7094. -3930.]>

    This shows the Galactic center's position in ICRS coordinates from the
    Sun's perspective.

    Works with unxt Quantities:

    >>> q = u.Q([0, 0, 0], "pc")
    >>> frame_op(q).round(0)
    Q([ -446., -7094., -3930.], 'pc')

    Transform a star in Galactocentric coordinates back to ICRS:

    >>> star_q = cx.Point.from_([-8112.9, 21.8, 29.0], "pc")
    >>> icrs_q = frame_op(star_q)
    >>> icrs_q = icrs_q.cconvert(cx.lonlat_sph3d)
    >>> print(icrs_q.uconvert({u.dimension("angle"): "deg", u.dimension("length"): "pc"}))
    <Point: chart=LonLatSpherical3D (lon[deg], lat[deg], distance[pc])
        [-80.728  38.775  24.996]>

    Notes
    -----
    This transformation is implemented by computing the inverse of the
    ICRS→Galactocentric transformation. The operator pipeline is simplified
    automatically for computational efficiency.

    """  # noqa: E501
    icrs2gcf = cxf.frame_transition(to_frame, from_frame)  # pylint: disable=W1114
    return icrs2gcf.inverse.simplify()  # ty: ignore[unresolved-attribute]
