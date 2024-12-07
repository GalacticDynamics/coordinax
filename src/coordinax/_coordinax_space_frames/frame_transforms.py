"""Astronomy reference frames."""
# ruff:noqa: N806

__all__: list[str] = []


from typing import Literal, TypeAlias

from jaxtyping import Array, Shaped
from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u

from .galactocentric import Galactocentric
from .icrs import ICRS
from coordinax._src.angles import Angle
from coordinax._src.distances import Distance
from coordinax._src.operators import (
    AbstractOperator,
    GalileanRotation,
    Identity,
    Sequence,
)
from coordinax._src.vectors.base import AbstractVel
from coordinax._src.vectors.d3 import AbstractPos3D, CartesianPos3D, CartesianVel3D

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
def frame_transform_op(
    from_frame: Galactocentric, to_frame: Galactocentric, /
) -> Sequence:
    """Return a sequence of operators for the Galactocentric frame self transformation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.frames as cxf

    >>> gcf_frame = cxf.Galactocentric()
    >>> frame_op = cxf.frame_transform_op(gcf_frame, gcf_frame)
    >>> frame_op
    Sequence((Identity(),))

    >>> gcf_frame2 = cxf.Galactocentric(roll=u.Quantity(10, "deg"))
    >>> frame_op2 = cxf.frame_transform_op(gcf_frame, gcf_frame2)
    >>> frame_op2
    Sequence(( _GCF2ICRSOperator( ... ), _ICRS2GCFOperator( ... ) ))

    """
    if from_frame == to_frame:
        return Sequence((Identity(),))

    # TODO: not go through ICRS for the self-transformation
    return _GCF2ICRSOperator(from_frame) | _ICRS2GCFOperator(to_frame)


# ---------------------------------------------------------------


def _icrs_cartesian_to_gcf_cartesian_matrix_vectors(
    frame: Galactocentric, /
) -> tuple[RotationMatrix, LengthVector, VelocityVector]:
    """ICRS->GCF transformation matrices and offsets."""
    # rotation matrix to align x(ICRS) with the vector to the Galactic center
    mat1 = GalileanRotation.from_euler("y", frame.galcen.lat).rotation
    mat2 = GalileanRotation.from_euler("z", -frame.galcen.lon).rotation
    # extra roll away from the Galactic x-z plane
    mat0 = GalileanRotation.from_euler("x", frame.roll - frame.roll0).rotation

    # construct transformation matrix and use it
    R = mat0 @ mat1 @ mat2

    # Now need to translate by Sun-Galactic center distance around x' and
    # rotate about y' to account for tilt due to Sun's height above the plane
    z_d = u.ustrip("", frame.z_sun / frame.galcen.distance)  # [radian]
    H = GalileanRotation.from_euler("y", u.Quantity(jnp.asin(z_d), "rad")).rotation

    # compute total matrices
    A = H @ R
    offset = -H @ (frame.galcen.distance * jnp.asarray([1.0, 0.0, 0.0]))

    return A, offset, convert(frame.galcen_v_sun, u.Quantity)


class _ICRS2GCFOperator(AbstractOperator):
    """Transform from ICRS to Galactocentric frame.

    .. warning::

        This operator is temporary and will be replaced as an operator Sequence
        of the individual transformations.

    Examples
    --------
    For this example we compare against the Astropy implementation of the
    frame transformation:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import astropy.coordinates as apyc

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

    >>> apy_gcf = apyc.Galactocentric()
    >>> apy_gcf
    <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
        (266.4051, -28.936175)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg)>

    >>> vega.transform_to(apy_gcf)
    <SkyCoord (Galactocentric: ...): (x, y, z) in pc
        (-8112.89970167, 21.79911216, 29.01384942)
     (v_x, v_y, v_z) in km / s
        (34.06711868, 234.61647066, -28.75976702)>

    Now we can use the `_ICRS2GCFOperator` to perform the same transformation:

    >>> vega_q = cx.vecs.LonLatSphericalPos.from_(vega.icrs.data)
    >>> vega_p = cx.vecs.LonCosLatSphericalVel.from_(vega.icrs.data.differentials["s"])

    >>> icrs_frame = cx.frames.ICRS()
    >>> gcf_frame = cx.frames.Galactocentric.from_(apy_gcf)

    >>> frame_op = cx.frames.frame_transform_op(icrs_frame, gcf_frame)
    >>> vega_gcf_q, vega_gcf_p = frame_op(vega_q, vega_p)
    >>> print(vega_gcf_q)
    <CartesianPos3D (x[pc], y[pc], z[pc])
        [-8112.899    21.799    29.014]>
    >>> print(vega_gcf_p.uconvert({"speed": "km/s"}))
    <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
        [ 34.067 234.616 -28.76 ]>

    It matches!

    """  # noqa: E501

    gcf: Galactocentric

    @property
    def is_inertial(self) -> Literal[True]:
        """Return that the operator is inertial.

        Examples
        --------
        >>> import coordinax.frames as cxf
        >>> frame_op = cxf.frame_transform_op(cxf.ICRS(), cxf.Galactocentric())
        >>> frame_op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> AbstractOperator:
        """Return the inverse operator -- Galactocentric to ICRS.

        Examples
        --------
        >>> import coordinax.frames as cxf

        >>> icrs, gcf = cxf.ICRS(), cxf.Galactocentric()
        >>> frame_op = cxf.frame_transform_op(icrs, gcf)

        >>> frame_op.inverse == cxf.frame_transform_op(gcf, icrs)
        Array(True, dtype=bool)

        """
        return _GCF2ICRSOperator(self.gcf)

    def _call_q(self, q: LengthVector, /) -> LengthVector:
        A, offset, _ = _icrs_cartesian_to_gcf_cartesian_matrix_vectors(self.gcf)

        return A @ q + offset

    @dispatch
    def __call__(self, q: LengthVector, /) -> LengthVector:
        """Transform q from ICRS Cartesian -> GCF Cartesian.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.frames as cxf

        >>> frame_op = cxf.frame_transform_op(cxf.ICRS(), cxf.Galactocentric())

        >>> q = u.Quantity([0, 0, 0], "pc")
        >>> frame_op(q)
        Quantity[...](Array([-8121.973, 0. , 20.8 ], dtype=float32), unit='pc')

        """
        return self._call_q(q)

    @dispatch
    def __call__(
        self, q: LengthVector, p: VelocityVector, /
    ) -> tuple[LengthVector, VelocityVector]:
        """Transform q and p from ICRS Cartesian -> GCF Cartesian.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.frames as cxf

        >>> frame_op = cxf.frame_transform_op(cxf.ICRS(), cxf.Galactocentric())

        >>> q = u.Quantity([0., 0, 0], "pc")
        >>> p = u.Quantity([0., 0, 0], "km/s")

        >>> frame_op(q, p)
        (Quantity['length'](Array([-8121.973, 0. , 20.8 ], dtype=float32), unit='pc'),
         Quantity['speed'](Array([ 12.9 , 245.6 , 7.78], dtype=float32), unit='km / s'))

        """
        # Compute the transformation matrix and offsets
        A, offset, offset_v = _icrs_cartesian_to_gcf_cartesian_matrix_vectors(self.gcf)

        # Apply the transformation to the position
        qp = A @ q + offset

        # Apply the transformation to the velocity
        jac = u.experimental.jacfwd(self._call_q, argnums=0, units=(q.unit,))(q)
        pp = jac @ p + offset_v

        return qp, pp

    @dispatch
    def __call__(
        self, qvec: AbstractPos3D, pvec: AbstractVel
    ) -> tuple[AbstractPos3D, AbstractVel]:
        r"""Transform q and p from ICRS Cartesian -> GCF Cartesian.

        Examples
        --------
        >>> import coordinax as cx
        >>> import coordinax.frames as cxf

        >>> frame_op = cxf.frame_transform_op(cxf.ICRS(), cxf.Galactocentric())

        >>> q = cx.CartesianPos3D.from_([0, 0, 0], "pc")
        >>> p = cx.CartesianVel3D.from_([0, 0, 0], "km/s")

        >>> newq, newp = frame_op(q, p)
        >>> print(newq, newp, sep="\n")
        <CartesianPos3D (x[pc], y[pc], z[pc])
            [-8121.973     0.       20.8  ]>
        <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
            [ 12.9  245.6    7.78]>

        """
        p = convert(pvec.represent_as(CartesianVel3D, qvec), u.Quantity)

        qp, pp = self(convert(qvec, u.Quantity), p)
        qpvec = CartesianPos3D.from_(qp)
        ppvec = CartesianVel3D.from_(pp)

        return qpvec, ppvec


@dispatch  # type: ignore[misc]
def frame_transform_op(
    from_frame: ICRS, to_frame: Galactocentric, /
) -> _ICRS2GCFOperator:
    """Return an ICRS to Galactocentric frame transformation operator.

    Examples
    --------
    >>> import coordinax.frames as cxf
    >>> icrs_frame = cxf.ICRS()
    >>> gcf_frame = cxf.Galactocentric()
    >>> frame_op = cxf.frame_transform_op(icrs_frame, gcf_frame)
    >>> frame_op
    _ICRS2GCFOperator(
      gcf=Galactocentric(
        galcen=LonLatSphericalPos( ... ),
        roll=Quantity[...](value=weak_i32[], unit=Unit("deg")),
        z_sun=Quantity[...](value=weak_f32[], unit=Unit("pc")),
        galcen_v_sun=CartesianVel3D( ... )
      )
    )

    """
    return _ICRS2GCFOperator(to_frame)


# ---------------------------------------------------------------


def _gcf_cartesian_to_icrs_cartesian_matrix_vectors(
    frame: Galactocentric, /
) -> tuple[RotationMatrix, LengthVector, VelocityVector]:
    """GCF->ICRS transformation matrices and offsets."""
    # ICRS -> GCF
    A, offset, offset_v = _icrs_cartesian_to_gcf_cartesian_matrix_vectors(frame)

    # GCF -> ICRS
    A = A.T  # (A^-1 = A^T)
    offset = A @ (-offset)
    offset_v = A @ (-offset_v)

    return A, offset, offset_v


class _GCF2ICRSOperator(AbstractOperator):
    """Transform from Galactocentric to ICRS frame.

    .. warning::

        This operator is temporary and will be replaced as an operator Sequence
        of the individual transformations.

    Examples
    --------
    For this example we compare against the Astropy implementation of the
    frame transformation:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import astropy.coordinates as apyc

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

    >>> vega.transform_to(apyc.ICRS())
    <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, pc)
        (279.23473479, 38.78368896, 25.)
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        (200., -286., -13.9)>

    Now we can use the `_GCF2ICRSOperator` to perform the same transformation:

    >>> vega_q = cx.CartesianPos3D.from_(vega.galactocentric.data)
    >>> vega_p = cx.CartesianVel3D.from_(vega.galactocentric.data.differentials["s"])

    >>> icrs_frame = cx.frames.ICRS()
    >>> gcf_frame = cx.frames.Galactocentric.from_(apy_gcf)

    >>> frame_op = cx.frames.frame_transform_op(gcf_frame, icrs_frame)
    >>> vega_icrs_q, vega_icrs_p = frame_op(vega_q, vega_p)

    >>> vega_icrs_q = vega_icrs_q.represent_as(cx.vecs.LonLatSphericalPos)
    >>> print(vega_icrs_q.uconvert({"angle": "deg", "length": "pc"}))
    <LonLatSphericalPos (lon[deg], lat[deg], distance[pc])
        [279.235  38.785  25.   ]>

    >>> vega_icrs_p = vega_icrs_p.represent_as(cx.vecs.LonCosLatSphericalVel, vega_icrs_q)
    >>> print(vega_icrs_p.uconvert({"angular speed": "mas / yr", "speed": "km/s"}))
    <LonCosLatSphericalVel (d_lon_coslat[mas / yr], d_lat[mas / yr], d_distance[km / s])
        [ 200.002 -286.002  -13.9  ]>

    It matches!

    """  # noqa: E501

    gcf: Galactocentric

    @property
    def is_inertial(self) -> Literal[True]:
        """Return that the operator is inertial.

        Examples
        --------
        >>> import coordinax.frames as cxf
        >>> frame_op = cxf.frame_transform_op(cxf.Galactocentric(), cxf.ICRS())
        >>> frame_op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> AbstractOperator:
        """Return the inverse operator -- ICRS to Galactocentric.

        Examples
        --------
        >>> import coordinax.frames as cxf

        >>> icrs, gcf = cxf.ICRS(), cxf.Galactocentric()
        >>> frame_op = cxf.frame_transform_op(gcf, icrs)

        >>> frame_op.inverse == cxf.frame_transform_op(icrs, gcf)
        Array(True, dtype=bool)

        """
        return _ICRS2GCFOperator(self.gcf)

    def _call_q(self, q: LengthVector, /) -> LengthVector:
        A, offset, _ = _gcf_cartesian_to_icrs_cartesian_matrix_vectors(self.gcf)

        return A @ q + offset

    @dispatch
    def __call__(self, q: LengthVector, /) -> LengthVector:
        """Transform q from GCF Cartesian -> ICRS Cartesian.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.frames as cxf

        >>> frame_op = cxf.frame_transform_op(cxf.Galactocentric(), cxf.ICRS())

        >>> q = u.Quantity([0, 0, 0], "pc")
        >>> frame_op(q).round(0)
        Quantity['length'](Array([ -446., -7094., -3930.], dtype=float32), unit='pc')

        """
        return self._call_q(q)

    @dispatch
    def __call__(
        self, q: LengthVector, p: VelocityVector, /
    ) -> tuple[LengthVector, VelocityVector]:
        r"""Transform q and p from GCF Cartesian -> ICRS Cartesian.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.frames as cxf

        >>> frame_op = cxf.frame_transform_op(cxf.Galactocentric(), cxf.ICRS())

        >>> q = u.Quantity([0., 0, 0], "pc")
        >>> p = u.Quantity([0., 0, 0], "km/s")

        >>> newq, newp = frame_op(q, p)
        >>> print(newq.round(0), newp.round(0), sep="\n")
        Quantity['length'](Array([ -446., -7094., -3930.], dtype=float32), unit='pc')
        Quantity['speed'](Array([-114., 122., -181.], dtype=float32), unit='km / s')

        """
        # Compute the transformation matrix and offsets
        A, offset, offset_v = _gcf_cartesian_to_icrs_cartesian_matrix_vectors(self.gcf)

        # Apply the transformation to the position
        qp = A @ q + offset

        # Apply the transformation to the velocity
        jac = u.experimental.jacfwd(self._call_q, argnums=0, units=(q.unit,))(q)
        pp = jac @ p + offset_v

        return qp, pp

    @dispatch
    def __call__(
        self, qvec: AbstractPos3D, pvec: AbstractVel
    ) -> tuple[AbstractPos3D, AbstractVel]:
        r"""Transform q and p from GCF Cartesian -> ICRS Cartesian.

        Examples
        --------
        >>> import coordinax as cx
        >>> import coordinax.frames as cxf

        >>> frame_op = cxf.frame_transform_op(cxf.Galactocentric(), cxf.ICRS())

        >>> q = cx.CartesianPos3D.from_([0, 0, 0], "pc")
        >>> p = cx.CartesianVel3D.from_([0, 0, 0], "km/s")

        >>> newq, newp = frame_op(q, p)
        >>> print(newq, newp, sep="\n")
        <CartesianPos3D (x[pc], y[pc], z[pc])
            [ -445.689 -7094.056 -3929.708]>
        <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
            [-113.868  122.047 -180.79 ]>

        """
        q = convert(qvec, u.Quantity)
        p = convert(pvec.represent_as(CartesianVel3D, qvec), u.Quantity)

        qp, pp = self(q, p)
        qpvec = CartesianPos3D.from_(qp)
        ppvec = CartesianVel3D.from_(pp)

        return qpvec, ppvec


@dispatch  # type: ignore[misc]
def frame_transform_op(
    from_frame: Galactocentric, to_frame: ICRS, /
) -> _GCF2ICRSOperator:
    """Return a Galactocentric to ICRS frame transformation operator.

    Examples
    --------
    >>> import coordinax.frames as cxf
    >>> icrs_frame = cxf.ICRS()
    >>> gcf_frame = cxf.Galactocentric()
    >>> frame_op = cxf.frame_transform_op(gcf_frame, icrs_frame)
    >>> frame_op
    _GCF2ICRSOperator(
      gcf=Galactocentric(
        galcen=LonLatSphericalPos( ... ),
        roll=Quantity[...](value=weak_i32[], unit=Unit("deg")),
        z_sun=Quantity[...](value=weak_f32[], unit=Unit("pc")),
        galcen_v_sun=CartesianVel3D( ... )
      )
    )

    """
    return _GCF2ICRSOperator(from_frame)
