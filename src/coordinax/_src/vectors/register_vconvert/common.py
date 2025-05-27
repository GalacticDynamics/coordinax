"""Transformations for Vectors."""

__all__: list[str] = []

from typing import Any, cast
from warnings import warn

import equinox as eqx
from plum import dispatch

import coordinax._src.vectors.custom_types as ct
from coordinax._src.vectors import d1, d2, d3, d4, dn
from coordinax._src.vectors.api import vconvert
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_acc import AbstractAcc
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel
from coordinax._src.vectors.exceptions import IrreversibleDimensionChange
from coordinax._src.vectors.private_api import combine_aux, wrap_vconvert_impl_params


def get_params_and_aux(obj: AbstractVector, /) -> tuple[ct.ParamsDict, ct.AuxDict]:
    # Get the parameters and auxiliary data
    p_and_aux = cast("dict[str, Any]", obj.asdict())
    # Separate the parameters from the auxiliary data
    comps = obj.components
    p = {k: p_and_aux.pop(k) for k in tuple(p_and_aux) if k in comps}
    in_aux = p_and_aux  # popped all the params out
    return p, in_aux


# =============================================================================


@dispatch.multi(
    (type[AbstractPos], AbstractPos),
    (type[AbstractVel], AbstractVel),
    (type[AbstractAcc], AbstractAcc),
)
def vconvert(
    target: type[AbstractVector],
    current: AbstractVector,
    /,
    units: ct.OptUSys = None,
    **out_aux: Any,
) -> AbstractVector:
    """AbstractPos -> vconvert_impl -> AbstractPos.

    This is the base case for the transformation of position vectors.

    """
    # Get the parameters and auxiliary data
    p_and_aux = cast("dict[str, Any]", current.asdict())
    # Separate the parameters from the auxiliary data
    comps = current.components
    p = {k: p_and_aux.pop(k) for k in tuple(p_and_aux) if k in comps}
    in_aux = p_and_aux  # popped all the params out

    # Parameters can be passed by kwarg, so we need to filter them out
    # from the auxiliary data. E.g. for RadialPos -> CartesianPos2D
    # we need to specify `y` by kwarg. It is a param, not an aux.
    p_by_kw = {k: out_aux.pop(k) for k in tuple(out_aux) if k in target.components}
    p_by_kw = eqx.error_if(p_by_kw, any(set(p_by_kw) & set(p)), "overlap in params.")
    p = p | p_by_kw

    # Convert the parameters, using the auxiliary data
    p, aux = vconvert(
        target, type(current), p, in_aux=in_aux, out_aux=out_aux, units=units
    )

    # Build the new vector
    return target(**(aux or {}), **p)


@dispatch.multi(
    (type[AbstractPos], type[AbstractPos], AbstractPos),
    (type[AbstractVel], type[AbstractVel], AbstractVel),
    (type[AbstractAcc], type[AbstractAcc], AbstractAcc),
)
def vconvert(
    to_vector: type[AbstractVector],
    from_vector: type[AbstractVector],
    current: AbstractVector,
    /,
    **out_aux: Any,
) -> AbstractVector:
    """Convert from one position vector to another.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    >>> x = cxv.CartesianPos2D.from_([3, 4], "km")
    >>> y = cxv.vconvert(cxv.PolarPos, cxv.CartesianPos2D, x)
    >>> print(y)
    <PolarPos: (r[km], phi[rad])
        [5.    0.927]>

    """
    current = eqx.error_if(
        current,
        not isinstance(current, from_vector),
        f"{from_vector} expected, got {type(current)}.",
    )
    return vconvert(to_vector, current, **out_aux)


# =============================================================================
# Self transform


@dispatch.multi(
    # Positions
    (type[d1.CartesianPos1D], type[d1.CartesianPos1D], ct.ParamsDict),
    (type[d1.RadialPos], type[d1.RadialPos], ct.ParamsDict),
    (type[d2.CartesianPos2D], type[d2.CartesianPos2D], ct.ParamsDict),
    (type[d2.PolarPos], type[d2.PolarPos], ct.ParamsDict),
    (type[d2.TwoSpherePos], type[d2.TwoSpherePos], ct.ParamsDict),
    (type[d3.CartesianPos3D], type[d3.CartesianPos3D], ct.ParamsDict),
    (type[d3.CylindricalPos], type[d3.CylindricalPos], ct.ParamsDict),
    (type[d3.SphericalPos], type[d3.SphericalPos], ct.ParamsDict),
    (type[d3.LonLatSphericalPos], type[d3.LonLatSphericalPos], ct.ParamsDict),
    (type[d3.MathSphericalPos], type[d3.MathSphericalPos], ct.ParamsDict),
    (type[dn.PoincarePolarVector], type[dn.PoincarePolarVector], ct.ParamsDict),
    (type[dn.CartesianPosND], type[dn.CartesianPosND], ct.ParamsDict),
    # Velocities
    (type[d1.CartesianVel1D], type[d1.CartesianVel1D], ct.ParamsDict),
    (type[d1.RadialVel], type[d1.RadialVel], ct.ParamsDict),
    (type[d2.CartesianVel2D], type[d2.CartesianVel2D], ct.ParamsDict),
    (type[d2.PolarVel], type[d2.PolarVel], ct.ParamsDict),
    (type[d2.TwoSphereVel], type[d2.TwoSphereVel], ct.ParamsDict),
    (type[d3.CartesianVel3D], type[d3.CartesianVel3D], ct.ParamsDict),
    (type[d3.CylindricalVel], type[d3.CylindricalVel], ct.ParamsDict),
    (type[d3.SphericalVel], type[d3.SphericalVel], ct.ParamsDict),
    (type[d3.LonLatSphericalVel], type[d3.LonLatSphericalVel], ct.ParamsDict),
    (type[d3.LonCosLatSphericalVel], type[d3.LonCosLatSphericalVel], ct.ParamsDict),
    (type[d3.MathSphericalVel], type[d3.MathSphericalVel], ct.ParamsDict),
    (type[dn.CartesianVelND], type[dn.CartesianVelND], ct.ParamsDict),
    # Accelerations
    (type[d1.CartesianAcc1D], type[d1.CartesianAcc1D], ct.ParamsDict),
    (type[d1.RadialAcc], type[d1.RadialAcc], ct.ParamsDict),
    (type[d2.CartesianAcc2D], type[d2.CartesianAcc2D], ct.ParamsDict),
    (type[d3.CartesianAcc3D], type[d3.CartesianAcc3D], ct.ParamsDict),
    (type[d3.CylindricalAcc], type[d3.CylindricalAcc], ct.ParamsDict),
    (type[d3.SphericalAcc], type[d3.SphericalAcc], ct.ParamsDict),
    (type[d3.LonLatSphericalAcc], type[d3.LonLatSphericalAcc], ct.ParamsDict),
    (type[d3.MathSphericalAcc], type[d3.MathSphericalAcc], ct.ParamsDict),
    (type[dn.CartesianAccND], type[dn.CartesianAccND], ct.ParamsDict),
)
def vconvert(
    to_vector: type[AbstractVector],
    from_vector: type[AbstractVector],
    params: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """Self transform.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    ## 1D:

    >>> params = {"x": 1}
    >>> cxv.vconvert(cxv.CartesianPos1D, cxv.CartesianPos1D, params)
    ({'x': 1}, {})

    >>> params = {"r": 1}
    >>> cxv.vconvert(cxv.RadialPos, cxv.RadialPos, params)
    ({'r': 1}, {})

    ## 2D:

    >>> params = {"x": 1, "y": 2}
    >>> cxv.vconvert(cxv.CartesianPos2D, cxv.CartesianPos2D, params)
    ({'x': 1, 'y': 2}, {})

    >>> params = {"r": 1, "phi": 10}
    >>> cxv.vconvert(cxv.PolarPos, cxv.PolarPos, params)
    ({'r': 1, 'phi': 10}, {})

    >>> params = {"theta": 10, "phi": 14}
    >>> cxv.vconvert(cxv.TwoSpherePos, cxv.TwoSpherePos, params)
    ({'theta': 10, 'phi': 14}, {})

    ## 3D:

    >>> params = {"x": 1, "y": 2, "z": 3}
    >>> cxv.vconvert(cxv.CartesianPos3D, cxv.CartesianPos3D, params)
    ({'x': 1, 'y': 2, 'z': 3}, {})

    >>> params = {"rho": 1, "phi": 2, "z": 3}
    >>> cxv.vconvert(cxv.CylindricalPos, cxv.CylindricalPos, params)
    ({'rho': 1, 'phi': 2, 'z': 3}, {})

    >>> params = {"r": 1, "theta": 2, "phi": 3}
    >>> cxv.vconvert(cxv.SphericalPos, cxv.SphericalPos, params)
    ({'r': 1, 'theta': 2, 'phi': 3}, {})

    >>> params = {"lon": 1, "lat": 2, "distance": 3}
    >>> cxv.vconvert(cxv.LonLatSphericalPos, cxv.LonLatSphericalPos, params)
    ({'lon': 1, 'lat': 2, 'distance': 3}, {})

    >>> params = {"r": 1, "theta": 2, "phi": 3}
    >>> cxv.vconvert(cxv.MathSphericalPos, cxv.MathSphericalPos, params)
    ({'r': 1, 'theta': 2, 'phi': 3}, {})

    >>> params = {"rho": 1, "pp_phi": 2, "z": 3, "dt_rho": 4, "dt_pp_phi": 5, "dt_z": 6}
    >>> cxv.vconvert(cxv.PoincarePolarVector, cxv.PoincarePolarVector, params)
    ({'rho': 1, 'pp_phi': 2, 'z': 3, 'dt_rho': 4, 'dt_pp_phi': 5, 'dt_z': 6}, {})

    >>> params = {"q": jnp.array([1, 2, 3, 4])}
    >>> cxv.vconvert(cxv.CartesianPosND, cxv.CartesianPosND, params)
    ({'q': Array([1, 2, 3, 4], dtype=int32)}, {})

    """
    return params, combine_aux(in_aux, out_aux)


@dispatch.multi(  # TODO: is the precedence needed?
    # Positions
    (type[d1.CartesianPos1D], d1.CartesianPos1D),
    (type[d1.RadialPos], d1.RadialPos),
    (type[d2.CartesianPos2D], d2.CartesianPos2D),
    (type[d2.PolarPos], d2.PolarPos),
    (type[d2.TwoSpherePos], d2.TwoSpherePos),
    (type[d3.CartesianPos3D], d3.CartesianPos3D),
    (type[d3.CylindricalPos], d3.CylindricalPos),
    (type[d3.SphericalPos], d3.SphericalPos),
    (type[d3.LonLatSphericalPos], d3.LonLatSphericalPos),
    (type[d3.MathSphericalPos], d3.MathSphericalPos),
    (type[d4.FourVector], d4.FourVector),
    (type[dn.PoincarePolarVector], dn.PoincarePolarVector),
    (type[dn.CartesianPosND], dn.CartesianPosND),
    # Velocities
    (type[d1.CartesianVel1D], d1.CartesianVel1D, AbstractPos),
    (type[d1.RadialVel], d1.RadialVel, AbstractPos),
    (type[d1.CartesianVel1D], d1.CartesianVel1D),  # q not needed
    (type[d1.RadialVel], d1.RadialVel),  # q not needed
    (type[d2.CartesianVel2D], d2.CartesianVel2D, d2.AbstractPos2D),
    (type[d2.CartesianVel2D], d2.CartesianVel2D),  # q not needed
    (type[d2.PolarVel], d2.PolarVel, d2.AbstractPos2D),
    (type[d2.PolarVel], d2.PolarVel),
    (type[d2.TwoSphereVel], d2.TwoSphereVel, d2.AbstractPos2D),
    (type[d2.TwoSphereVel], d2.TwoSphereVel),  # q not needed
    (type[d3.CartesianVel3D], d3.CartesianVel3D, AbstractPos),
    (type[d3.CartesianVel3D], d3.CartesianVel3D),  # q not needed
    (type[d3.CylindricalVel], d3.CylindricalVel, AbstractPos),
    (type[d3.CylindricalVel], d3.CylindricalVel),  # q not needed
    (type[d3.SphericalVel], d3.SphericalVel, AbstractPos),
    (type[d3.SphericalVel], d3.SphericalVel),  # q not needed
    (type[d3.LonLatSphericalVel], d3.LonLatSphericalVel, AbstractPos),
    (type[d3.LonLatSphericalVel], d3.LonLatSphericalVel),  # q not needed
    (type[d3.LonCosLatSphericalVel], d3.LonCosLatSphericalVel, AbstractPos),
    (type[d3.LonCosLatSphericalVel], d3.LonCosLatSphericalVel),  # q not needed
    (type[d3.MathSphericalVel], d3.MathSphericalVel, AbstractPos),
    (type[d3.MathSphericalVel], d3.MathSphericalVel),  # q not needed
    (type[d3.ProlateSpheroidalVel], d3.ProlateSpheroidalVel, AbstractPos),
    (type[d3.ProlateSpheroidalVel], d3.ProlateSpheroidalVel),  # q not needed
    (type[dn.CartesianVelND], dn.CartesianVelND, dn.AbstractPosND),
    (type[dn.CartesianVelND], dn.CartesianVelND),  # q not needed
    # Accelerations
    (type[d1.CartesianAcc1D], d1.CartesianAcc1D, AbstractVel, AbstractPos),
    (type[d1.RadialAcc], d1.RadialAcc, AbstractVel, AbstractPos),
    (type[d1.CartesianAcc1D], d1.CartesianAcc1D),  # q,p not needed
    (type[d1.RadialAcc], d1.RadialAcc),  # q,p not needed
    (type[d2.CartesianAcc2D], d2.CartesianAcc2D, d2.AbstractVel2D, d2.AbstractPos2D),
    (type[d2.CartesianAcc2D], d2.CartesianAcc2D),  # q,p not needed
    (type[d3.CartesianAcc3D], d3.CartesianAcc3D, AbstractVel, AbstractPos),
    (type[d3.CartesianAcc3D], d3.CartesianAcc3D),  # q,p not needed
    (type[d3.CylindricalAcc], d3.CylindricalAcc, AbstractVel, AbstractPos),
    (type[d3.CylindricalAcc], d3.CylindricalAcc),  # q,p not needed
    (type[d3.SphericalAcc], d3.SphericalAcc, AbstractVel, AbstractPos),
    (type[d3.SphericalAcc], d3.SphericalAcc),  # q,p not needed
    (type[d3.LonLatSphericalAcc], d3.LonLatSphericalAcc, AbstractVel, AbstractPos),
    (type[d3.LonLatSphericalAcc], d3.LonLatSphericalAcc),  # q,p not needed
    (type[d3.MathSphericalAcc], d3.MathSphericalAcc, AbstractVel, AbstractPos),
    (type[d3.MathSphericalAcc], d3.MathSphericalAcc),  # q,p not needed
    (type[dn.CartesianAccND], dn.CartesianAccND, dn.AbstractVelND, dn.AbstractPosND),
    (type[dn.CartesianAccND], dn.CartesianAccND),  # q,p not needed
)
def vconvert(
    target: type[AbstractVector], current: AbstractVector, /, *args: Any, **kw: Any
) -> AbstractVector:
    """Self transforms.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    ## 1D:

    >>> q = cxv.CartesianPos1D.from_(1, "m")
    >>> cxv.vconvert(cxv.CartesianPos1D, q) is q
    True

    >>> q = cxv.RadialPos.from_(1, "m")
    >>> cxv.vconvert(cxv.RadialPos, q) is q
    True

    >>> p = cxv.CartesianVel1D.from_(1, "m/s")
    >>> cxv.vconvert(cxv.CartesianVel1D, p) is p
    True
    >>> cxv.vconvert(cxv.CartesianVel1D, p, q) is p
    True

    >>> p = cxv.RadialVel.from_(1, "m/s")
    >>> cxv.vconvert(cxv.RadialVel, p) is p
    True
    >>> cxv.vconvert(cxv.RadialVel, p, q) is p
    True

    >>> a = cxv.CartesianAcc1D.from_(1, "m/s2")
    >>> cxv.vconvert(cxv.CartesianAcc1D, a) is a
    True
    >>> cxv.vconvert(cxv.CartesianAcc1D, a, p, q) is a
    True

    >>> a = cxv.RadialAcc.from_(1, "m/s2")
    >>> cxv.vconvert(cxv.RadialAcc, a) is a
    True
    >>> cxv.vconvert(cxv.RadialAcc, a, p, q) is a
    True

    ## 2D:

    >>> q = cxv.CartesianPos2D.from_([1, 2], "m")
    >>> cxv.vconvert(cxv.CartesianPos2D, q) is q
    True

    >>> q = cxv.PolarPos(r=u.Quantity(1, "m"), phi=u.Quantity(10, "deg"))
    >>> cxv.vconvert(cxv.PolarPos, q) is q
    True

    >>> q = cxv.TwoSpherePos(theta=u.Quantity(10, "deg"), phi=u.Quantity(14, "deg"))
    >>> cxv.vconvert(cxv.TwoSpherePos, q) is q
    True

    >>> p = cxv.CartesianVel2D.from_([1, 2], "m/s")
    >>> cxv.vconvert(cxv.CartesianVel2D, p) is p
    True

    >>> p = cxv.PolarVel(r=u.Quantity(1, "m/s"), phi=u.Quantity(10, "deg/s"))
    >>> cxv.vconvert(cxv.PolarVel, p) is p
    True

    >>> a = cxv.CartesianAcc2D.from_([1, 2], "m/s2")
    >>> cxv.vconvert(cxv.CartesianAcc2D, a) is a
    True

    ## 3D:

    Cartesian to Cartesian:

    >>> vec = cxv.CartesianPos3D.from_([1, 2, 3], "km")
    >>> cxv.vconvert(cxv.CartesianPos3D, vec) is vec
    True

    Cylindrical to Cylindrical:

    >>> vec = cxv.CylindricalPos(rho=u.Quantity(1, "km"),
    ...                          phi=u.Quantity(2, "deg"),
    ...                          z=u.Quantity(3, "km"))
    >>> cxv.vconvert(cxv.CylindricalPos, vec) is vec
    True

    Spherical to Spherical:

    >>> vec = cxv.SphericalPos(r=u.Quantity(1, "km"),
    ...                        theta=u.Quantity(2, "deg"),
    ...                        phi=u.Quantity(3, "deg"))
    >>> cxv.vconvert(cxv.SphericalPos, vec) is vec
    True

    LonLatSpherical to LonLatSpherical:

    >>> vec = cxv.LonLatSphericalPos(lon=u.Quantity(1, "deg"),
    ...                              lat=u.Quantity(2, "deg"),
    ...                              distance=u.Quantity(3, "km"))
    >>> cxv.vconvert(cxv.LonLatSphericalPos, vec) is vec
    True

    MathSpherical to MathSpherical:

    >>> vec = cxv.MathSphericalPos(r=u.Quantity(1, "km"),
    ...                            theta=u.Quantity(2, "deg"),
    ...                            phi=u.Quantity(3, "deg"))
    >>> cxv.vconvert(cxv.MathSphericalPos, vec) is vec
    True

    For these transformations the position does not matter since the
    self-transform returns the velocity unchanged.

    >>> vec = cxv.CartesianPos3D.from_([1, 2, 3], "km")

    Cartesian to Cartesian velocity:

    >>> dif = cxv.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> cxv.vconvert(cxv.CartesianVel3D, dif, vec) is dif
    True

    Cylindrical to Cylindrical velocity:

    >>> dif = cxv.CylindricalVel(rho=u.Quantity(1, "km/s"),
    ...                          phi=u.Quantity(2, "mas/yr"),
    ...                          z=u.Quantity(3, "km/s"))
    >>> cxv.vconvert(cxv.CylindricalVel, dif, vec) is dif
    True

    Spherical to Spherical velocity:

    >>> dif = cxv.SphericalVel(r=u.Quantity(1, "km/s"),
    ...                        theta=u.Quantity(2, "mas/yr"),
    ...                        phi=u.Quantity(3, "mas/yr"))
    >>> cxv.vconvert(cxv.SphericalVel, dif, vec) is dif
    True

    LonLatSpherical to LonLatSpherical velocity:

    >>> dif = cxv.LonLatSphericalVel(lon=u.Quantity(1, "mas/yr"),
    ...                              lat=u.Quantity(2, "mas/yr"),
    ...                              distance=u.Quantity(3, "km/s"))
    >>> cxv.vconvert(cxv.LonLatSphericalVel, dif, vec) is dif
    True

    LonCosLatSpherical to LonCosLatSpherical velocity:

    >>> dif = cxv.LonCosLatSphericalVel(lon_coslat=u.Quantity(1, "mas/yr"),
    ...                                 lat=u.Quantity(2, "mas/yr"),
    ...                                 distance=u.Quantity(3, "km/s"))
    >>> cxv.vconvert(cxv.LonCosLatSphericalVel, dif, vec) is dif
    True

    MathSpherical to MathSpherical velocity:

    >>> dif = cxv.MathSphericalVel(r=u.Quantity(1, "km/s"),
    ...                            theta=u.Quantity(2, "mas/yr"),
    ...                            phi=u.Quantity(3, "mas/yr"))
    >>> cxv.vconvert(cxv.MathSphericalVel, dif, vec) is dif
    True

    Similarly for the these accelerations:

    >>> a = cxv.CartesianAcc3D.from_([1, 1, 1], "m/s2")
    >>> cxv.vconvert(cxv.CartesianAcc3D, a) is a
    True

    ## N-D:

    >>> x = cxv.CartesianPosND.from_([1, 2, 3, 4], "km")
    >>> cxv.vconvert(cxv.CartesianPosND, x) is x
    True

    >>> v = cxv.CartesianVelND.from_([1, 2, 3, 4], "km/s")
    >>> cxv.vconvert(cxv.CartesianVelND, v, x) is v
    True
    >>> cxv.vconvert(cxv.CartesianVelND, v) is v
    True

    >>> a = cxv.CartesianAccND.from_([1, 2, 3, 4], "m/s2")
    >>> cxv.vconvert(cxv.CartesianAccND, a, v, x) is a
    True
    >>> cxv.vconvert(cxv.CartesianAccND, a) is a
    True

    """
    return current


# =============================================================================
# Generic Convert


@dispatch
def vconvert(
    to_vector: type[AbstractPos],
    from_vector: type[AbstractPos],
    params: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """AbstractPos -> CartesianPos1D -> AbstractPos."""
    params, aux = vconvert(
        from_vector.cartesian_type,
        from_vector,
        params,
        in_aux=in_aux,
        out_aux=None,
        units=units,
    )
    params, aux = vconvert(
        to_vector.cartesian_type,
        from_vector.cartesian_type,
        params,
        in_aux=aux,
        out_aux=aux,
        units=units,
    )
    params, aux = vconvert(
        to_vector,
        to_vector.cartesian_type,
        params,
        in_aux=aux,
        out_aux=out_aux,
        units=units,
    )
    return params, aux


# =============================================================================
# Dimension drop


@dispatch.multi(
    (type[d1.CartesianPos1D], type[d2.CartesianPos2D], ct.ParamsDict),
    (type[d1.CartesianPos1D], type[d3.CartesianPos3D], ct.ParamsDict),
    (type[d1.RadialPos], type[d2.PolarPos], ct.ParamsDict),
    (type[d2.CartesianPos2D], type[d3.CartesianPos3D], ct.ParamsDict),
    (type[d1.RadialPos], type[d3.SphericalPos], ct.ParamsDict),
    (type[d1.RadialPos], type[d3.MathSphericalPos], ct.ParamsDict),
)
@wrap_vconvert_impl_params
def vconvert(
    to_vector: type[AbstractPos],
    from_vector: type[AbstractPos],
    p: ct.ParamsDict,
    /,
    *,
    in_aux: ct.OptAuxDict = None,
    out_aux: ct.OptAuxDict = None,
    units: ct.OptUSys = None,
) -> tuple[ct.ParamsDict, ct.AuxDict]:
    """CartesianPos2D -> CartesianPos1D.

    The `y` and `z` coordinates are dropped.

    Examples
    --------
    >>> import warnings
    >>> import unxt as u
    >>> import coordinax.vecs as cxv

    ## 2D:

    >>> params = {"x": 1, "y": 2}

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.CartesianPos1D, cxv.CartesianPos2D, params)
    ({'x': 1}, {})

    >>> x = cxv.CartesianPos2D.from_([1, 2], "km")
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.CartesianPos1D, x)
    >>> print(y)
    <CartesianPos1D: (x) [km]
        [1]>

    >>> x = cxv.PolarPos(r=u.Quantity(1, "km"), phi=u.Quantity(10, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.RadialPos, x)
    >>> print(y)
    <RadialPos: (r) [km]
        [1]>

    ## 3D:

    >>> params = {"x": 1, "y": 2, "z": 3}

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.CartesianPos1D, cxv.CartesianPos3D, params)
    ({'x': 1}, {})

    >>> x = cxv.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.CartesianPos1D, x)
    >>> print(y)
    <CartesianPos1D: (x) [km]
        [1.]>

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.CartesianPos2D, cxv.CartesianPos3D, params)
    ({'x': 1, 'y': 2}, {})

    >>> x = cxv.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.CartesianPos2D, x)
    >>> print(y)
    <CartesianPos2D: (x, y) [km]
        [1. 2.]>

    >>> params = {"r": 1, "theta": 14, "phi": 10}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.RadialPos, cxv.SphericalPos,
    ...                       params, units=usys)
    ({'r': Array(1, dtype=int32, ...)}, {})

    >>> x = cxv.SphericalPos(r=u.Quantity(1, "km"),
    ...                     theta=u.Quantity(14, "deg"),
    ...                     phi=u.Quantity(10, "deg"))
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.RadialPos, x)
    >>> print(y)
    <RadialPos: (r) [km]
        [1]>

    >>> params = {"r": 1, "theta": 10, "phi": 14}
    >>> usys = u.unitsystem("km", "deg")

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     cxv.vconvert(cxv.RadialPos, cxv.MathSphericalPos,
    ...                       params, units=usys)
    ({'r': Array(1, dtype=int32, ...)}, {})

    >>> x = cxv.MathSphericalPos(r=u.Quantity(1, "km"),
    ...                          theta=u.Quantity(10, "deg"),
    ...                          phi=u.Quantity(14, "deg"))

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     y = cxv.vconvert(cxv.RadialPos, x)
    >>> print(y)
    <RadialPos: (r) [km]
        [1]>

    """
    warn("irreversible dimension change", IrreversibleDimensionChange, stacklevel=2)
    return {k: p[k] for k in to_vector.components}, (out_aux or {})
