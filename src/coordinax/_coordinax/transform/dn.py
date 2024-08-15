"""Transformations between representations."""

__all__ = ["represent_as"]


from plum import dispatch

import quaxed.array_api as xp

from coordinax._coordinax.d3.cylindrical import (
    CylindricalPosition,
    CylindricalVelocity,
)
from coordinax._coordinax.dn.poincare import PoincarePolarVector
from coordinax._coordinax.space import Space


@dispatch  # type: ignore[misc]
def represent_as(w: PoincarePolarVector, target: type[Space], /) -> Space:
    """Space -> PoincarePolarVector.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import Quantity

    >>> w = cx.Space(
    ...     length=cx.CartesianPosition3D.constructor([[[1, 2, 3], [4, 5, 6]]], "m"),
    ...     speed=cx.CartesianVelocity3D.constructor([[[1, 2, 3], [4, 5, 6]]], "m/s")
    ... )

    >>> cx.represent_as(w, cx.PoincarePolarVector)
    PoincarePolarVector(
        rho=Quantity[...](value=f32[1,2], unit=Unit("m")),
        pp_phi=Quantity[...]( value=f32[1,2], unit=Unit("m rad(1/2) / s(1/2)") ),
        z=Quantity[...](value=f32[1,2], unit=Unit("m")),
        d_rho=Quantity[...]( value=f32[1,2], unit=Unit("m / s") ),
        d_pp_phi=Quantity[...]( value=f32[1,2], unit=Unit("m rad(1/2) / s(1/2)") ),
        d_z=Quantity[...]( value=f32[1,2], unit=Unit("m / s") )
    )

    >>> cx.represent_as(w, cx.Space)
    Space({
        'length': CartesianPosition3D(
            x=Quantity[...](value=f32[1,2], unit=Unit("m")),
            y=Quantity[...](value=f32[1,2], unit=Unit("m")),
            z=Quantity[...](value=f32[1,2], unit=Unit("m"))
        ),
        'speed': CartesianVelocity3D(
            d_x=Quantity[...]( value=f32[1,2], unit=Unit("m / s") ),
            d_y=Quantity[...]( value=f32[1,2], unit=Unit("m / s") ),
            d_z=Quantity[...]( value=f32[1,2], unit=Unit("m / s") )
        )} )

    """
    phi = xp.atan2(w.d_pp_phi, w.pp_phi)
    d_phi = (w.pp_phi**2 + w.d_pp_phi**2) / 2 / w.rho**2  # TODO: note the abs

    return Space(
        length=CylindricalPosition(rho=w.rho, z=w.z, phi=phi),
        speed=CylindricalVelocity(d_rho=w.d_rho, d_z=w.d_z, d_phi=d_phi),
    )
