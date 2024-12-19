"""Transformations between representations."""

__all__: list[str] = []


from plum import dispatch

import quaxed.numpy as jnp

from coordinax._src.vectors.d3 import CylindricalPos, CylindricalVel
from coordinax._src.vectors.dn import PoincarePolarVector
from coordinax._src.vectors.space import Space


@dispatch  # type: ignore[misc]
def vconvert(target: type[Space], w: PoincarePolarVector, /) -> Space:
    """Space -> PoincarePolarVector.

    Examples
    --------
    >>> import coordinax as cx

    >>> w = cx.Space(
    ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
    ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
    ... )

    >>> cx.vconvert(cx.vecs.PoincarePolarVector, w)
    PoincarePolarVector(
        rho=Quantity[...](value=f32[1,2], unit=Unit("m")),
        pp_phi=Quantity[...]( value=f32[1,2], unit=Unit("m rad(1/2) / s(1/2)") ),
      z=Quantity[...](value=i32[1,2], unit=Unit("m")),
      d_rho=Quantity[...]( value=f32[1,2], unit=Unit("m / s") ),
      d_pp_phi=Quantity[...]( value=f32[1,2], unit=Unit("m rad(1/2) / s(1/2)") ),
      d_z=Quantity[...]( value=f32[1,2], unit=Unit("m / s") )
    )

    >>> cx.vconvert(cx.Space, w)
    Space({
        'length': CartesianPos3D(
            x=Quantity[...](value=i32[1,2], unit=Unit("m")),
            y=Quantity[...](value=i32[1,2], unit=Unit("m")),
            z=Quantity[...](value=i32[1,2], unit=Unit("m"))
        ),
        'speed': CartesianVel3D(
            d_x=Quantity[...]( value=i32[1,2], unit=Unit("m / s") ),
            d_y=Quantity[...]( value=i32[1,2], unit=Unit("m / s") ),
            d_z=Quantity[...]( value=i32[1,2], unit=Unit("m / s") )
        )
    })

    """
    phi = jnp.atan2(w.d_pp_phi, w.pp_phi)
    d_phi = (w.pp_phi**2 + w.d_pp_phi**2) / 2 / w.rho**2  # TODO: note the abs

    return Space(
        length=CylindricalPos(rho=w.rho, z=w.z, phi=phi),
        speed=CylindricalVel(d_rho=w.d_rho, d_z=w.d_z, d_phi=d_phi),
    )
