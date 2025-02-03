"""Transformations between representations."""

__all__: list[str] = []


from typing import cast

from plum import dispatch

import quaxed.numpy as jnp
from unxt.quantity import AbstractQuantity

from coordinax._src.vectors.d3 import CylindricalPos, CylindricalVel
from coordinax._src.vectors.dn import PoincarePolarVector
from coordinax._src.vectors.space import Space


@dispatch
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
      dt_rho=Quantity[...]( value=f32[1,2], unit=Unit("m / s") ),
      dt_pp_phi=Quantity[...]( value=f32[1,2], unit=Unit("m rad(1/2) / s(1/2)") ),
      dt_z=Quantity[...]( value=f32[1,2], unit=Unit("m / s") )
    )

    >>> cx.vconvert(cx.Space, w)
    Space({
        'length': CartesianPos3D(
            x=Quantity[...](value=i32[1,2], unit=Unit("m")),
            y=Quantity[...](value=i32[1,2], unit=Unit("m")),
            z=Quantity[...](value=i32[1,2], unit=Unit("m"))
        ),
        'speed': CartesianVel3D(
            x=Quantity[...]( value=i32[1,2], unit=Unit("m / s") ),
            y=Quantity[...]( value=i32[1,2], unit=Unit("m / s") ),
            z=Quantity[...]( value=i32[1,2], unit=Unit("m / s") )
        )
    })

    """
    phi = cast(AbstractQuantity, jnp.atan2(w.dt_pp_phi, w.pp_phi))
    dt_phi = (w.pp_phi**2 + w.dt_pp_phi**2) / 2 / w.rho**2  # TODO: note the abs

    return Space(
        length=CylindricalPos(rho=w.rho, z=w.z, phi=phi),
        speed=CylindricalVel(rho=w.dt_rho, z=w.dt_z, phi=dt_phi),
    )
