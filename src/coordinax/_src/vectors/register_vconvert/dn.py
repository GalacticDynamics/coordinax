"""Transformations between representations."""

__all__: list[str] = []


from typing import cast

from plum import dispatch

import quaxed.numpy as jnp
import unxt as u

from coordinax._src.vectors.collection import Space
from coordinax._src.vectors.d3 import CylindricalPos, CylindricalVel
from coordinax._src.vectors.dn import PoincarePolarVector


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
      rho=Quantity([[2.23606801, 6.40312433]], unit='m'),
      pp_phi=Quantity([[0., 0.]], unit='m / s(1/2)'),
      z=Quantity([[3, 6]], unit='m'),
      dt_rho=Quantity([[2.23606801, 6.40312433]], unit='m / s'),
      dt_pp_phi=Quantity([[0., 0.]], unit='m / s(1/2)'),
      dt_z=Quantity([[3, 6]], unit='m / s')
    )

    >>> cx.vconvert(cx.Space, w)
    Space({
      'length': CartesianPos3D(
        x=Quantity([[1, 4]], unit='m'),
        y=Quantity([[2, 5]], unit='m'),
        z=Quantity([[3, 6]], unit='m')
      ),
      'speed': CartesianVel3D(
        x=Quantity([[1, 4]], unit='m / s'),
        y=Quantity([[2, 5]], unit='m / s'),
        z=Quantity([[3, 6]], unit='m / s')
      )
    })

    """
    phi = cast(u.AbstractQuantity, jnp.atan2(w.dt_pp_phi, w.pp_phi))
    dt_phi = (w.pp_phi**2 + w.dt_pp_phi**2) / 2 / w.rho**2  # TODO: note the abs

    return Space(
        length=CylindricalPos(rho=w.rho, z=w.z, phi=phi),
        speed=CylindricalVel(rho=w.dt_rho, z=w.dt_z, phi=dt_phi),
    )
