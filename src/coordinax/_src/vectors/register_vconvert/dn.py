"""Transformations between representations."""

__all__: tuple[str, ...] = ()


from typing import TYPE_CHECKING, cast

from plum import dispatch

import quaxed.numpy as jnp

from coordinax._src.vectors.collection import KinematicSpace
from coordinax._src.vectors.d3 import CylindricalPos, CylindricalVel
from coordinax._src.vectors.dn import PoincarePolarVector

if TYPE_CHECKING:
    import unxt  # noqa: ICN001


@dispatch  # TODO: KinematicSpace[CylindricalPos] -- plum#212
def vconvert(target: type[KinematicSpace], w: PoincarePolarVector, /) -> KinematicSpace:
    """Space -> PoincarePolarVector.

    Examples
    --------
    >>> import coordinax as cx

    >>> w = cx.KinematicSpace(
    ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
    ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
    ... )

    >>> cx.vconvert(cx.vecs.PoincarePolarVector, w)
    PoincarePolarVector(
      rho=Q([[2.23606801, 6.40312433]], 'm'),
      pp_phi=Q([[0., 0.]], 'm / s(1/2)'),
      z=Q([[3, 6]], 'm'),
      dt_rho=Q([[2.23606801, 6.40312433]], 'm / s'),
      dt_pp_phi=Q([[0., 0.]], 'm / s(1/2)'),
      dt_z=Q([[3, 6]], 'm / s')
    )

    >>> cx.vconvert(cx.KinematicSpace, w)
    KinematicSpace({
      'length': CartesianPos3D(x=Q([[1, 4]], 'm'), y=Q([[2, 5]], 'm'),
                               z=Q([[3, 6]], 'm')),
      'speed': CartesianVel3D(x=Q([[1, 4]], 'm / s'), y=Q([[2, 5]], 'm / s'),
                              z=Q([[3, 6]], 'm / s'))
    })

    """
    phi = cast("unxt.AbstractQuantity", jnp.atan2(w.dt_pp_phi, w.pp_phi))
    dt_phi = (w.pp_phi**2 + w.dt_pp_phi**2) / 2 / w.rho**2  # TODO: note the abs

    return KinematicSpace(
        length=CylindricalPos(rho=w.rho, z=w.z, phi=phi),
        speed=CylindricalVel(rho=w.dt_rho, z=w.dt_z, phi=dt_phi),
    )
