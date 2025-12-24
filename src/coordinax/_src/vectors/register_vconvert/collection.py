"""Transformations between representations."""

__all__: tuple[str, ...] = ()


from plum import dispatch

import quaxed.numpy as jnp

from coordinax._src.vectors.collection import KinematicSpace
from coordinax._src.vectors.d3 import CartesianPos3D, CartesianVel3D
from coordinax._src.vectors.dn import PoincarePolarVector


@dispatch  # TODO: KinematicSpace[PosT] for all types -- plum#212
def vconvert(target: type[KinematicSpace], w: KinematicSpace, /) -> KinematicSpace:
    """Space -> KinematicSpace.

    Examples
    --------
    >>> import coordinax as cx

    >>> w = cx.KinematicSpace(
    ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
    ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
    ... )

    >>> cx.vconvert(cx.KinematicSpace, w)
    KinematicSpace({ 'length': CartesianPos3D(...),
             'speed': CartesianVel3D(...) })

    """
    return w


@dispatch
def vconvert(
    target: type[PoincarePolarVector], w: KinematicSpace, /
) -> PoincarePolarVector:
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

    """
    q = w["length"].vconvert(CartesianPos3D)
    p = w["speed"].vconvert(CartesianVel3D, q)

    rho = jnp.hypot(q.x, q.y)
    phi = jnp.arctan2(q.x, q.y)

    v_rho = (q.x * p.x + q.y * p.y) / rho
    v_phi = q.x * p.y - q.y * p.x

    # pg. 437, Papaphillipou & Laskar (1996)
    sqrt2theta = jnp.sqrt(jnp.abs(2 * v_phi))
    pp_phi = sqrt2theta * jnp.cos(phi)
    pp_phidot = sqrt2theta * jnp.sin(phi)

    return PoincarePolarVector(
        rho=rho, pp_phi=pp_phi, z=q.z, dt_rho=v_rho, dt_pp_phi=pp_phidot, dt_z=p.z
    )
