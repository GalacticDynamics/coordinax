"""Transformations between representations."""

__all__: list[str] = []


from plum import dispatch

import quaxed.numpy as jnp

from coordinax._src.vectors.collection import Space
from coordinax._src.vectors.d3 import CartesianPos3D, CartesianVel3D
from coordinax._src.vectors.dn import PoincarePolarVector


@dispatch
def vconvert(target: type[Space], w: Space, /) -> Space:
    """Space -> Space.

    Examples
    --------
    >>> import coordinax as cx

    >>> w = cx.Space(
    ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
    ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
    ... )

    >>> cx.vconvert(cx.Space, w)
    Space({ 'length': CartesianPos3D( ... ),
             'speed': CartesianVel3D( ... ) })

    """
    return w


@dispatch
def vconvert(target: type[PoincarePolarVector], w: Space, /) -> PoincarePolarVector:
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
