"""Transformations between representations."""

__all__: list[str] = []


from plum import dispatch

import quaxed.numpy as jnp

from coordinax._src.d3.cylindrical import CylindricalPos, CylindricalVel
from coordinax._src.dn.poincare import PoincarePolarVector
from coordinax._src.space import Space


@dispatch
def represent_as(w: Space, target: type[Space]) -> Space:
    """Space -> Space.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import Quantity

    >>> w = cx.Space(
    ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
    ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
    ... )

    >>> cx.represent_as(w, cx.Space)
    Space({
        'length': CartesianPos3D( ... ),
        'speed': CartesianVel3D( ... )} )

    """
    return w


@dispatch
def represent_as(w: Space, target: type[PoincarePolarVector], /) -> PoincarePolarVector:
    """Space -> PoincarePolarVector.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import Quantity

    >>> w = cx.Space(
    ...     length=cx.CartesianPos3D.from_([[[1, 2, 3], [4, 5, 6]]], "m"),
    ...     speed=cx.CartesianVel3D.from_([[[1, 2, 3], [4, 5, 6]]], "m/s")
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

    """
    q = w["length"].represent_as(CylindricalPos)
    p = w["speed"].represent_as(CylindricalVel, q)

    # pg. 437, Papaphillipou & Laskar (1996)
    sqrt2theta = jnp.sqrt(jnp.abs(2 * q.rho**2 * p.d_phi))
    pp_phi = sqrt2theta * jnp.cos(q.phi)
    pp_phidot = sqrt2theta * jnp.sin(q.phi)

    return PoincarePolarVector(
        rho=q.rho, pp_phi=pp_phi, z=q.z, d_rho=p.d_rho, d_pp_phi=pp_phidot, d_z=p.d_z
    )
