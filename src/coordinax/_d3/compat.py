"""Compatibility via :func:`plum.convert`."""

__all__: list[str] = []


import astropy.coordinates as apyc
import astropy.units as apyu
from jaxtyping import Shaped
from plum import add_conversion_method, conversion_method, convert

import quaxed.array_api as xp
from unxt import Quantity

from .base import Abstract3DVector
from .builtin import (
    Cartesian3DVector,
    CartesianDifferential3D,
    CylindricalDifferential,
    CylindricalVector,
    SphericalDifferential,
    SphericalVector,
)
from coordinax._utils import dataclass_values, full_shaped

#####################################################################
# Constructors


@Cartesian3DVector.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[Cartesian3DVector], obj: apyc.BaseRepresentation
) -> Cartesian3DVector:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> from astropy.coordinates import CartesianRepresentation
    >>> from coordinax import Cartesian3DVector

    >>> cart = CartesianRepresentation(1, 2, 3, unit="kpc")
    >>> vec = Cartesian3DVector.constructor(cart)
    >>> vec.x
    Quantity['length'](Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.CartesianRepresentation)
    return cls(x=obj.x, y=obj.y, z=obj.z)


@SphericalVector.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[SphericalVector], obj: apyc.BaseRepresentation
) -> SphericalVector:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import PhysicsSphericalRepresentation
    >>> from coordinax import SphericalVector

    >>> sph = PhysicsSphericalRepresentation(r=1 * u.kpc, theta=2 * u.deg,
    ...                                      phi=3 * u.deg)
    >>> vec = SphericalVector.constructor(sph)
    >>> vec.r
    Quantity['length'](Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.PhysicsSphericalRepresentation)
    return cls(r=obj.r, phi=obj.phi, theta=obj.theta)


@CylindricalVector.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[CylindricalVector], obj: apyc.BaseRepresentation
) -> CylindricalVector:
    """Construct from a :class:`astropy.coordinates.BaseRepresentation`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import CylindricalRepresentation
    >>> from coordinax import CylindricalVector

    >>> cyl = CylindricalRepresentation(rho=1 * u.kpc, phi=2 * u.deg,
    ...                                 z=30 * u.pc)
    >>> vec = CylindricalVector.constructor(cyl)
    >>> vec.rho
    Quantity['length'](Array(1., dtype=float32), unit='kpc')

    """
    obj = obj.represent_as(apyc.CylindricalRepresentation)
    return cls(rho=obj.rho, phi=obj.phi, z=obj.z)


@CartesianDifferential3D.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[CartesianDifferential3D], obj: apyc.CartesianDifferential
) -> CartesianDifferential3D:
    """Construct from a :class:`astropy.coordinates.CartesianDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import CartesianDifferential
    >>> from coordinax import CartesianDifferential3D

    >>> dcart = CartesianDifferential(1, 2, 3, unit="km/s")
    >>> dif = CartesianDifferential3D.constructor(dcart)
    >>> dif.d_x
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_x=obj.d_x, d_y=obj.d_y, d_z=obj.d_z)


@SphericalDifferential.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[SphericalDifferential], obj: apyc.PhysicsSphericalDifferential
) -> SphericalDifferential:
    """Construct from a :class:`astropy.coordinates.PhysicsSphericalDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import PhysicsSphericalDifferential
    >>> from coordinax import SphericalDifferential

    >>> dsph = PhysicsSphericalDifferential(d_r=1 * u.km / u.s, d_theta=2 * u.mas/u.yr,
    ...                                     d_phi=3 * u.mas/u.yr)
    >>> dif = SphericalDifferential.constructor(dsph)
    >>> dif.d_r
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_r=obj.d_r, d_phi=obj.d_phi, d_theta=obj.d_theta)


@CylindricalDifferential.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[CylindricalDifferential], obj: apyc.CylindricalDifferential
) -> CylindricalDifferential:
    """Construct from a :class:`astropy.coordinates.CylindricalDifferential`.

    Examples
    --------
    >>> import astropy.units as u
    >>> import astropy.coordinates as apyc
    >>> from coordinax import CylindricalDifferential

    >>> dcyl = apyc.CylindricalDifferential(d_rho=1 * u.km / u.s, d_phi=2 * u.mas/u.yr,
    ...                                     d_z=2 * u.km / u.s)
    >>> dif = CylindricalDifferential.constructor(dcyl)
    >>> dif.d_rho
    Quantity['speed'](Array(1., dtype=float32), unit='km / s')

    """
    return cls(d_rho=obj.d_rho, d_phi=obj.d_phi, d_z=obj.d_z)


#####################################################################
# Quantity


@conversion_method(Abstract3DVector, Quantity)  # type: ignore[misc]
def vec_to_q(obj: Abstract3DVector, /) -> Shaped[Quantity["length"], "*batch 3"]:
    """`coordinax.Abstract3DVector` -> `unxt.Quantity`."""
    cart = full_shaped(obj.represent_as(Cartesian3DVector))
    return xp.stack(tuple(dataclass_values(cart)), axis=-1)


@conversion_method(CartesianDifferential3D, Quantity)  # type: ignore[misc]
def vec_diff_to_q(
    obj: CartesianDifferential3D, /
) -> Shaped[Quantity["speed"], "*batch 3"]:
    """`coordinax.CartesianDifferential3D` -> `unxt.Quantity`."""
    return xp.stack(tuple(dataclass_values(full_shaped(obj))), axis=-1)


#####################################################################
# Astropy


# =====================================
# Cartesian3DVector


# @conversion_method(Cartesian3DVector, apyc.BaseRepresentation)
# @conversion_method(Cartesian3DVector, apyc.CartesianRepresentation)
def cart3_to_apycart3(obj: Cartesian3DVector, /) -> apyc.CartesianRepresentation:
    """`coordinax.Cartesian3DVector` -> `astropy.CartesianRepresentation`."""
    return apyc.CartesianRepresentation(
        x=convert(obj.x, apyu.Quantity),
        y=convert(obj.y, apyu.Quantity),
        z=convert(obj.z, apyu.Quantity),
    )


# TODO: use decorator when https://github.com/beartype/plum/pull/135
add_conversion_method(Cartesian3DVector, apyc.BaseRepresentation, cart3_to_apycart3)
add_conversion_method(
    Cartesian3DVector, apyc.CartesianRepresentation, cart3_to_apycart3
)


@conversion_method(apyc.CartesianRepresentation, Cartesian3DVector)  # type: ignore[misc]
def apycart3_to_cart3(obj: apyc.CartesianRepresentation, /) -> Cartesian3DVector:
    """`astropy.CartesianRepresentation` -> `coordinax.Cartesian3DVector`."""
    return Cartesian3DVector.constructor(obj)


# =====================================
# SphericalVector


# @conversion_method(SphericalVector, apyc.BaseRepresentation)
# @conversion_method(
#     SphericalVector, apyc.PhysicsSphericalRepresentation
# )
def sph_to_apysph(obj: SphericalVector, /) -> apyc.PhysicsSphericalRepresentation:
    """`coordinax.SphericalVector` -> `astropy.PhysicsSphericalRepresentation`."""
    return apyc.PhysicsSphericalRepresentation(
        r=convert(obj.r, apyu.Quantity),
        phi=convert(obj.phi, apyu.Quantity),
        theta=convert(obj.theta, apyu.Quantity),
    )


# TODO: use decorator when https://github.com/beartype/plum/pull/135
add_conversion_method(SphericalVector, apyc.BaseRepresentation, sph_to_apysph)
add_conversion_method(
    SphericalVector, apyc.PhysicsSphericalRepresentation, sph_to_apysph
)


@conversion_method(apyc.PhysicsSphericalRepresentation, SphericalVector)  # type: ignore[misc]
def apysph_to_sph(obj: apyc.PhysicsSphericalRepresentation, /) -> SphericalVector:
    """`astropy.PhysicsSphericalRepresentation` -> `coordinax.SphericalVector`."""
    return SphericalVector.constructor(obj)


# =====================================
# CylindricalVector


# @conversion_method(CylindricalVector, apyc.BaseRepresentation)
# @conversion_method(CylindricalVector, apyc.CylindricalRepresentation)
def cyl_to_apycyl(obj: CylindricalVector, /) -> apyc.CylindricalRepresentation:
    """`coordinax.CylindricalVector` -> `astropy.CylindricalRepresentation`."""
    return apyc.CylindricalRepresentation(
        rho=convert(obj.rho, apyu.Quantity),
        phi=convert(obj.phi, apyu.Quantity),
        z=convert(obj.z, apyu.Quantity),
    )


# TODO: use decorator when https://github.com/beartype/plum/pull/135
add_conversion_method(CylindricalVector, apyc.BaseRepresentation, cyl_to_apycyl)
add_conversion_method(CylindricalVector, apyc.CylindricalRepresentation, cyl_to_apycyl)


@conversion_method(apyc.CylindricalRepresentation, CylindricalVector)  # type: ignore[misc]
def apycyl_to_cyl(obj: apyc.CylindricalRepresentation, /) -> CylindricalVector:
    """`astropy.CylindricalRepresentation` -> `coordinax.CylindricalVector`."""
    return CylindricalVector.constructor(obj)


# =====================================
# CartesianDifferential3D


# @conversion_method(CartesianDifferential3D, apyc.BaseDifferential)
# @conversion_method(
#     CartesianDifferential3D, apyc.CartesianDifferential
# )
def diffcart3_to_apycart3(
    obj: CartesianDifferential3D, /
) -> apyc.CartesianDifferential:
    """`coordinax.CartesianDifferential3D` -> `astropy.CartesianDifferential`."""
    return apyc.CartesianDifferential(
        d_x=convert(obj.d_x, apyu.Quantity),
        d_y=convert(obj.d_y, apyu.Quantity),
        d_z=convert(obj.d_z, apyu.Quantity),
    )


# TODO: use decorator when https://github.com/beartype/plum/pull/135
add_conversion_method(
    CartesianDifferential3D, apyc.BaseDifferential, diffcart3_to_apycart3
)
add_conversion_method(
    CartesianDifferential3D, apyc.CartesianDifferential, diffcart3_to_apycart3
)


@conversion_method(  # type: ignore[misc]
    apyc.CartesianDifferential, CartesianDifferential3D
)
def apycart3_to_diffcart3(
    obj: apyc.CartesianDifferential, /
) -> CartesianDifferential3D:
    """`astropy.CartesianDifferential` -> `coordinax.CartesianDifferential3D`."""
    return CartesianDifferential3D.constructor(obj)


# =====================================
# SphericalDifferential


# @conversion_method(SphericalDifferential, apyc.BaseDifferential)
# @conversion_method(
#     SphericalDifferential, apyc.PhysicsSphericalDifferential
# )
def diffsph_to_apysph(
    obj: SphericalDifferential, /
) -> apyc.PhysicsSphericalDifferential:
    """`coordinax.SphericalDifferential` -> `astropy.PhysicsSphericalDifferential`."""
    return apyc.PhysicsSphericalDifferential(
        d_r=convert(obj.d_r, apyu.Quantity),
        d_phi=convert(obj.d_phi, apyu.Quantity),
        d_theta=convert(obj.d_theta, apyu.Quantity),
    )


# TODO: use decorator when https://github.com/beartype/plum/pull/135
add_conversion_method(SphericalDifferential, apyc.BaseDifferential, diffsph_to_apysph)
add_conversion_method(
    SphericalDifferential, apyc.PhysicsSphericalDifferential, diffsph_to_apysph
)


@conversion_method(  # type: ignore[misc]
    apyc.PhysicsSphericalDifferential, SphericalDifferential
)
def apysph_to_diffsph(
    obj: apyc.PhysicsSphericalDifferential, /
) -> SphericalDifferential:
    """`astropy.PhysicsSphericalDifferential` -> `coordinax.SphericalDifferential`."""
    return SphericalDifferential.constructor(obj)


# =====================================
# CylindricalDifferential


# @conversion_method(CylindricalDifferential, apyc.BaseDifferential)
# @conversion_method(
#     CylindricalDifferential, apyc.CylindricalDifferential
# )
def diffcyl_to_apycyl(obj: CylindricalDifferential, /) -> apyc.CylindricalDifferential:
    """`coordinax.CylindricalDifferential` -> `astropy.CylindricalDifferential`."""
    return apyc.CylindricalDifferential(
        d_rho=convert(obj.d_rho, apyu.Quantity),
        d_phi=convert(obj.d_phi, apyu.Quantity),
        d_z=convert(obj.d_z, apyu.Quantity),
    )


# TODO: use decorator when https://github.com/beartype/plum/pull/135
add_conversion_method(CylindricalDifferential, apyc.BaseDifferential, diffcyl_to_apycyl)
add_conversion_method(
    CylindricalDifferential, apyc.CylindricalDifferential, diffcyl_to_apycyl
)


@conversion_method(  # type: ignore[misc]
    apyc.CylindricalDifferential, CylindricalDifferential
)
def apycyl_to_diffcyl(obj: apyc.CylindricalDifferential, /) -> CylindricalDifferential:
    """`astropy.CylindricalDifferential` -> `coordinax.CylindricalDifferential`."""
    return CylindricalDifferential.constructor(obj)


#####################################################################
