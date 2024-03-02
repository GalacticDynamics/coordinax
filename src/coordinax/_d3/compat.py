"""Compatibility via :func:`plum.convert`."""

__all__: list[str] = []


import astropy.coordinates as apyc
import astropy.units as apyu
from jaxtyping import Shaped
from plum import conversion_method, convert

import array_api_jax_compat as xp
from jax_quantity import Quantity

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
# Using the registered `plum.convert`


@Cartesian3DVector.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[Cartesian3DVector], obj: apyc.CartesianRepresentation
) -> Cartesian3DVector:
    """Construct from a :class:`astropy.coordinates.CartesianRepresentation`.

    Examples
    --------
    >>> from astropy.coordinates import CartesianRepresentation
    >>> from coordinax import Cartesian3DVector

    >>> cart = CartesianRepresentation(1, 2, 3, unit="kpc")
    >>> vec = Cartesian3DVector.constructor(cart)
    >>> vec.x
    Quantity['length'](Array(1., dtype=float32), unit='kpc')

    """
    return convert(obj, cls)


@SphericalVector.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[SphericalVector], obj: apyc.PhysicsSphericalRepresentation
) -> SphericalVector:
    """Construct from a :class:`astropy.coordinates.PhysicsSphericalRepresentation`.

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
    return convert(obj, cls)


@CylindricalVector.constructor._f.register  # noqa: SLF001
def constructor(
    cls: type[CylindricalVector], obj: apyc.CylindricalRepresentation
) -> CylindricalVector:
    """Construct from a :class:`astropy.coordinates.CylindricalRepresentation`.

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
    return convert(obj, cls)


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
    return convert(obj, cls)


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
    return convert(obj, cls)


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
    return convert(obj, cls)


#####################################################################
# Quantity


@conversion_method(type_from=Abstract3DVector, type_to=Quantity)  # type: ignore[misc]
def vec_to_q(obj: Abstract3DVector, /) -> Shaped[Quantity["length"], "*batch 3"]:
    """`coordinax.Abstract3DVector` -> `jax_quantity.Quantity`."""
    cart = full_shaped(obj.represent_as(Cartesian3DVector))
    return xp.stack(tuple(dataclass_values(cart)), axis=-1)


@conversion_method(type_from=CartesianDifferential3D, type_to=Quantity)  # type: ignore[misc]
def vec_diff_to_q(
    obj: CartesianDifferential3D, /
) -> Shaped[Quantity["speed"], "*batch 3"]:
    """`coordinax.CartesianDifferential3D` -> `jax_quantity.Quantity`."""
    return xp.stack(tuple(dataclass_values(full_shaped(obj))), axis=-1)


#####################################################################
# Astropy


# =====================================
# Cartesian3DVector


@conversion_method(type_from=Cartesian3DVector, type_to=apyc.CartesianRepresentation)  # type: ignore[misc]
def cart3_to_apycart3(obj: Cartesian3DVector, /) -> apyc.CartesianRepresentation:
    """`coordinax.Cartesian3DVector` -> `astropy.CartesianRepresentation`."""
    return apyc.CartesianRepresentation(
        x=convert(obj.x, apyu.Quantity),
        y=convert(obj.y, apyu.Quantity),
        z=convert(obj.z, apyu.Quantity),
    )


@conversion_method(type_from=apyc.CartesianRepresentation, type_to=Cartesian3DVector)  # type: ignore[misc]
def apycart3_to_cart3(obj: apyc.CartesianRepresentation, /) -> Cartesian3DVector:
    """`astropy.CartesianRepresentation` -> `coordinax.Cartesian3DVector`."""
    return Cartesian3DVector(x=obj.x, y=obj.y, z=obj.z)


# =====================================
# SphericalVector


@conversion_method(
    type_from=SphericalVector,
    type_to=apyc.PhysicsSphericalRepresentation,  # type: ignore[misc]
)
def sph_to_apysph(obj: SphericalVector, /) -> apyc.PhysicsSphericalRepresentation:
    """`coordinax.SphericalVector` -> `astropy.PhysicsSphericalRepresentation`."""
    return apyc.PhysicsSphericalRepresentation(
        r=convert(obj.r, apyu.Quantity),
        phi=convert(obj.phi, apyu.Quantity),
        theta=convert(obj.theta, apyu.Quantity),
    )


@conversion_method(
    type_from=apyc.PhysicsSphericalRepresentation,
    type_to=SphericalVector,  # type: ignore[misc]
)
def apysph_to_sph(obj: apyc.PhysicsSphericalRepresentation, /) -> SphericalVector:
    """`astropy.PhysicsSphericalRepresentation` -> `coordinax.SphericalVector`."""
    return SphericalVector(r=obj.r, phi=obj.phi, theta=obj.theta)


# =====================================
# CylindricalVector


@conversion_method(type_from=CylindricalVector, type_to=apyc.CylindricalRepresentation)  # type: ignore[misc]
def cyl_to_apycyl(obj: CylindricalVector, /) -> apyc.CylindricalRepresentation:
    """`coordinax.CylindricalVector` -> `astropy.CylindricalRepresentation`."""
    return apyc.CylindricalRepresentation(
        rho=convert(obj.rho, apyu.Quantity),
        phi=convert(obj.phi, apyu.Quantity),
        z=convert(obj.z, apyu.Quantity),
    )


@conversion_method(type_from=apyc.CylindricalRepresentation, type_to=CylindricalVector)  # type: ignore[misc]
def apycyl_to_cyl(obj: apyc.CylindricalRepresentation, /) -> CylindricalVector:
    """`astropy.CylindricalRepresentation` -> `coordinax.CylindricalVector`."""
    return CylindricalVector(rho=obj.rho, phi=obj.phi, z=obj.z)


# =====================================
# CartesianDifferential3D


@conversion_method(  # type: ignore[misc]
    type_from=CartesianDifferential3D, type_to=apyc.CartesianDifferential
)
def diffcart3_to_apycart3(
    obj: CartesianDifferential3D, /
) -> apyc.CartesianDifferential:
    """`coordinax.CartesianDifferential3D` -> `astropy.CartesianDifferential`."""
    return apyc.CartesianDifferential(
        d_x=convert(obj.d_x, apyu.Quantity),
        d_y=convert(obj.d_y, apyu.Quantity),
        d_z=convert(obj.d_z, apyu.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    type_from=apyc.CartesianDifferential, type_to=CartesianDifferential3D
)
def apycart3_to_diffcart3(
    obj: apyc.CartesianDifferential, /
) -> CartesianDifferential3D:
    """`astropy.CartesianDifferential` -> `coordinax.CartesianDifferential3D`."""
    return CartesianDifferential3D(d_x=obj.d_x, d_y=obj.d_y, d_z=obj.d_z)


# =====================================
# SphericalDifferential


@conversion_method(  # type: ignore[misc]
    type_from=SphericalDifferential,
    type_to=apyc.PhysicsSphericalDifferential,
)
def diffsph_to_apysph(
    obj: SphericalDifferential, /
) -> apyc.PhysicsSphericalDifferential:
    """`coordinax.SphericalDifferential` -> `astropy.PhysicsSphericalDifferential`."""
    return apyc.PhysicsSphericalDifferential(
        d_r=convert(obj.d_r, apyu.Quantity),
        d_phi=convert(obj.d_phi, apyu.Quantity),
        d_theta=convert(obj.d_theta, apyu.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    type_from=apyc.PhysicsSphericalDifferential,
    type_to=SphericalDifferential,
)
def apysph_to_diffsph(
    obj: apyc.PhysicsSphericalDifferential, /
) -> SphericalDifferential:
    """`astropy.PhysicsSphericalDifferential` -> `coordinax.SphericalDifferential`."""
    return SphericalDifferential(d_r=obj.d_r, d_phi=obj.d_phi, d_theta=obj.d_theta)


# =====================================
# CylindricalDifferential


@conversion_method(  # type: ignore[misc]
    type_from=CylindricalDifferential, type_to=apyc.CylindricalDifferential
)
def diffcyl_to_apycyl(obj: CylindricalDifferential, /) -> apyc.CylindricalDifferential:
    """`coordinax.CylindricalDifferential` -> `astropy.CylindricalDifferential`."""
    return apyc.CylindricalDifferential(
        d_rho=convert(obj.d_rho, apyu.Quantity),
        d_phi=convert(obj.d_phi, apyu.Quantity),
        d_z=convert(obj.d_z, apyu.Quantity),
    )


@conversion_method(  # type: ignore[misc]
    type_from=apyc.CylindricalDifferential, type_to=CylindricalDifferential
)
def apycyl_to_diffcyl(obj: apyc.CylindricalDifferential, /) -> CylindricalDifferential:
    """`astropy.CylindricalDifferential` -> `coordinax.CylindricalDifferential`."""
    return CylindricalDifferential(d_rho=obj.d_rho, d_phi=obj.d_phi, d_z=obj.d_z)


#####################################################################
