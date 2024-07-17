"""Compatibility via :func:`plum.convert`."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method

import quaxed.array_api as xp
from unxt import Quantity

from .base import AbstractPosition3D
from .cartesian import CartesianAcceleration3D, CartesianPosition3D, CartesianVelocity3D
from coordinax._utils import field_values, full_shaped

#####################################################################
# Quantity


@conversion_method(AbstractPosition3D, Quantity)  # type: ignore[misc]
def vec_to_q(obj: AbstractPosition3D, /) -> Shaped[Quantity["length"], "*batch 3"]:
    """`coordinax.AbstractPosition3D` -> `unxt.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> from unxt import Quantity

    >>> vec = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], unit="kpc"))
    >>> convert(vec, Quantity)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='kpc')

    >>> vec = cx.SphericalPosition(r=Quantity(1, unit="kpc"),
    ...                          theta=Quantity(2, unit="deg"),
    ...                          phi=Quantity(3, unit="deg"))
    >>> convert(vec, Quantity)
    Quantity['length'](Array([0.03485167, 0.0018265 , 0.99939084], dtype=float32),
                       unit='kpc')

    >>> vec = cx.CylindricalPosition(rho=Quantity(1, unit="kpc"),
    ...                            phi=Quantity(2, unit="deg"),
    ...                            z=Quantity(3, unit="pc"))
    >>> convert(vec, Quantity)
    Quantity['length'](Array([0.99939084, 0.0348995 , 0.003     ], dtype=float32),
                       unit='kpc')

    """
    cart = full_shaped(obj.represent_as(CartesianPosition3D))
    return xp.stack(tuple(field_values(cart)), axis=-1)


@conversion_method(CartesianAcceleration3D, Quantity)  # type: ignore[misc]
@conversion_method(CartesianVelocity3D, Quantity)  # type: ignore[misc]
def vec_diff_to_q(obj: CartesianVelocity3D, /) -> Shaped[Quantity["speed"], "*batch 3"]:
    """`coordinax.CartesianVelocity3D` -> `unxt.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> from unxt import Quantity

    >>> dif = cx.CartesianVelocity3D.constructor(Quantity([1, 2, 3], unit="km/s"))
    >>> convert(dif, Quantity)
    Quantity['speed'](Array([1., 2., 3.], dtype=float32), unit='km / s')

    >>> dif2 = cx.CartesianAcceleration3D.constructor(Quantity([1, 2, 3], unit="km/s2"))
    >>> convert(dif2, Quantity)
    Quantity['acceleration'](Array([1., 2., 3.], dtype=float32), unit='km / s2')

    """
    return xp.stack(tuple(field_values(full_shaped(obj))), axis=-1)
