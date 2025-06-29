"""Built-in 4-vector classes."""

__all__: list[str] = []


import jax.numpy as jnp
from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity

import coordinax.vecs as cxv
from .spacetime import FourVector


@conversion_method(type_from=FourVector, type_to=BareQuantity)
def fourvec_to_quantity(obj: FourVector, /) -> Shaped[BareQuantity, "*batch 4"]:
    """`coordinax.AbstractPos3D` -> `unxt.quantity.BareQuantity`.

    Convert the 4-vector to a Quantity array with the components as the last
    dimension.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.vecs.FourVector (t=u.Quantity([1, 2], "yr"),
    ...                        q=u.Quantity([[1, 2, 3], [4, 5, 6]], "pc"))

    >>> convert(w, u.quantity.BareQuantity).uconvert("pc")
    BareQuantity(Array([[0.3066014, 1. , 2. , 3. ],
                        [0.6132028, 4. , 4.9999995, 6. ]],
                       dtype=float32, ...),
                 unit='pc')

    """
    cart = convert(obj.q, BareQuantity)
    ct = convert(obj.c * obj.t[..., None], BareQuantity)
    return jnp.concat([ct, cart], axis=-1)


@conversion_method(type_from=FourVector, type_to=u.Quantity)
def fourvec_to_quantity(obj: FourVector, /) -> Shaped[u.Quantity["length"], "*batch 4"]:
    """`coordinax.AbstractPos3D` -> `unxt.Quantity`.

    Convert the 4-vector to a Quantity array with the components as the last
    dimension.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.vecs.FourVector (t=u.Quantity([1, 2], "yr"),
    ...                        q=u.Quantity([[1, 2, 3], [4, 5, 6]], "pc"))

    >>> convert(w, u.Quantity).uconvert("pc")
    Quantity(Array([[0.3066014, 1. , 2. , 3. ],
                              [0.6132028, 4. , 4.9999995, 6. ]],
                             dtype=float32, ...),
                       unit='pc')

    """
    cart: u.Quantity = convert(obj.q, u.Quantity)
    ct = obj.c * obj.t[..., None]
    return jnp.concat([ct, cart], axis=-1)


@conversion_method(type_from=FourVector, type_to=cxv.CartesianPos3D)  # type: ignore[arg-type]
def convert_4vec_to_cart3d(obj: FourVector, /) -> cxv.CartesianPos3D:
    """Convert a 4-vector to a Cartesian 3-vector.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> print(convert(w, cx.vecs.CartesianPos3D))
    <CartesianPos3D: (x, y, z) [m]
        [1 2 3]>

    """
    return convert(obj.q, cxv.CartesianPos3D)


@conversion_method(type_from=FourVector, type_to=cxv.CylindricalPos)  # type: ignore[arg-type]
def convert_4vec_to_cylindrical(obj: FourVector, /) -> cxv.CylindricalPos:
    """Convert a 4-vector to a Cylindrical 3-vector.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> print(convert(w, cx.vecs.CylindricalPos))
    <CylindricalPos: (rho[m], phi[rad], z[m])
        [2.236 1.107 3.   ]>

    """
    return convert(obj.q, cxv.CylindricalPos)


@conversion_method(type_from=FourVector, type_to=cxv.SphericalPos)  # type: ignore[arg-type]
def convert_4vec_to_spherical(obj: FourVector, /) -> cxv.SphericalPos:
    """Convert a 4-vector to a spherical 3-vector.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> print(convert(w, cx.SphericalPos))
    <SphericalPos: (r[m], theta[rad], phi[rad])
        [3.742 0.641 1.107]>

    """
    return convert(obj.q, cxv.SphericalPos)


@conversion_method(type_from=FourVector, type_to=cxv.LonLatSphericalPos)  # type: ignore[arg-type]
def convert_4vec_to_lonlat_spherical(obj: FourVector, /) -> cxv.LonLatSphericalPos:
    """Convert a 4-vector to a lon-lat spherical 3-vector.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> print(convert(w, cx.vecs.LonLatSphericalPos))
    <LonLatSphericalPos: (lon[rad], lat[deg], distance[m])
        [ 1.107 53.301  3.742]>

    """
    return convert(obj.q, cxv.LonLatSphericalPos)


@conversion_method(type_from=FourVector, type_to=cxv.MathSphericalPos)  # type: ignore[arg-type]
def convert_4vec_to_mathsph(obj: FourVector, /) -> cxv.MathSphericalPos:
    """Convert a 4-vector to a math spherical 3-vector.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> print(convert(w, cx.vecs.MathSphericalPos))
    <MathSphericalPos: (r[m], theta[rad], phi[rad])
        [3.742 1.107 0.641]>

    """
    return convert(obj.q, cxv.MathSphericalPos)
