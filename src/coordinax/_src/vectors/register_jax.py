"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from typing import TypeVar

from jax import Array
from jaxtyping import Shaped
from plum import conversion_method
from zeroth import zeroth

import quaxed.numpy as jnp
from dataclassish import field_values

from .api import cartesian_vector_type
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.utils import full_shaped

T = TypeVar("T")


@conversion_method(type_from=AbstractVector, type_to=Array)  # type: ignore[arg-type,type-abstract]
def vec_diff_to_array(obj: AbstractVector, /) -> Shaped[Array, "*batch N"]:
    """`coordinax.AbstractVector` -> `Array`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> import unxt as u

    >>> pos = cx.vecs.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, Array)
    Array([1.], dtype=float32)

    >>> vel = cx.vecs.CartesianVel1D.from_([1.0], "km/s")
    >>> convert(vel, Array)
    Array([1.], dtype=float32)

    >>> acc = cx.vecs.CartesianAcc1D.from_([1.0], "km/s2")
    >>> convert(acc, Array)
    Array([1.], dtype=float32)

    >>> pos = cx.vecs.RadialPos.from_([1.0], "km")
    >>> convert(pos, Array)
    Array([1.], dtype=float32)

    >>> pos = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> convert(pos, Array)
    Array([1, 2], dtype=int32)

    >>> vel = cx.vecs.CartesianVel2D.from_([1, 2], "km/s")
    >>> convert(vel, Array)
    Array([1, 2], dtype=int32)

    >>> acc = cx.vecs.CartesianAcc2D.from_([1, 2], "km/s2")
    >>> convert(acc, Array)
    Array([1, 2], dtype=int32)

    >>> pos = cx.vecs.PolarPos(u.Quantity(1, "km"), u.Quantity(0, "deg"))
    >>> convert(pos, Array)
    Array([1., 0.], dtype=float32, ...)

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, Array)
    Array([1., 2., 3.], dtype=float32)

    >>> vel = cx.CartesianVel3D.from_([1.0, 2.0, 3.0], "km/s")
    >>> convert(vel, Array)
    Array([1., 2., 3.], dtype=float32)

    >>> acc = cx.vecs.CartesianAcc3D.from_([1.0, 2.0, 3.0], "km/s2")
    >>> convert(acc, Array)
    Array([1., 2., 3.], dtype=float32)

    >>> pos = cx.SphericalPos(u.Quantity(1.0, "km"), u.Quantity(0, "deg"), u.Quantity(0, "deg"))
    >>> convert(pos, Array)
    Array([0., 0., 1.], dtype=float32, ...)

    >>> pos = cx.vecs.CylindricalPos(u.Quantity(1, "km"), u.Quantity(0, "deg"), u.Quantity(0, "km"))
    >>> convert(pos, Array)
    Array([1., 0., 0.], dtype=float32, ...)

    >>> pos = cx.vecs.CartesianPosND.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, Array)
    Array([[1.],
           [2.],
           [3.]], dtype=float32)

    >>> vel = cx.vecs.CartesianVelND.from_([1.0, 2.0, 3.0], "km/s")
    >>> convert(vel, Array)
    Array([[1.],
           [2.],
           [3.]], dtype=float32)

    """  # noqa: E501
    cart = obj.vconvert(cartesian_vector_type(obj))  # convert vector to Cartesian
    cart = full_shaped(cart)  # ensure full shape
    comp_qs = field_values(cart)
    unit = zeroth(comp_qs).unit
    comp_arrs = tuple(x.ustrip(unit) for x in field_values(cart))
    return jnp.stack(comp_arrs, axis=-1)
