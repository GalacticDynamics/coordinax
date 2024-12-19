"""Built-in vector classes."""

__all__ = [
    "CartesianGeneric3D",
]

from dataclasses import fields
from typing import TypeVar, final

import equinox as eqx
import jax

import quaxed.numpy as jnp
from unxt.quantity import AbstractQuantity, Quantity

import coordinax._src.typing as ct
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base.mixins import AvalMixin

VT = TypeVar("VT", bound="CartesianGeneric3D")


@final
class CartesianGeneric3D(AvalMixin, AbstractVector):
    """Generic 3D Cartesian coordinates.

    The fields of this class are not restricted to any specific dimensions.

    Examples
    --------
    >>> import coordinax as cx
    >>> vec = cx.vecs.CartesianGeneric3D.from_([1, 2, 3], "kg m /s")
    >>> print(vec)
    <CartesianGeneric3D (x[kg m / s], y[kg m / s], z[kg m / s])
        [1 2 3]>

    """

    x: ct.BatchableScalarQ = eqx.field(converter=Quantity.from_)

    y: ct.BatchableScalarQ = eqx.field(converter=Quantity.from_)

    z: ct.BatchableScalarQ = eqx.field(converter=Quantity.from_)

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "CartesianGeneric3D":
        """Negate the `coordinax.CartesianGeneric3D`.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.vecs.CartesianGeneric3D.from_([1, 2, 3], "km")
        >>> print(-q)
        <CartesianGeneric3D (x[km], y[km], z[km])
        [-1 -2 -3]>

        """
        return jax.tree.map(jnp.negative, self)


# =====================================================
# Constructors


@CartesianGeneric3D.from_.dispatch  # type: ignore[attr-defined, misc]
def from_(
    cls: type[CartesianGeneric3D],
    obj: AbstractQuantity,  # TODO: Shaped[AbstractQuantity, "*batch 3"]
    /,
) -> CartesianGeneric3D:
    """Construct a 3D Cartesian position.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianGeneric3D.from_(u.Quantity([1, 2, 3], "m"))
    >>> print(vec)
    <CartesianGeneric3D (x[m], y[m], z[m])
        [1 2 3]>

    """
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)
