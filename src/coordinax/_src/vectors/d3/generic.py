"""Built-in vector classes."""

__all__ = [
    "CartesianGeneric3D",
]

from typing import TypeVar, final

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u

import coordinax._src.typing as ct
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.mixins import AvalMixin

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

    x: ct.BatchableScalarQ = eqx.field(converter=u.Quantity.from_)

    y: ct.BatchableScalarQ = eqx.field(converter=u.Quantity.from_)

    z: ct.BatchableScalarQ = eqx.field(converter=u.Quantity.from_)

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianGeneric3D._dimensionality()
        3

        """
        return 3

    def norm(self) -> u.AbstractQuantity:
        """Compute the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.vecs.CartesianGeneric3D.from_([1, 2, 3], "km")
        >>> print(q.norm())
        Quantity['length'](Array(3.7416575, dtype=float32), unit='km')

        """
        return jnp.sqrt(self.x**2 + self.y**2 + self.z**2)
