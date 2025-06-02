"""Built-in vector classes."""

__all__ = [
    "Cartesian3D",
]

from typing import TypeVar, final
from typing_extensions import override

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u

import coordinax._src.custom_types as ct
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.mixins import AvalMixin

VT = TypeVar("VT", bound="Cartesian3D")


@final
class Cartesian3D(AvalMixin, AbstractVector):
    """Generic 3D Cartesian coordinates.

    The fields of this class are not restricted to any specific dimensions.
    For specific dimensions, use the specialized classes:

    - `coordinax.vecs.CartesianPos3D`
    - `coordinax.vecs.CartesianVel3D`
    - `coordinax.vecs.CartesianAcc3D`

    Examples
    --------
    >>> import coordinax as cx
    >>> vec = cx.vecs.Cartesian3D.from_([1, 2, 3], "kg m /s")
    >>> print(vec)
    <Cartesian3D: (x, y, z) [kg m / s]
        [1 2 3]>

    """

    x: ct.BBtScalarQ = eqx.field(converter=u.Quantity.from_)

    y: ct.BBtScalarQ = eqx.field(converter=u.Quantity.from_)

    z: ct.BBtScalarQ = eqx.field(converter=u.Quantity.from_)

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.Cartesian3D._dimensionality()
        3

        """
        return 3

    def norm(self) -> u.AbstractQuantity:
        """Compute the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.vecs.Cartesian3D.from_([1, 2, 3], "km")
        >>> print(q.norm())
        Quantity['length'](3.7416575, unit='km')

        """
        return jnp.sqrt(self.x**2 + self.y**2 + self.z**2)

    @override
    @property
    def dimensions(self) -> dict[str, u.dims.AbstractDimension]:  # type: ignore[override]
        """Vector physical dimensions.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.Cartesian3D.dimensions
        <property object at ...>

        >>> q = cx.vecs.Cartesian3D.from_([1, 2, 3], "km")
        >>> q.dimensions
        {'x': PhysicalType('length'), 'y': PhysicalType('length'),
         'z': PhysicalType('length')}

        """
        return {k: u.dimension_of(getattr(self, k)) for k in self.components}
