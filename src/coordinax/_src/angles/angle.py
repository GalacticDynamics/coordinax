"""Angular quantities."""

__all__ = ["Angle"]

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Shaped

import unxt as u
from unxt._src.units.api import AbstractUnits

from .base import AbstractAngle


@final
class Angle(AbstractAngle):
    """Angular quantity.

    Examples
    --------
    >>> from coordinax.angle import Angle

    Create an Angle:

    >>> q = Angle(1, "rad")
    >>> q
    Angle(Array(1, dtype=int32, ...), unit='rad')

    Create an Angle array:

    >>> q = Angle([1, 2, 3], "deg")
    >>> q
    Angle(Array([1, 2, 3], dtype=int32), unit='deg')

    Do math on an Angle:

    >>> 2 * q
    Angle(Array([2, 4, 6], dtype=int32), unit='deg')

    >>> import unxt as u
    >>> q % u.Quantity(4, "deg")
    Angle(Array([1, 2, 3], dtype=int32), unit='deg')

    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=jnp.asarray)
    """The value of the `AbstractQuantity`."""

    unit: AbstractUnits = eqx.field(static=True, converter=u.unit)
    """The unit associated with this value."""
