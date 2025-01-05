"""Distance quantities."""

__all__ = ["Distance", "DistanceModulus", "Parallax"]

from dataclasses import KW_ONLY
from typing import final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Shaped

import quaxed.numpy as jnp
import unxt as u
from unxt._src.units.api import AbstractUnits

from .base import AbstractDistance

parallax_base_length = u.Quantity(1, "AU")
angle_dimension = u.dimension("angle")
length_dimension = u.dimension("length")


@final
class Distance(AbstractDistance):
    """Distance quantities.

    The distance is a quantity with dimensions of length.

    Examples
    --------
    >>> from coordinax.distance import Distance
    >>> Distance(10, "km")
    Distance(Array(10, dtype=int32, ...), unit='km')

    The units are checked to have length dimensions.

    >>> try: Distance(10, "s")
    ... except ValueError as e: print(e)
    Distance must have dimensions length.

    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=jnp.asarray)
    """The value of the `AbstractQuantity`."""

    unit: AbstractUnits = eqx.field(static=True, converter=u.unit)
    """The unit associated with this value."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if u.dimension_of(self) != length_dimension:
            msg = "Distance must have dimensions length."
            raise ValueError(msg)


@final
class DistanceModulus(AbstractDistance):
    """Distance modulus quantity.

    Examples
    --------
    >>> from coordinax.distance import DistanceModulus
    >>> DistanceModulus(10, "mag")
    DistanceModulus(Array(10, dtype=int32, ...), unit='mag')

    The units are checked to have magnitude dimensions.

    >>> try: DistanceModulus(10, "pc")
    ... except ValueError as e: print(e)
    Distance modulus must have units of magnitude.

    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=jnp.asarray)
    """The value of the `AbstractQuantity`."""

    unit: AbstractUnits = eqx.field(static=True, converter=u.unit)
    """The unit associated with this value."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if self.unit != u.unit("mag"):
            msg = "Distance modulus must have units of magnitude."
            raise ValueError(msg)


@final
class Parallax(AbstractDistance):
    """Parallax distance quantity.

    Examples
    --------
    >>> from coordinax.distance import Parallax
    >>> Parallax(1, "mas")
    Parallax(Array(1, dtype=int32, ...), unit='mas')

    The units are checked to have angle dimensions.

    >>> try: Parallax(1, "pc")
    ... except ValueError as e: print(e)
    Parallax must have angular dimensions.

    The parallax is checked to be non-negative by default.

    >>> try: Parallax(-1, "mas")
    ... except Exception: print("negative")
    negative

    To disable this check, set `check_negative=False`.

    >>> Parallax(-1, "mas", check_negative=False)
    Parallax(Array(-1, dtype=int32, ...), unit='mas')

    """

    value: Shaped[Array, "*shape"] = eqx.field(converter=jnp.asarray)
    """The value of the `AbstractQuantity`."""

    unit: AbstractUnits = eqx.field(static=True, converter=u.unit)
    """The unit associated with this value."""

    _: KW_ONLY
    check_negative: bool = eqx.field(default=True, static=True, compare=False)
    """Whether to check that the parallax is strictly non-negative.

    Theoretically the parallax must be strictly non-negative (:math:`\tan(p) = 1
    AU / d`), however noisy direct measurements of the parallax can be negative.
    """

    def __check_init__(self) -> None:
        """Check the initialization."""
        if u.dimension_of(self) != angle_dimension:
            msg = "Parallax must have angular dimensions."
            raise ValueError(msg)

        if self.check_negative:
            eqx.error_if(
                self.value,
                jnp.any(jnp.less(self.value, 0)),
                "Parallax must be non-negative.",
            )
