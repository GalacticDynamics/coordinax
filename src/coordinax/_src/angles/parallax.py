"""Parallax quantity."""

__all__ = ["Parallax"]

from dataclasses import KW_ONLY
from typing import final

import equinox as eqx
import jax.numpy as jnp

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AbstractQuantity

from .base import AbstractAngle

parallax_base_length = u.Quantity(1, "AU")
angle_dimension = u.dimension("angle")


@final
class Parallax(AbstractAngle):
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

    @property
    def distance(self) -> AbstractQuantity:  # TODO: more specific type
        r"""The distance.

        The distance is calculated as :math:`1 AU / \tan(p)`.

        Examples
        --------
        >>> from coordinax.distance import Parallax
        >>> p = Parallax(1, "mas")
        >>> p.distance.to("kpc")
        Distance(Array(1., dtype=float32, ...), unit='kpc')

        """
        from coordinax.distance import Distance

        return Distance.from_(self)

    @property
    def parallax(self) -> "Parallax":
        """The parallax of the distance.

        Examples
        --------
        >>> from coordinax.distance import Parallax
        >>> p = Parallax(1, "mas")
        >>> p.parallax is p
        True

        """
        return self

    @property
    def distance_modulus(self) -> AbstractQuantity:  # TODO: more specific type
        """The distance modulus.

        Examples
        --------
        >>> from coordinax.distance import Parallax
        >>> Parallax(1, "mas").distance_modulus
        DistanceModulus(Array(10., dtype=float32), unit='mag')

        """
        from coordinax.distance import DistanceModulus

        return DistanceModulus.from_(self)
