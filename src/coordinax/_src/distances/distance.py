"""Distance quantities."""

__all__ = ["Distance", "DistanceModulus"]

from typing import final

import unxt as u
from unxt.quantity import AbstractQuantity

from .base import AbstractDistance

angle_dimension = u.dimension("angle")
length_dimension = u.dimension("length")


##############################################################################


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

    def __check_init__(self) -> None:
        """Check the initialization."""
        if u.dimension_of(self) != length_dimension:
            msg = "Distance must have dimensions length."
            raise ValueError(msg)

    @property
    def distance(self) -> "Distance":
        """The distance.

        Examples
        --------
        >>> from coordinax.distance import Distance
        >>> d = Distance(10, "km")
        >>> d.distance is d
        True

        """
        return self

    @property
    def parallax(self) -> AbstractQuantity:  # TODO: more specific type
        r"""The parallax of the distance.

        The parallax is calculated as :math:`\arctan(1 AU / d)`.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> from coordinax.distance import Distance
        >>> d = Distance(1, "pc")
        >>> jnp.round(d.parallax.to("arcsec"), 2)
        Parallax(Array(1., dtype=float32, ...), unit='arcsec')

        """
        from coordinax.angle import Parallax

        return Parallax.from_(self)

    @property
    def distance_modulus(self) -> "DistanceModulus":
        """The distance modulus.

        Examples
        --------
        >>> from coordinax.distance import Distance
        >>> d = Distance(1, "pc")
        >>> d.distance_modulus
        DistanceModulus(Array(-5., dtype=float32), unit='mag')

        """
        return DistanceModulus.from_(self)


##############################################################################


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

    def __check_init__(self) -> None:
        """Check the initialization."""
        if self.unit != u.unit("mag"):
            msg = "Distance modulus must have units of magnitude."
            raise ValueError(msg)

    @property
    def distance(self) -> Distance:
        """The distance.

        The distance is calculated as :math:`10^{(m / 5 + 1)}`.

        Examples
        --------
        >>> from coordinax.distance import DistanceModulus
        >>> DistanceModulus(10, "mag").distance
        Distance(Array(1000., dtype=float32, ...), unit='pc')

        """
        return Distance.from_(self)

    @property
    def parallax(self) -> AbstractQuantity:  # TODO: more specific type
        """The parallax.

        Examples
        --------
        >>> from coordinax.distance import DistanceModulus
        >>> DistanceModulus(10, "mag").parallax.to("mas")
        Parallax(Array(0.99999994, dtype=float32, ...), unit='mas')

        """
        from coordinax.angle import Parallax

        return Parallax.from_(self)

    @property
    def distance_modulus(self) -> "DistanceModulus":
        """The distance modulus.

        Examples
        --------
        >>> from coordinax.distance import DistanceModulus
        >>> DistanceModulus(10, "mag").distance_modulus
        DistanceModulus(Array(10, dtype=int32, ...), unit='mag')

        """
        return self
