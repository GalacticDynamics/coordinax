"""Base classes for distance quantities."""

__all__: list[str] = []


from plum import add_promotion_rule

from unxt.quantity import AbstractQuantity, Quantity


class AbstractDistance(AbstractQuantity):  # type: ignore[misc]
    """Distance quantities."""

    @property
    def distance(self) -> "AbstractDistance":  # TODO: more specific type
        """The distance.

        Examples
        --------
        >>> from coordinax.distance import Distance
        >>> d = Distance(10, "km")
        >>> d.distance is d
        True

        >>> from coordinax.distance import DistanceModulus
        >>> DistanceModulus(10, "mag").distance
        Distance(Array(1000., dtype=float32, ...), unit='pc')

        >>> from coordinax.distance import Parallax
        >>> p = Parallax(1, "mas")
        >>> p.distance.to("kpc")
        Distance(Array(1., dtype=float32, ...), unit='kpc')

        """
        from coordinax.distance import Distance

        return Distance.from_(self)

    @property
    def distance_modulus(self) -> "AbstractDistance":
        """The distance modulus.

        Examples
        --------
        >>> from coordinax.distance import Distance
        >>> d = Distance(1, "pc")
        >>> d.distance_modulus
        DistanceModulus(Array(-5., dtype=float32), unit='mag')

        >>> from coordinax.distance import DistanceModulus
        >>> DistanceModulus(10, "mag").distance_modulus
        DistanceModulus(Array(10, dtype=int32, ...), unit='mag')

        >>> from coordinax.distance import Parallax
        >>> Parallax(1, "mas").distance_modulus
        DistanceModulus(Array(10., dtype=float32), unit='mag')

        """
        from coordinax.distance import DistanceModulus

        return DistanceModulus.from_(self)

    @property
    def parallax(self) -> "AbstractDistance":  # TODO: more specific type
        r"""The parallax from a distance.

        The parallax is calculated as :math:`\arctan(1 AU / d)`.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> from coordinax.distance import Distance

        >>> d = Distance(1, "pc")
        >>> jnp.round(d.parallax.to("arcsec"), 2)
        Parallax(Array(1., dtype=float32, ...), unit='arcsec')

        >>> from coordinax.distance import DistanceModulus
        >>> DistanceModulus(10, "mag").parallax.to("mas")
        Parallax(Array(0.99999994, dtype=float32, ...), unit='mas')

        >>> from coordinax.distance import Parallax
        >>> p = Parallax(1, "mas")
        >>> p.parallax is p
        True

        """
        from coordinax.angle import Parallax

        return Parallax.from_(self)


# Add a rule that when a AbstractDistance interacts with a Quantity, the
# distance degrades to a Quantity. This is necessary for many operations, e.g.
# division of a distance by non-dimensionless quantity where the resulting units
# are not those of a distance.
add_promotion_rule(AbstractDistance, Quantity, Quantity)
