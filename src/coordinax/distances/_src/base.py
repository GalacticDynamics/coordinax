"""Base classes for distance quantities."""

__all__: tuple[str, ...] = ()


from plum import add_promotion_rule

import unxt as u


class AbstractDistance(u.AbstractQuantity):
    """Distance quantities."""

    @property
    def distance(self) -> "AbstractDistance":  # TODO: more specific type
        """The distance.

        Examples
        --------
        >>> import coordinax.distances as cxd
        >>> d = cxd.Distance(10, "km")
        >>> d.distance is d
        True

        >>> cxd.DistanceModulus(10, "mag").distance
        Distance(Array(1000., dtype=float64, ...), unit='pc')

        >>> p = cxd.Parallax(1, "mas")
        >>> p.distance.to("kpc")
        Distance(Array(1., dtype=float64, ...), unit='kpc')

        """
        from coordinax.distances import Distance  # noqa: PLC0415

        return Distance.from_(self)

    @property
    def distance_modulus(self) -> "AbstractDistance":
        """The distance modulus.

        Examples
        --------
        >>> from coordinax.distances import Distance
        >>> d = Distance(1, "pc")
        >>> d.distance_modulus
        DistanceModulus(Array(-5., dtype=float64), unit='mag')

        >>> from coordinax.distances import DistanceModulus
        >>> DistanceModulus(10, "mag").distance_modulus
        DistanceModulus(Array(10, dtype=int64, ...), unit='mag')

        >>> from coordinax.distances import Parallax
        >>> Parallax(1, "mas").distance_modulus
        DistanceModulus(Array(10., dtype=float64), unit='mag')

        """
        from coordinax.distances import DistanceModulus  # noqa: PLC0415

        return DistanceModulus.from_(self)

    @property
    def parallax(self) -> "AbstractDistance":  # TODO: more specific type
        r"""The parallax from a distance.

        The parallax is calculated as $\arctan(1 AU / d)$.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> from coordinax.distances import Distance

        >>> d = Distance(1, "pc")
        >>> jnp.round(d.parallax.to("arcsec"), 2)
        Parallax(Array(1., dtype=float64, ...), unit='arcsec')

        >>> from coordinax.distances import DistanceModulus
        >>> DistanceModulus(10, "mag").parallax.to("mas")
        Parallax(Array(1., dtype=float64, ...), unit='mas')

        >>> from coordinax.distances import Parallax
        >>> p = Parallax(1, "mas")
        >>> p.parallax is p
        True

        """
        from coordinax.angles import Parallax  # noqa: PLC0415

        return Parallax.from_(self)


# Add a rule that when a AbstractDistance interacts with a Quantity, the
# distance degrades to a Quantity. This is necessary for many operations, e.g.
# division of a distance by non-dimensionless quantity where the resulting units
# are not those of a distance.
add_promotion_rule(AbstractDistance, u.Q, u.Q)
add_promotion_rule(AbstractDistance, u.quantity.BareQuantity, u.quantity.BareQuantity)
