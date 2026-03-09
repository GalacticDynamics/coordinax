"""Base classes for distance quantities."""

__all__: tuple[str, ...] = ()


from plum import add_promotion_rule

import unxt as u


class AbstractDistance(u.AbstractQuantity):  # type: ignore[misc]
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
        Distance(1000., 'pc')

        >>> from coordinax.distance import Parallax
        >>> p = Parallax(1, "mas")
        >>> p.distance.to("kpc")
        Distance(1., 'kpc')

        """
        from coordinax.distance import Distance  # noqa: PLC0415

        return Distance.from_(self)

    @property
    def distance_modulus(self) -> "AbstractDistance":
        """The distance modulus.

        Examples
        --------
        >>> from coordinax.distance import Distance
        >>> d = Distance(1, "pc")
        >>> d.distance_modulus
        DistanceModulus(-5., 'mag')

        >>> from coordinax.distance import DistanceModulus
        >>> DistanceModulus(10, "mag").distance_modulus
        DistanceModulus(10, 'mag')

        >>> from coordinax.distance import Parallax
        >>> Parallax(1, "mas").distance_modulus
        DistanceModulus(10., 'mag')

        """
        from coordinax.distance import DistanceModulus  # noqa: PLC0415

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
        Parallax(1., 'arcsec')

        >>> from coordinax.distance import DistanceModulus
        >>> DistanceModulus(10, "mag").parallax.to("mas")
        Parallax(0.99999994, 'mas')

        >>> from coordinax.distance import Parallax
        >>> p = Parallax(1, "mas")
        >>> p.parallax is p
        True

        """
        from coordinax.angle import Parallax  # noqa: PLC0415

        return Parallax.from_(self)


# Add a rule that when a AbstractDistance interacts with a Quantity, the
# distance degrades to a Quantity. This is necessary for many operations, e.g.
# division of a distance by non-dimensionless quantity where the resulting units
# are not those of a distance.
add_promotion_rule(AbstractDistance, u.Quantity, u.Quantity)
add_promotion_rule(AbstractDistance, u.quantity.BareQuantity, u.quantity.BareQuantity)
