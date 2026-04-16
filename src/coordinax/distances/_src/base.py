"""Base classes for distance quantities."""

__all__: tuple[str, ...] = ("AbstractDistance",)


from typing import cast

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

        >>> import coordinax.astro as cxastro
        >>> cxastro.DistanceModulus(10, "mag").distance
        Distance(1000., 'pc')

        >>> p = cxastro.Parallax(1, "mas")
        >>> p.distance.to("kpc")
        Distance(1., 'kpc')

        """
        from coordinax.distances import Distance  # noqa: PLC0415

        return cast("Distance", Distance.from_(self))


# Add a rule that when a AbstractDistance interacts with a Quantity, the
# distance degrades to a Quantity. This is necessary for many operations, e.g.
# division of a distance by non-dimensionless quantity where the resulting units
# are not those of a distance.
add_promotion_rule(AbstractDistance, u.Q, u.Q)
add_promotion_rule(AbstractDistance, u.quantity.BareQuantity, u.quantity.BareQuantity)


# Add a rule that when a AbstractDistance interacts with a Quantity, the
# distance degrades to a Quantity. This is necessary for many operations, e.g.
# division of a distance by non-dimensionless quantity where the resulting units
# are not those of a distance.
add_promotion_rule(AbstractDistance, u.quantity.AbstractAngle, u.Q)
