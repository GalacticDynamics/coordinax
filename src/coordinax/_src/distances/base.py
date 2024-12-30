"""Base classes for distance quantities."""

__all__: list[str] = []

from abc import abstractmethod

from plum import add_promotion_rule

from unxt.quantity import AbstractQuantity, Quantity


class AbstractDistance(AbstractQuantity):  # type: ignore[misc]
    """Distance quantities."""

    @property
    @abstractmethod
    def distance(self) -> "Distance":
        """The distance."""

    @property
    @abstractmethod
    def parallax(self) -> "Parallax":
        """The parallax."""

    @property
    @abstractmethod
    def distance_modulus(self) -> "DistanceModulus":
        """The distance modulus."""


# Add a rule that when a AbstractDistance interacts with a Quantity, the
# distance degrades to a Quantity. This is necessary for many operations, e.g.
# division of a distance by non-dimensionless quantity where the resulting units
# are not those of a distance.
add_promotion_rule(AbstractDistance, Quantity, Quantity)
