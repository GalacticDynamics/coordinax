"""Base classes for distance quantities."""

__all__: list[str] = []

from abc import abstractmethod

from unxt.quantity import AbstractQuantity


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
