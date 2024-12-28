"""Representation of coordinates in different systems."""

__all__ = ["AbstractPos4D"]


from abc import abstractmethod
from typing import NoReturn

from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractPos, AbstractVector


class AbstractPos4D(AbstractPos):
    """Abstract representation of 4D coordinates in different systems."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.AbstractPos4D._dimensionality()
        4

        """
        return 4

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> NoReturn:  # type: ignore[override]
        msg = "Not yet implemented"
        raise NotImplementedError(msg)
