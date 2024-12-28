"""Representation of coordinates in different systems."""

__all__ = ["AbstractAcc2D", "AbstractPos2D", "AbstractVel2D"]


from abc import abstractmethod

from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import (
    AbstractAcc,
    AbstractPos,
    AbstractVector,
    AbstractVel,
)


class AbstractPos2D(AbstractPos):
    """Abstract representation of 2D coordinates in different systems."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianPos2D._dimensionality()
        2

        """
        return 2

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianPos2D

        return CartesianPos2D

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractVel2D"]:
        raise NotImplementedError


#####################################################################


class AbstractVel2D(AbstractVel):
    """Abstract representation of 2D vector differentials."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianVel2D._dimensionality()
        2

        """
        return 2

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianVel2D

        return CartesianVel2D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractPos2D]:
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type[AbstractAcc]:
        raise NotImplementedError


#####################################################################


class AbstractAcc2D(AbstractAcc):
    """Abstract representation of 2D vector accelerations."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianAcc2D._dimensionality()
        2

        """
        return 2

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianAcc2D

        return CartesianAcc2D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVel2D]:
        raise NotImplementedError
