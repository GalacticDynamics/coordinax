"""Representation of coordinates in different systems."""

__all__ = ["AbstractAcc1D", "AbstractPos1D", "AbstractVel1D"]


from abc import abstractmethod
from typing_extensions import override

from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_acc import AbstractAcc
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel

#####################################################################


class AbstractPos1D(AbstractPos):
    """Abstract representation of 1D coordinates in different systems."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianPos1D._dimensionality()
        1

        """
        return 1

    @override
    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:  # type: ignore[override]
        from .cartesian import CartesianPos1D

        return CartesianPos1D

    @override
    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractVel1D"]:  # type: ignore[override]
        raise NotImplementedError


#####################################################################


class AbstractVel1D(AbstractVel):
    """Abstract representation of 1D differentials in different systems."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianVel1D._dimensionality()
        1

        """
        return 1

    @override
    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:  # type: ignore[override]
        from .cartesian import CartesianVel1D

        return CartesianVel1D

    @override
    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractPos1D]:  # type: ignore[override]
        raise NotImplementedError

    @override
    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type[AbstractAcc]:  # type: ignore[override]
        raise NotImplementedError


#####################################################################


class AbstractAcc1D(AbstractAcc):
    """Abstract representation of 1D acceleration in different systems."""

    @classmethod
    def _dimensionality(cls) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianAcc1D._dimensionality()
        1

        """
        return 1

    @override
    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:  # type: ignore[override]
        from .cartesian import CartesianAcc1D

        return CartesianAcc1D

    @override
    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVel1D]:  # type: ignore[override]
        raise NotImplementedError
