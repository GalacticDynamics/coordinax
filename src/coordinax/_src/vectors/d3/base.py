"""Representation of coordinates in different systems."""

__all__ = ["AbstractAcc3D", "AbstractPos3D", "AbstractVel3D"]


from abc import abstractmethod
from typing_extensions import override

from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import (
    AbstractAcc,
    AbstractPos,
    AbstractVector,
    AbstractVel,
)


class AbstractPos3D(AbstractPos):
    """Abstract representation of 3D coordinates in different systems."""

    @override
    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianPos3D

        return CartesianPos3D

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractVel3D"]:
        raise NotImplementedError


class AbstractVel3D(AbstractVel):
    """Abstract representation of 3D vector differentials."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianVel3D

        return CartesianVel3D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractPos3D]:
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type[AbstractAcc]:
        raise NotImplementedError


class AbstractAcc3D(AbstractAcc):
    """Abstract representation of 3D vector accelerations."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianAcc3D

        return CartesianAcc3D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVel3D]:
        raise NotImplementedError
