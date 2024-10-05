"""Representation of coordinates in different systems."""

__all__ = ["AbstractPos3D", "AbstractVelocity3D", "AbstractAcceleration3D"]


from abc import abstractmethod
from typing_extensions import override

from coordinax._src.base import (
    AbstractAcceleration,
    AbstractPos,
    AbstractVector,
    AbstractVelocity,
)
from coordinax._src.utils import classproperty


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
    def differential_cls(cls) -> type["AbstractVelocity3D"]:
        raise NotImplementedError


class AbstractVelocity3D(AbstractVelocity):
    """Abstract representation of 3D vector differentials."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianVelocity3D

        return CartesianVelocity3D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractPos3D]:
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type[AbstractAcceleration]:
        raise NotImplementedError


class AbstractAcceleration3D(AbstractAcceleration):
    """Abstract representation of 3D vector accelerations."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianAcceleration3D

        return CartesianAcceleration3D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVelocity3D]:
        raise NotImplementedError
