"""Representation of coordinates in different systems."""

__all__ = ["AbstractPosition2D", "AbstractVelocity2D", "AbstractAcceleration2D"]


from abc import abstractmethod

from coordinax._src.base import (
    AbstractAcceleration,
    AbstractPosition,
    AbstractVector,
    AbstractVelocity,
)
from coordinax._src.utils import classproperty


class AbstractPosition2D(AbstractPosition):
    """Abstract representation of 2D coordinates in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianPosition2D

        return CartesianPosition2D

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractVelocity2D"]:
        raise NotImplementedError


#####################################################################


class AbstractVelocity2D(AbstractVelocity):
    """Abstract representation of 2D vector differentials."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianVelocity2D

        return CartesianVelocity2D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractPosition2D]:
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type[AbstractAcceleration]:
        raise NotImplementedError


#####################################################################


class AbstractAcceleration2D(AbstractAcceleration):
    """Abstract representation of 2D vector accelerations."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        from .cartesian import CartesianAcceleration2D

        return CartesianAcceleration2D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVelocity2D]:
        raise NotImplementedError
