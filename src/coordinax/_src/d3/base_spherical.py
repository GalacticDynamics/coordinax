"""Built-in vector classes."""

__all__ = [
    "AbstractSphericalPosition",
    "AbstractSphericalVelocity",
    "AbstractSphericalAcceleration",
]

from abc import abstractmethod
from typing_extensions import override

from unxt import Quantity

from .base import AbstractAcceleration3D, AbstractPosition3D, AbstractVelocity3D
from coordinax._src.utils import classproperty

_90d = Quantity(90, "deg")
_180d = Quantity(180, "deg")
_360d = Quantity(360, "deg")


class AbstractSphericalPosition(AbstractPosition3D):
    """Abstract spherical vector representation."""

    @override
    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> "type[AbstractSphericalVelocity]": ...


class AbstractSphericalVelocity(AbstractVelocity3D):
    """Spherical differential representation."""

    @override
    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractSphericalPosition]: ...

    @override
    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> "type[AbstractSphericalAcceleration]": ...


class AbstractSphericalAcceleration(AbstractAcceleration3D):
    """Spherical acceleration representation."""

    @override
    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractSphericalVelocity]: ...
