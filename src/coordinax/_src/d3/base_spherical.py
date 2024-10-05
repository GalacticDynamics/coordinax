"""Built-in vector classes."""

__all__ = [
    "AbstractSphericalPos",
    "AbstractSphericalVel",
    "AbstractSphericalAcc",
]

from abc import abstractmethod
from typing_extensions import override

from unxt import Quantity

from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from coordinax._src.utils import classproperty

_90d = Quantity(90, "deg")
_180d = Quantity(180, "deg")
_360d = Quantity(360, "deg")


class AbstractSphericalPos(AbstractPos3D):
    """Abstract spherical vector representation."""

    @override
    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> "type[AbstractSphericalVel]": ...


class AbstractSphericalVel(AbstractVel3D):
    """Spherical differential representation."""

    @override
    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractSphericalPos]: ...

    @override
    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> "type[AbstractSphericalAcc]": ...


class AbstractSphericalAcc(AbstractAcc3D):
    """Spherical acceleration representation."""

    @override
    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractSphericalVel]: ...
