"""Representation of coordinates in different systems."""

__all__ = ["Abstract2DVector", "Abstract2DVectorDifferential"]


from abc import abstractmethod

from coordinax._base import (
    AbstractVector,
    AbstractVectorBase,
    AbstractVectorDifferential,
)
from coordinax._utils import classproperty


class Abstract2DVector(AbstractVector):
    """Abstract representation of 2D coordinates in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVectorBase]:
        from .builtin import Cartesian2DVector

        return Cartesian2DVector

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["Abstract2DVectorDifferential"]:
        raise NotImplementedError


class Abstract2DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 2D vector differentials."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVectorBase]:
        from .builtin import CartesianDifferential2D

        return CartesianDifferential2D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[Abstract2DVector]:
        raise NotImplementedError
