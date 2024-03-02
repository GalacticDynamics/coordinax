"""Representation of coordinates in different systems."""

__all__ = ["Abstract3DVector", "Abstract3DVectorDifferential"]


from abc import abstractmethod

from coordinax._base import (
    AbstractVector,
    AbstractVectorBase,
    AbstractVectorDifferential,
)
from coordinax._utils import classproperty


class Abstract3DVector(AbstractVector):
    """Abstract representation of 3D coordinates in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVectorBase]:
        from .builtin import Cartesian3DVector

        return Cartesian3DVector

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["Abstract3DVectorDifferential"]:
        raise NotImplementedError


class Abstract3DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 3D vector differentials."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVectorBase]:
        from .builtin import CartesianDifferential3D

        return CartesianDifferential3D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[Abstract3DVector]:
        raise NotImplementedError
