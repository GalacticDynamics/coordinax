"""Representation of coordinates in different systems."""

__all__ = ["Abstract1DVector", "Abstract1DVectorDifferential"]


from abc import abstractmethod

from vector._base import AbstractVector, AbstractVectorBase, AbstractVectorDifferential
from vector._utils import classproperty


class Abstract1DVector(AbstractVector):
    """Abstract representation of 1D coordinates in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVectorBase]:
        from .builtin import Cartesian1DVector

        return Cartesian1DVector

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["Abstract1DVectorDifferential"]:
        raise NotImplementedError


class Abstract1DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 1D differentials in different systems."""

    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVectorBase]:
        from .builtin import CartesianDifferential1D

        return CartesianDifferential1D

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[Abstract1DVector]:
        raise NotImplementedError
