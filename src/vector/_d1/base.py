"""Representation of coordinates in different systems."""

__all__ = ["Abstract1DVector", "Abstract1DVectorDifferential"]


import equinox as eqx

from vector._base import AbstractVector, AbstractVectorBase, AbstractVectorDifferential
from vector._utils import classproperty


class Abstract1DVector(AbstractVector):
    """Abstract representation of 1D coordinates in different systems."""

    @classproperty
    def _cartesian_cls(self: type[AbstractVectorBase]) -> type[AbstractVectorBase]:
        from .builtin import Cartesian1DVector

        return Cartesian1DVector


class Abstract1DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 1D differentials in different systems."""

    vector_cls: eqx.AbstractClassVar[type[Abstract1DVector]]

    @classproperty
    def _cartesian_cls(self: type[AbstractVectorBase]) -> type[AbstractVectorBase]:
        from .builtin import CartesianDifferential1D

        return CartesianDifferential1D
