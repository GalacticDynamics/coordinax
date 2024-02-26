"""Representation of coordinates in different systems."""

__all__ = ["Abstract1DVector", "Abstract1DVectorDifferential"]


import equinox as eqx

from vector._base import AbstractVector, AbstractVectorBase, AbstractVectorDifferential


class Abstract1DVector(AbstractVector):
    """Abstract representation of 1D coordinates in different systems."""

    @property
    def _cartesian_cls(self) -> type[AbstractVectorBase]:
        from .builtin import Cartesian1DVector

        return Cartesian1DVector


class Abstract1DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 1D differentials in different systems."""

    vector_cls: eqx.AbstractClassVar[type[Abstract1DVector]]

    @property
    def _cartesian_cls(self) -> type[AbstractVectorBase]:
        from .builtin import CartesianDifferential1D

        return CartesianDifferential1D
