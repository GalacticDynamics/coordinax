"""Representation of coordinates in different systems."""

__all__ = ["Abstract2DVector", "Abstract2DVectorDifferential"]


import equinox as eqx

from vector._base import AbstractVector, AbstractVectorBase, AbstractVectorDifferential


class Abstract2DVector(AbstractVector):
    """Abstract representation of 2D coordinates in different systems."""

    @property
    def _cartesian_cls(self) -> type[AbstractVectorBase]:
        from .builtin import Cartesian2DVector

        return Cartesian2DVector


class Abstract2DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 2D vector differentials."""

    vector_cls: eqx.AbstractClassVar[type[Abstract2DVector]]

    @property
    def _cartesian_cls(self) -> type[AbstractVectorBase]:
        from .builtin import CartesianDifferential2D

        return CartesianDifferential2D
