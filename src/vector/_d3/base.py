"""Representation of coordinates in different systems."""

__all__ = ["Abstract3DVector", "Abstract3DVectorDifferential"]


import equinox as eqx

from vector._base import AbstractVector, AbstractVectorBase, AbstractVectorDifferential
from vector._utils import classproperty


class Abstract3DVector(AbstractVector):
    """Abstract representation of 3D coordinates in different systems."""

    @classproperty
    def _cartesian_cls(self: type[AbstractVectorBase]) -> type[AbstractVectorBase]:
        from .builtin import Cartesian3DVector

        return Cartesian3DVector


class Abstract3DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 3D vector differentials."""

    vector_cls: eqx.AbstractClassVar[type[Abstract3DVector]]

    @classproperty
    def _cartesian_cls(self: type[AbstractVectorBase]) -> type[AbstractVectorBase]:
        from .builtin import CartesianDifferential3D

        return CartesianDifferential3D
