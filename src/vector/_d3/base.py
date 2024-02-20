"""Representation of coordinates in different systems."""

__all__ = ["Abstract3DVector", "Abstract3DVectorDifferential"]


import equinox as eqx

from vector._base import AbstractVector, AbstractVectorDifferential


class Abstract3DVector(AbstractVector):
    """Abstract representation of 3D coordinates in different systems."""


class Abstract3DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 3D vector differentials."""

    vector_cls: eqx.AbstractClassVar[type[Abstract3DVector]]
