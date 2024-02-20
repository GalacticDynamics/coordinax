"""Representation of coordinates in different systems."""

__all__ = ["Abstract2DVector", "Abstract2DVectorDifferential"]


import equinox as eqx

from vector._base import AbstractVector, AbstractVectorDifferential


class Abstract2DVector(AbstractVector):
    """Abstract representation of 2D coordinates in different systems."""


class Abstract2DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 2D vector differentials."""

    vector_cls: eqx.AbstractClassVar[type[Abstract2DVector]]
