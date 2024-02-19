"""Representation of coordinates in different systems."""

__all__ = ["Abstract1DVector", "Abstract1DVectorDifferential"]


import equinox as eqx

from vector._base import AbstractVector, AbstractVectorDifferential


class Abstract1DVector(AbstractVector):
    """Abstract representation of 1D coordinates in different systems."""


class Abstract1DVectorDifferential(AbstractVectorDifferential):
    """Abstract representation of 1D differentials in different systems."""

    vector_cls: eqx.AbstractClassVar[type[Abstract1DVector]]
