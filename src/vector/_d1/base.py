"""Representation of coordinates in different systems."""

__all__ = ["Abstract1DVector"]

from typing import TypeVar

from vector._base import AbstractVector

T = TypeVar("T", bound="AbstractVector")


class Abstract1DVector(AbstractVector):
    """Abstract representation of 1D coordinates in different systems."""
