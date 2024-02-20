"""Representation of coordinates in different systems."""

__all__ = ["Abstract2DVector"]

from typing import TypeVar

from vector._base import AbstractVector

T = TypeVar("T", bound="AbstractVector")


class Abstract2DVector(AbstractVector):
    """Abstract representation of 2D coordinates in different systems."""
