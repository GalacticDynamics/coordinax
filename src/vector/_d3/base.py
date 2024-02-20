"""Representation of coordinates in different systems."""

__all__ = ["Abstract3DVector"]

from typing import TypeVar

from vector._base import AbstractVector

T = TypeVar("T", bound="AbstractVector")


class Abstract3DVector(AbstractVector):
    """Abstract representation of 3D coordinates in different systems."""
