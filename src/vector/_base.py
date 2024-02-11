"""Representation of coordinates in different systems."""

__all__ = ["AbstractVector", "Abstract1DVector", "Abstract2DVector", "Abstract3DVector"]

from typing import Any, TypeVar

import equinox as eqx

T = TypeVar("T", bound="AbstractVector")


class AbstractVector(eqx.Module):  # type: ignore[misc]
    """Abstract representation of coordinates in different systems."""

    def represent_as(self, target: type[T], **kwargs: Any) -> T:
        """Represent the vector as another type."""
        from ._transform import represent_as

        return represent_as(self, target, **kwargs)


class Abstract1DVector(AbstractVector):
    """Abstract representation of 1D coordinates in different systems."""


class Abstract2DVector(AbstractVector):
    """Abstract representation of 2D coordinates in different systems."""


class Abstract3DVector(AbstractVector):
    """Abstract representation of 3D coordinates in different systems."""
