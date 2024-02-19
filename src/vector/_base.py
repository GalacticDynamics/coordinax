"""Representation of coordinates in different systems."""

__all__ = ["AbstractVector"]

from typing import Any, TypeVar

import equinox as eqx

T = TypeVar("T", bound="AbstractVector")


class AbstractVector(eqx.Module):  # type: ignore[misc]
    """Abstract representation of coordinates in different systems."""

    def represent_as(self, target: type[T], **kwargs: Any) -> T:
        """Represent the vector as another type."""
        from ._transform import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, **kwargs)
