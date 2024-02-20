"""Representation of coordinates in different systems."""

__all__ = ["AbstractVector", "AbstractVectorDifferential"]

from typing import Any, TypeVar

import equinox as eqx

T = TypeVar("T", bound="AbstractVector")
DT = TypeVar("DT", bound="AbstractVectorDifferential")


class AbstractVector(eqx.Module):  # type: ignore[misc]
    """Abstract representation of coordinates in different systems."""

    def represent_as(self, target: type[T], /, **kwargs: Any) -> T:
        """Represent the vector as another type."""
        from ._transform import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, **kwargs)


class AbstractVectorDifferential(eqx.Module):  # type: ignore[misc]
    """Abstract representation of vector differentials in different systems."""

    vector_cls: eqx.AbstractClassVar[type[AbstractVector]]

    def represent_as(
        self, target: type[DT], position: AbstractVector, /, **kwargs: Any
    ) -> DT:
        """Represent the vector as another type."""
        from ._transform import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, position, **kwargs)
