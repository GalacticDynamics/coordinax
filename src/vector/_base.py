"""Representation of coordinates in different systems."""

__all__ = ["AbstractVectorBase", "AbstractVector", "AbstractVectorDifferential"]

import warnings
from abc import abstractmethod
from functools import partial
from typing import Any, TypeVar

import equinox as eqx
import jax

BT = TypeVar("BT", bound="AbstractVectorBase")
VT = TypeVar("VT", bound="AbstractVector")
DT = TypeVar("DT", bound="AbstractVectorDifferential")


class AbstractVectorBase(eqx.Module):  # type: ignore[misc]
    """Base class for all vector types."""

    # ===============================================================
    # Convenience methods

    @abstractmethod
    def represent_as(self, target: type[BT], /, *args: Any, **kwargs: Any) -> BT:
        """Represent the vector as another type."""
        raise NotImplementedError


class AbstractVector(AbstractVectorBase):
    """Abstract representation of coordinates in different systems."""

    # ===============================================================
    # Convenience methods

    @partial(jax.jit, static_argnums=1)
    def represent_as(self, target: type[VT], /, *args: Any, **kwargs: Any) -> VT:
        """Represent the vector as another type."""
        if any(args):
            warnings.warn("Extra arguments are ignored.", UserWarning, stacklevel=2)

        from ._transform import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, **kwargs)


class AbstractVectorDifferential(AbstractVectorBase):
    """Abstract representation of vector differentials in different systems."""

    vector_cls: eqx.AbstractClassVar[type[AbstractVector]]

    # ===============================================================
    # Convenience methods

    @partial(jax.jit, static_argnums=1)
    def represent_as(
        self, target: type[DT], position: AbstractVector, /, *args: Any, **kwargs: Any
    ) -> DT:
        """Represent the vector as another type."""
        if any(args):
            warnings.warn("Extra arguments are ignored.", UserWarning, stacklevel=2)

        from ._transform import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, position, **kwargs)
