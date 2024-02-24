"""Representation of coordinates in different systems."""

__all__ = ["AbstractVectorBase", "AbstractVector", "AbstractVectorDifferential"]

import warnings
from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar

import array_api_jax_compat as xp
import equinox as eqx
import jax
import jax.numpy as jnp
from jax_quantity import Quantity

from ._utils import dataclass_items, full_shaped

if TYPE_CHECKING:
    from typing_extensions import Self

if TYPE_CHECKING:
    from typing_extensions import Self

BT = TypeVar("BT", bound="AbstractVectorBase")
VT = TypeVar("VT", bound="AbstractVector")
DT = TypeVar("DT", bound="AbstractVectorDifferential")


_0m = Quantity(0, "meter")
_0d = Quantity(0, "rad")
_pid = Quantity(xp.pi, "rad")
_2pid = Quantity(2 * xp.pi, "rad")


class AbstractVectorBase(eqx.Module):  # type: ignore[misc]
    """Base class for all vector types.

    A vector is a collection of components that can be represented in different
    coordinate systems. This class provides a common interface for all vector
    types. All fields of the vector are expected to be components of the vector.
    """

    # ===============================================================
    # Array

    @property
    def shape(self) -> Any:
        """Get the shape of the vector's components.

        When represented as a single array, the vector has an additional
        dimension at the end for the components.
        """
        return jnp.broadcast_shapes(*self.shapes.values())

    def flatten(self) -> "Self":
        """Flatten the vector."""
        return replace(self, **{k: v.flatten() for k, v in dataclass_items(self)})

    def reshape(self, *args: Any, order: str = "C") -> "Self":
        """Reshape the components of the vector."""
        # TODO: enable not needing to make a full-shaped copy
        full = full_shaped(self)
        return replace(
            self, **{k: v.reshape(*args, order=order) for k, v in dataclass_items(full)}
        )

    # ===============================================================
    # Collection

    def asdict(
        self,
        *,
        dict_factory: Callable[[Any], Mapping[str, Quantity]] = dict,  # TODO: full hint
    ) -> Mapping[str, Quantity]:
        """Return the vector as a Mapping.

        Parameters
        ----------
        dict_factory : type[Mapping]
            The type of the mapping to return.

        Returns
        -------
        Mapping[str, Quantity]
            The vector as a mapping.

        See Also
        --------
        `dataclasses.asdict`
            This applies recursively to the components of the vector.

        """
        return dict_factory(dataclass_items(self))

    @property
    def components(self) -> tuple[str, ...]:
        """Vector component names."""
        return tuple(f.name for f in fields(self))

    @property
    def shapes(self) -> Mapping[str, tuple[int, ...]]:
        """Get the shapes of the vector's components."""
        return MappingProxyType({k: v.shape for k, v in dataclass_items(self)})

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

    @abstractmethod
    def norm(self) -> Quantity:
        """Return the norm of the vector."""
        # TODO: make a generic method that works on all dimensions
        raise NotImplementedError


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

    @abstractmethod
    def norm(self, position: AbstractVector, /) -> Quantity:
        """Return the norm of the vector."""
        # TODO: make a generic method that works on all dimensions
        raise NotImplementedError
