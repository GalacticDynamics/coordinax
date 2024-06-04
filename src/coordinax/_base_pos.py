"""Representation of coordinates in different systems."""

__all__ = ["AbstractPosition"]

import operator
import warnings
from abc import abstractmethod
from dataclasses import replace
from functools import partial
from inspect import isabstract
from typing import TYPE_CHECKING, Any, TypeVar

import jax
from jaxtyping import ArrayLike

from unxt import Quantity

from ._base import AbstractVector
from ._utils import classproperty, dataclass_items

if TYPE_CHECKING:
    from typing_extensions import Self

VT = TypeVar("VT", bound="AbstractPosition")

VECTOR_CLASSES: set[type["AbstractPosition"]] = set()


class AbstractPosition(AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of coordinates in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered if it is not an abstract class.
        """
        # TODO: a more robust check using equinox.
        if isabstract(cls) or cls.__name__.startswith("Abstract"):
            return

        VECTOR_CLASSES.add(cls)

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractVelocity"]:
        """Return the corresponding differential vector class.

        Examples
        --------
        >>> from coordinax import RadialPosition, SphericalPosition

        >>> RadialPosition.differential_cls.__name__
        'RadialVelocity'

        >>> SphericalPosition.differential_cls.__name__
        'SphericalVelocity'

        """
        raise NotImplementedError

    # ===============================================================
    # Array

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        The default implementation is to go through Cartesian coordinates.
        """
        cart = self.represent_as(self._cartesian_cls)
        return (-cart).represent_as(type(self))

    # -----------------------------------------------------
    # Binary arithmetic operations

    @AbstractVector.__add__.dispatch  # type: ignore[misc]
    def __add__(
        self: "AbstractPosition", other: "AbstractPosition"
    ) -> "AbstractPosition":  # TODO: use Self
        """Add another object to this vector."""
        # The base implementation is to convert to Cartesian and perform the
        # operation.  Cartesian coordinates do not have any branch cuts or
        # singularities or ranges that need to be handled, so this is a safe
        # default.
        return operator.add(
            self.represent_as(self._cartesian_cls),
            other.represent_as(self._cartesian_cls),
        ).represent_as(type(self))

    @AbstractVector.__sub__.dispatch  # type: ignore[misc]
    def __sub__(
        self: "AbstractPosition", other: "AbstractPosition"
    ) -> "AbstractPosition":  # TODO: use Self
        """Add another object to this vector."""
        # The base implementation is to convert to Cartesian and perform the
        # operation.  Cartesian coordinates do not have any branch cuts or
        # singularities or ranges that need to be handled, so this is a safe
        # default.
        return operator.sub(
            self.represent_as(self._cartesian_cls),
            other.represent_as(self._cartesian_cls),
        ).represent_as(type(self))

    @AbstractVector.__mul__.dispatch  # type: ignore[misc]
    def __mul__(
        self: "AbstractPosition", other: ArrayLike
    ) -> "AbstractPosition":  # TODO: use Self
        return replace(self, **{k: v * other for k, v in dataclass_items(self)})

    @AbstractVector.__rmul__.dispatch  # type: ignore[misc]
    def __rmul__(
        self: "AbstractPosition", other: ArrayLike
    ) -> "AbstractPosition":  # TODO: use Self
        return replace(self, **{k: other * v for k, v in dataclass_items(self)})

    @AbstractVector.__truediv__.dispatch  # type: ignore[misc]
    def __truediv__(
        self: "AbstractPosition", other: ArrayLike
    ) -> "AbstractPosition":  # TODO: use Self
        return replace(self, **{k: v / other for k, v in dataclass_items(self)})

    # ===============================================================
    # Convenience methods

    @partial(jax.jit, static_argnums=1)
    def represent_as(self, target: type[VT], /, *args: Any, **kwargs: Any) -> VT:
        """Represent the vector as another type.

        Parameters
        ----------
        target : type[AbstractPosition]
            The type to represent the vector as.
        *args : Any
            Extra arguments. Raises a warning if any are given.
        **kwargs : Any
            Extra keyword arguments.

        Returns
        -------
        AbstractPosition
            The vector represented as the target type.

        Warns
        -----
        UserWarning
            If extra arguments are given.

        Examples
        --------
        We assume the following imports:

        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D, SphericalPosition

        We can represent a vector as another type:

        >>> x, y, z = Quantity(1, "meter"), Quantity(2, "meter"), Quantity(3, "meter")
        >>> vec = CartesianPosition3D(x=x, y=y, z=z)
        >>> sph = vec.represent_as(SphericalPosition)
        >>> sph
        SphericalPosition(
            r=Distance(value=f32[], unit=Unit("m")),
            theta=Quantity[...](value=f32[], unit=Unit("rad")),
            phi=Quantity[...](value=f32[], unit=Unit("rad")) )
        >>> sph.r
        Distance(Array(3.7416575, dtype=float32), unit='m')

        """
        if any(args):
            warnings.warn("Extra arguments are ignored.", UserWarning, stacklevel=2)

        from ._transform import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, **kwargs)

    @partial(jax.jit)
    def norm(self) -> Quantity["length"]:
        """Return the norm of the vector.

        Returns
        -------
        Quantity
            The norm of the vector.

        Examples
        --------
        We assume the following imports:
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition3D

        We can compute the norm of a vector
        >>> x, y, z = Quantity(1, "meter"), Quantity(2, "meter"), Quantity(3, "meter")
        >>> vec = CartesianPosition3D(x=x, y=y, z=z)
        >>> vec.norm()
        Quantity['length'](Array(3.7416575, dtype=float32), unit='m')

        """
        return self.represent_as(self._cartesian_cls).norm()
