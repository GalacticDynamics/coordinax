"""Vector."""

from typing import ClassVar, overload

__all__ = (
    "AbstractRoleFlag",
    "Pos",
    "pos",
    "Vel",
    "vel",
    "Acc",
    "acc",
    "DIM_TO_ROLE_MAP",
)

import abc
import functools as ft

import unxt as u

from .euclidean import AbstractRep

Time = u.dimension("time")


@overload
def d_dt_dim(dim: None, order: int, /) -> None: ...
@overload
def d_dt_dim(dim: u.AbstractDimension | str, order: int, /) -> u.AbstractDimension: ...
def d_dt_dim(
    dim: u.AbstractDimension | str | None, order: int, /
) -> u.AbstractDimension | None:
    """Return the dimension of the time derivative of the given dimension."""
    if dim is None:
        return None
    return u.dimension(dim) / (Time**order)


class AbstractRoleFlag:
    """Flag for vector role (position, velocity, acceleration, etc.).

    Attributes
    ----------
    order : int
        Time-derivative order of the role (e.g. 0=pos, 1=vel, 2=acc, ...).

    """

    order: ClassVar[int]  # type: ignore[misc]
    """Time-derivative order of the role (e.g. 0=pos, 1=vel, 2=acc, ...)."""

    @classmethod
    @ft.cache
    def dimensions(cls, rep: AbstractRep, /) -> dict[str, u.AbstractDimension | None]:
        """Return the dimensions for this role for the given representation."""
        return {
            c: d_dt_dim(d, cls.order)
            for c, d in zip(rep.components, rep.coord_dimensions, strict=True)
        }

    @classmethod
    @abc.abstractmethod
    def derivative(cls) -> type["AbstractRoleFlag"]:
        """Return role flag for the time derivative of this role."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    @abc.abstractmethod
    def antiderivative(cls) -> type["AbstractRoleFlag"]:
        """Return role flag for the time antiderivative of this role."""
        raise NotImplementedError  # pragma: no cover


class Pos(AbstractRoleFlag):
    """Position role flag (0th time derivative)."""

    order: ClassVar[int] = 0

    @classmethod
    def derivative(cls) -> type[AbstractRoleFlag]:
        """Return role flag for the time derivative of this role."""
        return Vel

    @classmethod
    def antiderivative(cls) -> type[AbstractRoleFlag]:
        """Return role flag for the time antiderivative of this role."""
        return NotImplemented


pos = Pos()


class Vel(AbstractRoleFlag):
    """Velocity role flag (1st time derivative)."""

    order: ClassVar[int] = 1

    @classmethod
    def derivative(cls) -> type[AbstractRoleFlag]:
        """Return role flag for the time derivative of this role."""
        return Acc

    @classmethod
    def antiderivative(cls) -> type[AbstractRoleFlag]:
        """Return role flag for the time antiderivative of this role."""
        return Pos


vel = Vel()


class Acc(AbstractRoleFlag):
    """Acceleration role flag (2nd time derivative)."""

    order: ClassVar[int] = 2

    @classmethod
    def derivative(cls) -> type[AbstractRoleFlag]:
        """Return role flag for the time derivative of this role."""
        return NotImplemented

    @classmethod
    def antiderivative(cls) -> type[AbstractRoleFlag]:
        """Return role flag for the time antiderivative of this role."""
        return Vel


acc = Acc()


# Mapping from dimension to role flag
DIM_TO_ROLE_MAP: dict[u.AbstractDimension, type["AbstractRoleFlag"]] = {
    u.dimension("length"): Pos,
    u.dimension("angle"): Pos,
    u.dimension("speed"): Vel,
    u.dimension("angular speed"): Vel,
    u.dimension("acceleration"): Acc,
    u.dimension("angular acceleration"): Acc,
}
