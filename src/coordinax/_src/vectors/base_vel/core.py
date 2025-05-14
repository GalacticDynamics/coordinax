"""Representation of velocities in different systems."""

__all__ = ["AbstractVel", "VELOCITY_CLASSES"]

import functools as ft
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx

import unxt as u

from coordinax._src.utils import classproperty
from coordinax._src.vectors import api
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.mixins import AvalMixin

if TYPE_CHECKING:
    import coordinax.vecs


class AbstractVel(AvalMixin, AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of vector differentials in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered.
        """
        super().__init_subclass__(**kwargs)

        VELOCITY_CLASSES_MUTABLE[cls] = None

    # ===============================================================
    # Vector API

    @classproperty
    @classmethod
    def cartesian_type(cls) -> "type[coordinax.vecs.AbstractVel]":
        """Return the corresponding Cartesian vector class."""
        return api.cartesian_vector_type(cls)

    @classproperty
    @classmethod
    def time_derivative_cls(cls) -> "type[coordinax.vecs.AbstractAcc]":
        """Return the corresponding time derivative class."""
        return api.time_derivative_vector_type(cls)

    @classproperty
    @classmethod
    def time_antiderivative_cls(cls) -> type[AbstractPos]:
        """Return the corresponding time antiderivative class."""
        return api.time_antiderivative_vector_type(cls)

    @classmethod
    def time_nth_derivative_cls(
        cls, *, n: int
    ) -> "type[coordinax.vecs.AbstractVector]":
        """Return the corresponding time nth derivative class."""
        return api.time_nth_derivative_vector_type(cls, n=n)

    # ===============================================================
    # Convenience methods

    @ft.partial(eqx.filter_jit)
    def norm(self: "AbstractVel", q: AbstractPos, /) -> u.Quantity["speed"]:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> q = cx.vecs.CartesianPos2D.from_([1, 1], "km")
        >>> p = cx.vecs.PolarVel(r=u.Quantity(1, "km/s"), phi=u.Quantity(1, "deg/s"))

        >>> p.norm(q).uconvert('km / s')
        Quantity(Array(1.0003046, dtype=float32), unit='km / s')

        """
        cart_vel = cast(AbstractVel, self.vconvert(self.cartesian_type, q))
        return cart_vel.norm(q)  # type: ignore[call-arg,misc]


#: Registered velocity classes.
VELOCITY_CLASSES_MUTABLE: dict[type[AbstractVel], None] = {}
VELOCITY_CLASSES = VELOCITY_CLASSES_MUTABLE.keys()
