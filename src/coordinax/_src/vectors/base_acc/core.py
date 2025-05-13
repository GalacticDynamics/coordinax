"""Abstract Accelerations."""

__all__ = ["AbstractAcc", "ACCELERATION_CLASSES"]

import functools as ft
from typing import TYPE_CHECKING, Any, cast

import jax

import unxt as u

from coordinax._src.utils import classproperty
from coordinax._src.vectors import api
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel
from coordinax._src.vectors.mixins import AvalMixin

if TYPE_CHECKING:
    import coordinax.vecs


class AbstractAcc(AvalMixin, AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of vector differentials in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered.
        """
        super().__init_subclass__(**kwargs)

        ACCELERATION_CLASSES_MUTABLE[cls] = None

    # ===============================================================
    # Vector API

    @classproperty
    @classmethod
    def cartesian_type(cls) -> "type[coordinax.vecs.AbstractAcc]":
        """Return the corresponding Cartesian vector class."""
        return api.cartesian_vector_type(cls)

    @classproperty
    @classmethod
    def time_derivative_cls(cls) -> "type[coordinax.vecs.AbstractVector]":
        """Return the corresponding time derivative class."""
        return api.time_derivative_vector_type(cls)

    @classproperty
    @classmethod
    def time_antiderivative_cls(cls) -> type[AbstractVel]:
        """Return the corresponding time antiderivative class."""
        return api.time_antiderivative_vector_type(cls)

    @classmethod
    def time_nth_derivative_cls(cls, n: int) -> "type[coordinax.vecs.AbstractVector]":
        """Return the corresponding time nth derivative class."""
        return api.time_nth_derivative_vector_type(cls, n=n)

    # ===============================================================
    # Convenience methods

    @ft.partial(jax.jit)
    def norm(
        self: "AbstractAcc", p: AbstractVel, q: AbstractPos, /
    ) -> u.Quantity["acceleration"]:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> q = cx.vecs.CartesianPos3D.from_([1, 2, 3], "kpc")
        >>> p = cx.vecs.CartesianVel3D.from_([4, 5, 6], "km/s")
        >>> a = cx.vecs.CartesianAcc3D.from_([3, 4, 0], "m/s2")

        >>> a = a.vconvert(cx.vecs.CylindricalAcc, p, q)

        >>> a.norm(p, q)
        Quantity(Array(5..., dtype=float32), unit='m / s2')

        """
        cart_cls = self.cartesian_type
        cart_acc = cast(AbstractAcc, self.vconvert(cart_cls, p, q))
        return cart_acc.norm(p, q)


# -----------------

ACCELERATION_CLASSES_MUTABLE: dict[type[AbstractAcc], None] = {}
ACCELERATION_CLASSES = ACCELERATION_CLASSES_MUTABLE.keys()
