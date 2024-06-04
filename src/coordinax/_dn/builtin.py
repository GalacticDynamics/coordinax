"""Built-in vector classes."""

__all__ = [
    # Position
    "CartesianPositionND",
    # Differential
    "CartesianVelocityND",
]

from dataclasses import replace
from functools import partial
from typing import final

import equinox as eqx
import jax
from typing_extensions import override

import quaxed.array_api as xp
from unxt import Quantity

import coordinax._typing as ct
from .base import AbstractPositionND, AbstractPositionNDDifferential
from coordinax._base import AbstractVector
from coordinax._base_pos import AbstractPosition
from coordinax._utils import classproperty

##############################################################################
# Position


@final
class CartesianPositionND(AbstractPositionND):
    """N-dimensional Cartesian vector representation.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax import CartesianPositionND

    A 1D vector:

    >>> q = CartesianPositionND(Quantity([[1]], "kpc"))
    >>> q.q
    Quantity['length'](Array([[1.]], dtype=float32), unit='kpc')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = CartesianPositionND(Quantity([1, 2], "kpc"))
    >>> q.q
    Quantity['length'](Array([1., 2.], dtype=float32), unit='kpc')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = CartesianPositionND(Quantity([1, 2, 3], "kpc"))
    >>> q.q
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='kpc')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = CartesianPositionND(Quantity([1, 2, 3, 4], "kpc"))
    >>> q.q
    Quantity['length'](Array([1., 2., 3., 4.], dtype=float32), unit='kpc')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = CartesianPositionND(Quantity([1, 2, 3, 4, 5], "kpc"))
    >>> q.q
    Quantity['length'](Array([1., 2., 3., 4., 5.], dtype=float32), unit='kpc')
    >>> q.shape
    ()

    """

    q: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""N-D coordinate :math:`\vec{x} \in (-\infty,+\infty)`.

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @classproperty
    @classmethod
    @override
    def differential_cls(cls) -> type["CartesianVelocityND"]:  # type: ignore[override]
        return CartesianVelocityND

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPositionND

        A 3D vector:

        >>> q = CartesianPositionND(Quantity([1, 2, 3], "kpc"))
        >>> (-q).q
        Quantity['length'](Array([-1., -2., -3.], dtype=float32), unit='kpc')

        """
        return replace(self, q=-self.q)

    # -----------------------------------------------------
    # Binary operations

    @AbstractVector.__add__.dispatch  # type: ignore[misc]
    def __add__(
        self: "CartesianPositionND", other: AbstractPosition, /
    ) -> "CartesianPositionND":
        """Add two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPositionND

        A 3D vector:

        >>> q1 = CartesianPositionND(Quantity([1, 2, 3], "kpc"))
        >>> q2 = CartesianPositionND(Quantity([2, 3, 4], "kpc"))
        >>> (q1 + q2).q
        Quantity['length'](Array([3., 5., 7.], dtype=float32), unit='kpc')

        """
        cart = other.represent_as(CartesianPositionND)
        return replace(self, q=self.q + cart.q)

    @AbstractVector.__sub__.dispatch  # type: ignore[misc]
    def __sub__(
        self: "CartesianPositionND", other: AbstractPosition, /
    ) -> "CartesianPositionND":
        """Subtract two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPositionND

        A 3D vector:

        >>> q1 = CartesianPositionND(Quantity([1, 2, 3], "kpc"))
        >>> q2 = CartesianPositionND(Quantity([2, 3, 4], "kpc"))
        >>> (q1 - q2).q
        Quantity['length'](Array([-1., -1., -1.], dtype=float32), unit='kpc')

        """
        cart = other.represent_as(CartesianPositionND)
        return replace(self, q=self.q - cart.q)

    # -----------------------------------------------------

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPositionND

        A 3D vector:

        >>> q = CartesianPositionND(Quantity([1, 2, 3], "kpc"))
        >>> q.norm()
        Quantity['length'](Array(3.7416575, dtype=float32), unit='kpc')

        """
        return xp.linalg.vector_norm(self.q, axis=-1)


##############################################################################
# Differential


@final
class CartesianVelocityND(AbstractPositionNDDifferential):
    """Cartesian differential representation.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax import CartesianVelocityND

    A 1D vector:

    >>> q = CartesianVelocityND(Quantity([[1]], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([[1.]], dtype=float32), unit='km / s')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = CartesianVelocityND(Quantity([1, 2], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = CartesianVelocityND(Quantity([1, 2, 3], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2., 3.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = CartesianVelocityND(Quantity([1, 2, 3, 4], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2., 3., 4.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = CartesianVelocityND(Quantity([1, 2, 3, 4, 5], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2., 3., 4., 5.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    """

    d_q: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""N-D speed :math:`d\vec{x}/dt \in (-\infty, \infty).

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPositionND]:
        return CartesianPositionND

    @partial(jax.jit)
    def norm(self, _: AbstractPositionND | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianVelocityND

        A 3D vector:

        >>> c = CartesianVelocityND(Quantity([1, 2, 3], "km/s"))
        >>> c.norm()
        Quantity['speed'](Array(3.7416575, dtype=float32), unit='km / s')

        """
        return xp.linalg.vector_norm(self.d_q, axis=-1)
