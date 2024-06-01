"""Built-in vector classes."""

__all__ = [
    # Position
    "CartesianNDVector",
    # Differential
    "CartesianDifferentialND",
]

from dataclasses import replace
from functools import partial
from typing import Any, final

import equinox as eqx
import jax
from typing_extensions import override

import quaxed.array_api as xp
from unxt import Quantity

import coordinax._typing as ct
from .base import AbstractNDVector, AbstractNDVectorDifferential
from coordinax._base_pos import AbstractPosition
from coordinax._utils import classproperty

##############################################################################
# Position


@final
class CartesianNDVector(AbstractNDVector):
    """N-dimensional Cartesian vector representation.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax import CartesianNDVector

    A 1D vector:

    >>> q = CartesianNDVector(Quantity([[1]], "kpc"))
    >>> q.q
    Quantity['length'](Array([[1.]], dtype=float32), unit='kpc')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = CartesianNDVector(Quantity([1, 2], "kpc"))
    >>> q.q
    Quantity['length'](Array([1., 2.], dtype=float32), unit='kpc')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = CartesianNDVector(Quantity([1, 2, 3], "kpc"))
    >>> q.q
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='kpc')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = CartesianNDVector(Quantity([1, 2, 3, 4], "kpc"))
    >>> q.q
    Quantity['length'](Array([1., 2., 3., 4.], dtype=float32), unit='kpc')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = CartesianNDVector(Quantity([1, 2, 3, 4, 5], "kpc"))
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
    def differential_cls(cls) -> type["CartesianDifferentialND"]:  # type: ignore[override]
        return CartesianDifferentialND

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianNDVector

        A 3D vector:

        >>> q = CartesianNDVector(Quantity([1, 2, 3], "kpc"))
        >>> (-q).q
        Quantity['length'](Array([-1., -2., -3.], dtype=float32), unit='kpc')

        """
        return replace(self, q=-self.q)

    # -----------------------------------------------------
    # Binary operations

    def __add__(self, other: Any, /) -> "CartesianNDVector":
        """Add two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianNDVector

        A 3D vector:

        >>> q1 = CartesianNDVector(Quantity([1, 2, 3], "kpc"))
        >>> q2 = CartesianNDVector(Quantity([2, 3, 4], "kpc"))
        >>> (q1 + q2).q
        Quantity['length'](Array([3., 5., 7.], dtype=float32), unit='kpc')

        """
        if not isinstance(other, AbstractPosition):
            msg = f"Cannot add {self._cartesian_cls!r} and {type(other)!r}."
            raise TypeError(msg)

        cart = other.represent_as(CartesianNDVector)
        return replace(self, q=self.q + cart.q)

    def __sub__(self, other: Any, /) -> "CartesianNDVector":
        """Subtract two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianNDVector

        A 3D vector:

        >>> q1 = CartesianNDVector(Quantity([1, 2, 3], "kpc"))
        >>> q2 = CartesianNDVector(Quantity([2, 3, 4], "kpc"))
        >>> (q1 - q2).q
        Quantity['length'](Array([-1., -1., -1.], dtype=float32), unit='kpc')

        """
        if not isinstance(other, AbstractPosition):
            msg = f"Cannot subtract {self._cartesian_cls!r} and {type(other)!r}."
            raise TypeError(msg)

        cart = other.represent_as(CartesianNDVector)
        return replace(self, q=self.q - cart.q)

    # -----------------------------------------------------

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianNDVector

        A 3D vector:

        >>> q = CartesianNDVector(Quantity([1, 2, 3], "kpc"))
        >>> q.norm()
        Quantity['length'](Array(3.7416575, dtype=float32), unit='kpc')

        """
        return xp.linalg.vector_norm(self.q, axis=-1)


##############################################################################
# Differential


@final
class CartesianDifferentialND(AbstractNDVectorDifferential):
    """Cartesian differential representation.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax import CartesianDifferentialND

    A 1D vector:

    >>> q = CartesianDifferentialND(Quantity([[1]], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([[1.]], dtype=float32), unit='km / s')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = CartesianDifferentialND(Quantity([1, 2], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = CartesianDifferentialND(Quantity([1, 2, 3], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2., 3.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = CartesianDifferentialND(Quantity([1, 2, 3, 4], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2., 3., 4.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = CartesianDifferentialND(Quantity([1, 2, 3, 4, 5], "km/s"))
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
    def integral_cls(cls) -> type[CartesianNDVector]:
        return CartesianNDVector

    @partial(jax.jit)
    def norm(self, _: AbstractNDVector | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianDifferentialND

        A 3D vector:

        >>> c = CartesianDifferentialND(Quantity([1, 2, 3], "km/s"))
        >>> c.norm()
        Quantity['speed'](Array(3.7416575, dtype=float32), unit='km / s')

        """
        return xp.linalg.vector_norm(self.d_q, axis=-1)
