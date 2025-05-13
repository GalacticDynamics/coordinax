"""Built-in vector classes."""

__all__ = ["CartesianAccND", "CartesianPosND", "CartesianVelND"]

import functools as ft
from typing import final
from typing_extensions import override

import equinox as eqx
import jax
import quax_blocks

import quaxed.numpy as jnp
import unxt as u

import coordinax._src.custom_types as ct
from .base import AbstractAccND, AbstractPosND, AbstractVelND
from coordinax._src.distances import BBtLength
from coordinax._src.vectors.base.cartesian import AbstractCartesian

##############################################################################
# Position


@final
class CartesianPosND(AbstractPosND, AbstractCartesian, quax_blocks.NumpyNegMixin):
    """N-dimensional Cartesian vector representation.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    A 1D vector:

    >>> q = cx.vecs.CartesianPosND.from_([[1]], "km")
    >>> q.q
    Quantity(Array([[1]], dtype=int32), unit='km')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2], "km"))
    >>> q.q
    Quantity(Array([1, 2], dtype=int32), unit='km')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3], "km"))
    >>> q.q
    Quantity(Array([1, 2, 3], dtype=int32), unit='km')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4], "km"))
    >>> q.q
    Quantity(Array([1, 2, 3, 4], dtype=int32), unit='km')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4, 5], "km"))
    >>> q.q
    Quantity(Array([1, 2, 3, 4, 5], dtype=int32), unit='km')
    >>> q.shape
    ()

    """

    q: BBtLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""N-D coordinate :math:`\vec{x} \in (-\infty,+\infty)`.

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @override
    def _dimensionality(self) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        A 3D vector:

        >>> cx.vecs.CartesianPosND(u.Quantity([1, 2, 3], "km"))._dimensionality()
        3

        """
        return self.q.shape[-1]

    # ===============================================================
    # Quax API

    @override
    def aval(self) -> jax.core.ShapedArray:
        """Simpler aval than superclass."""
        return self.q.aval()

    # -----------------------------------------------------

    @override
    @ft.partial(eqx.filter_jit, inline=True)
    def norm(self) -> BBtLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        A 3D vector:

        >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3], "km"))
        >>> q.norm()
        Quantity(Array(3.7416575, dtype=float32), unit='km')

        """
        return jnp.linalg.vector_norm(self.q, axis=-1)


##############################################################################
# Velocity


@final
class CartesianVelND(AbstractCartesian, AbstractVelND):
    """Cartesian differential representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    A 1D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([[1]], "km/s"))
    >>> q.q
    Quantity(Array([[1]], dtype=int32), unit='km / s')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([1, 2], "km/s"))
    >>> q.q
    Quantity(Array([1, 2], dtype=int32), unit='km / s')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3], "km/s"))
    >>> q.q
    Quantity(Array([1, 2, 3], dtype=int32), unit='km / s')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3, 4], "km/s"))
    >>> q.q
    Quantity(Array([1, 2, 3, 4], dtype=int32), unit='km / s')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3, 4, 5], "km/s"))
    >>> q.q
    Quantity(Array([1, 2, 3, 4, 5], dtype=int32), unit='km / s')
    >>> q.shape
    ()

    """

    q: ct.BBtSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""N-D speed :math:`d\vec{x}/dt \in (-\infty, \infty).

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @override
    def _dimensionality(self) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        A 3D vector:

        >>> cx.vecs.CartesianVelND(u.Quantity([1, 2, 3], "km/s"))._dimensionality()
        3

        """
        return self.q.shape[-1]

    @override
    @ft.partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractPosND | None = None, /) -> ct.BBtSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        A 3D vector:

        >>> c = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3], "km/s"))
        >>> c.norm()
        Quantity(Array(3.7416575, dtype=float32), unit='km / s')

        """
        return jnp.linalg.vector_norm(self.q, axis=-1)


##############################################################################
# Acceleration


@final
class CartesianAccND(AbstractCartesian, AbstractAccND):
    """Cartesian N-dimensional acceleration representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    A 1D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([[1]], "km/s2"))
    >>> q.q
    Quantity(Array([[1]], dtype=int32), unit='km / s2')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([1, 2], "km/s2"))
    >>> q.q
    Quantity(Array([1, 2], dtype=int32), unit='km / s2')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([1, 2, 3], "km/s2"))
    >>> q.q
    Quantity(Array([1, 2, 3], dtype=int32), unit='km / s2')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([1, 2, 3, 4], "km/s2"))
    >>> q.q
    Quantity(Array([1, 2, 3, 4], dtype=int32), unit='km / s2')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([1, 2, 3, 4, 5], "km/s2"))
    >>> q.q
    Quantity(Array([1, 2, 3, 4, 5], dtype=int32), unit='km / s2')
    >>> q.shape
    ()

    """

    q: ct.BBtAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""N-D acceleration :math:`d\vec{x}/dt^2 \in (-\infty, \infty).

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @override
    def _dimensionality(self) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        A 3D vector:

        >>> cx.vecs.CartesianAccND(u.Quantity([1, 2, 3], "km/s2"))._dimensionality()
        3

        """
        return self.q.shape[-1]

    @override
    @ft.partial(eqx.filter_jit, inline=True)
    def norm(  # type: ignore[override]
        self,
        velocity: AbstractVelND | None = None,
        position: AbstractPosND | None = None,
        /,
    ) -> ct.BBtAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        A 3D vector:

        >>> c = cx.vecs.CartesianAccND(u.Quantity([1, 2, 3], "km/s2"))
        >>> c.norm()
        Quantity(Array(3.7416575, dtype=float32), unit='km / s2')

        """
        return jnp.linalg.vector_norm(self.q, axis=-1)
