"""Representation of coordinates in different systems."""

__all__ = ["AbstractPos", "POSITION_CLASSES"]

import functools as ft
from inspect import isabstract
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import quax_blocks
from jaxtyping import Array, Shaped
from plum import convert
from quax import quaxify

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity as FastQ

from coordinax._src.custom_types import BBtScalarQ
from coordinax._src.utils import classproperty
from coordinax._src.vectors import api
from coordinax._src.vectors.base import AbstractVector, ToUnitsOptions
from coordinax._src.vectors.mixins import AvalMixin

if TYPE_CHECKING:
    import coordinax.vecs

_vec_matmul = quaxify(jax.numpy.vectorize(jax.numpy.matmul, signature="(N,N),(N)->(N)"))


class AbstractPos(
    AvalMixin, quax_blocks.NumpyNegMixin["coordinax.vecs.AbstractPos"], AbstractVector
):  # pylint: disable=abstract-method
    """Abstract representation of coordinates in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered if it is not an abstract class.
        """
        super().__init_subclass__(**kwargs)

        # TODO: a more robust check using equinox.
        if isabstract(cls) or cls.__name__.startswith("Abstract"):
            return

        POSITION_CLASSES_MUTABLE[cls] = None

    # ===============================================================
    # Vector API

    @classproperty
    @classmethod
    def cartesian_type(cls) -> "type[coordinax.vecs.AbstractPos]":
        """Return the corresponding Cartesian vector class."""
        return api.cartesian_vector_type(cls)

    @classproperty
    @classmethod
    def time_derivative_cls(cls) -> "type[coordinax.vecs.AbstractVel]":
        """Return the corresponding time derivative class."""
        return api.time_derivative_vector_type(cls)

    @classproperty
    @classmethod
    def time_antiderivative_cls(cls) -> "type[coordinax.vecs.AbstractVector]":
        """Return the corresponding time antiderivative class."""
        return api.time_antiderivative_vector_type(cls)

    @classmethod
    def time_nth_derivative_cls(cls, n: int) -> "type[coordinax.vecs.AbstractVector]":
        """Return the corresponding time nth derivative class."""
        return api.time_nth_derivative_vector_type(cls, n=n)

    # ===============================================================
    # Python API

    def __rmatmul__(self, other: Shaped[Array, "N N"]) -> Any:
        """Matrix multiplication with another object.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax as cx

        >>> q = cx.CartesianPos3D.from_([[1., 0, 0 ],
        ...                              [0 , 1, 0 ],
        ...                              [0 , 0, 1 ]], "kpc")[None]
        >>> q.shape
        (1, 3)

        >>> R_z = rot = jnp.asarray([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])

        >>> print(R_z @ q)
        <CartesianPos3D: (x, y, z) [kpc]
            [[[ 0.  1.  0.]
              [-1.  0.  0.]
              [ 0.  0.  1.]]]>

        """
        # TODO: figure out how to do this without converting back to arrays.
        cart_cls = self.cartesian_type
        cartvec = self.vconvert(cart_cls)
        q: FastQ = convert(cartvec.uconvert(ToUnitsOptions.consistent), FastQ)
        newq = _vec_matmul(other, q)
        newvec = cart_cls.from_(newq)
        return newvec.vconvert(type(self))

    def __abs__(self) -> u.AbstractQuantity:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.vecs.CartesianPos2D.from_([3, 4], "m")
        >>> abs(vec)
        BareQuantity(Array(5., dtype=float32), unit='m')

        """
        return self.norm()  # type: ignore[misc]

    # ===============================================================
    # Convenience methods

    @ft.partial(eqx.filter_jit, inline=True)
    def norm(self) -> BBtScalarQ:
        """Return the norm of the vector.

        Returns
        -------
        Quantity
            The norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> v = cx.vecs.CartesianPos1D.from_([-1], "km")
        >>> v.norm()
        BareQuantity(Array(1., dtype=float32), unit='km')

        >>> v = cx.vecs.CartesianPos2D.from_([3, 4], "km")
        >>> v.norm()
        BareQuantity(Array(5., dtype=float32), unit='km')

        >>> v = cx.vecs.PolarPos(r=u.Quantity(3, "km"), phi=u.Quantity(90, "deg"))
        >>> v.norm()
        Distance(Array(3, dtype=int32, ...), unit='km')

        >>> v = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> v.norm()
        BareQuantity(Array(3.7416575, dtype=float32), unit='m')

        """
        return jnp.linalg.vector_norm(self, axis=-1)  # type: ignore[arg-type]


#: Registered position classes.
POSITION_CLASSES_MUTABLE: dict[type[AbstractPos], None] = {}
POSITION_CLASSES = POSITION_CLASSES_MUTABLE.keys()
