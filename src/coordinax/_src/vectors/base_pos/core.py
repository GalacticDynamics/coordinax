"""Representation of coordinates in different systems."""

__all__ = ["AbstractPos"]

from abc import abstractmethod
from functools import partial
from inspect import isabstract
from typing import Any

import equinox as eqx
import jax
from jaxtyping import Array, Shaped
from plum import convert
from quax import quaxify

import quaxed.numpy as jnp
import unxt as u

from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractVector, ToUnitsOptions
from coordinax._src.vectors.mixins import AvalMixin

# TODO: figure out public API for this
POSITION_CLASSES: set[type["AbstractPos"]] = set()

_vec_matmul = quaxify(jax.numpy.vectorize(jax.numpy.matmul, signature="(N,N),(N)->(N)"))


class AbstractPos(AvalMixin, AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of coordinates in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered if it is not an abstract class.
        """
        # TODO: a more robust check using equinox.
        if isabstract(cls) or cls.__name__.startswith("Abstract"):
            return

        POSITION_CLASSES.add(cls)

    @classproperty
    @classmethod
    @abstractmethod
    def _cartesian_cls(cls) -> type["AbstractVector"]:
        """Return the corresponding Cartesian vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialPos._cartesian_cls
        <class 'coordinax...CartesianPos1D'>

        >>> cx.SphericalPos._cartesian_cls
        <class 'coordinax...CartesianPos3D'>

        """
        # TODO: something nicer than this for getting the corresponding class
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractVel"]:
        """Return the corresponding differential vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialPos.differential_cls.__name__
        'RadialVel'

        >>> cx.SphericalPos.differential_cls.__name__
        'SphericalVel'

        """
        raise NotImplementedError

    # ===============================================================
    # Unary operations

    __neg__ = jnp.negative

    # ===============================================================
    # Binary operations

    def __eq__(self: "AbstractPos", other: object) -> Any:
        """Element-wise equality of two positions.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import unxt as u
        >>> import coordinax as cx

        Showing the broadcasting, then element-wise comparison of two vectors:

        >>> vec1 = cx.CartesianPos3D.from_([[1, 2, 3], [1, 2, 4]], "m")
        >>> vec2 = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> jnp.equal(vec1, vec2)
        Array([ True, False], dtype=bool)

        Showing the change of representation:

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> vec1 = vec.vconvert(cx.SphericalPos)
        >>> vec2 = vec.vconvert(cx.vecs.MathSphericalPos)
        >>> jnp.equal(vec1, vec2)
        Array(True, dtype=bool)

        Quick run-through of each dimensionality:

        >>> vec1 = cx.vecs.CartesianPos1D.from_([1], "m")
        >>> vec2 = cx.vecs.RadialPos.from_([1], "m")
        >>> jnp.equal(vec1, vec2)
        Array(True, dtype=bool)

        >>> vec1 = cx.vecs.CartesianPos2D.from_([2, 0], "m")
        >>> vec2 = cx.vecs.PolarPos(r=u.Quantity(2, "m"), phi=u.Quantity(0, "rad"))
        >>> jnp.equal(vec1, vec2)
        Array(True, dtype=bool)

        """
        if not isinstance(other, AbstractPos):
            return NotImplemented

        rhs = other.vconvert(type(self))
        return super().__eq__(rhs)

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
        <CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[[ 0.  1.  0.]
              [-1.  0.  0.]
              [ 0.  0.  1.]]]>

        """
        # TODO: figure out how to do this without converting back to arrays.
        cartvec = self.vconvert(self._cartesian_cls)
        q: u.Quantity = convert(cartvec.uconvert(ToUnitsOptions.consistent), u.Quantity)
        newq = _vec_matmul(other, q)
        newvec = self._cartesian_cls.from_(newq)
        return newvec.vconvert(type(self))

    # ===============================================================
    # Convenience methods

    @partial(eqx.filter_jit, inline=True)
    def norm(self) -> BatchableLength:
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
        Quantity['length'](Array(1., dtype=float32), unit='km')

        >>> v = cx.vecs.CartesianPos2D.from_([3, 4], "km")
        >>> v.norm()
        Quantity['length'](Array(5., dtype=float32), unit='km')

        >>> v = cx.vecs.PolarPos(r=u.Quantity(3, "km"), phi=u.Quantity(90, "deg"))
        >>> v.norm()
        Distance(Array(3, dtype=int32, ...), unit='km')

        >>> v = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> v.norm()
        Quantity['length'](Array(3.7416575, dtype=float32), unit='m')

        """
        return jnp.linalg.vector_norm(self, axis=-1)
