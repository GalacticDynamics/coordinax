"""Representation of coordinates in different systems."""

__all__ = ["AbstractPos", "POSITION_CLASSES"]

from abc import abstractmethod
from functools import partial
from inspect import isabstract
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
from jaxtyping import Array, Shaped
from plum import convert
from quax import quaxify

import quaxed.numpy as jnp
from quaxed.experimental import arrayish
from unxt.quantity import AbstractQuantity, UncheckedQuantity as FastQ

from coordinax._src.typing import BatchableScalarQ
from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractVector, ToUnitsOptions
from coordinax._src.vectors.mixins import AvalMixin

if TYPE_CHECKING:
    import coordinax.vecs

_vec_matmul = quaxify(jax.numpy.vectorize(jax.numpy.matmul, signature="(N,N),(N)->(N)"))


class AbstractPos(
    AvalMixin, arrayish.NumpyNegMixin["coordinax.vecs.AbstractPos"], AbstractVector
):  # pylint: disable=abstract-method
    """Abstract representation of coordinates in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered if it is not an abstract class.
        """
        # TODO: a more robust check using equinox.
        if isabstract(cls) or cls.__name__.startswith("Abstract"):
            return

        POSITION_CLASSES_MUTABLE[cls] = None

    # ===============================================================
    # Vector API

    @classproperty
    @classmethod
    @abstractmethod
    def _cartesian_cls(cls) -> "type[coordinax.vecs.AbstractVector]":
        """Return the corresponding Cartesian vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialPos._cartesian_cls
        <class 'coordinax...CartesianPos1D'>

        >>> cx.SphericalPos._cartesian_cls
        <class 'coordinax...CartesianPos3D'>

        """
        raise NotImplementedError  # pragma: no cover

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["coordinax.vecs.AbstractVel"]:
        """Return the corresponding differential vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialPos.differential_cls.__name__
        'RadialVel'

        >>> cx.SphericalPos.differential_cls.__name__
        'SphericalVel'

        """
        raise NotImplementedError  # pragma: no cover

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
        <CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[[ 0.  1.  0.]
              [-1.  0.  0.]
              [ 0.  0.  1.]]]>

        """
        # TODO: figure out how to do this without converting back to arrays.
        cartvec = self.vconvert(self._cartesian_cls)
        q: FastQ = convert(cartvec.uconvert(ToUnitsOptions.consistent), FastQ)
        newq = _vec_matmul(other, q)
        newvec = self._cartesian_cls.from_(newq)
        return newvec.vconvert(type(self))

    def __abs__(self) -> AbstractQuantity:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> vec = cx.vecs.CartesianPos2D.from_([3, 4], "m")
        >>> abs(vec)
        UncheckedQuantity(Array(5., dtype=float32), unit='m')

        """
        return self.norm()  # type: ignore[misc]

    # ===============================================================
    # Convenience methods

    @partial(eqx.filter_jit, inline=True)
    def norm(self) -> BatchableScalarQ:
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
        UncheckedQuantity(Array(1., dtype=float32), unit='km')

        >>> v = cx.vecs.CartesianPos2D.from_([3, 4], "km")
        >>> v.norm()
        UncheckedQuantity(Array(5., dtype=float32), unit='km')

        >>> v = cx.vecs.PolarPos(r=u.Quantity(3, "km"), phi=u.Quantity(90, "deg"))
        >>> v.norm()
        Distance(Array(3, dtype=int32, ...), unit='km')

        >>> v = cx.CartesianPos3D.from_([1, 2, 3], "m")
        >>> v.norm()
        UncheckedQuantity(Array(3.7416575, dtype=float32), unit='m')

        """
        return jnp.linalg.vector_norm(self, axis=-1)  # type: ignore[arg-type]


#: Registered position classes.
POSITION_CLASSES_MUTABLE: dict[type[AbstractPos], None] = {}
POSITION_CLASSES = POSITION_CLASSES_MUTABLE.keys()
