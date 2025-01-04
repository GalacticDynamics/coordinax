"""Representation of velocities in different systems."""

__all__ = ["AbstractVel"]

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u

from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_pos import AbstractPos

if TYPE_CHECKING:
    from typing import Self

VELOCITY_CLASSES: set[type["AbstractVel"]] = set()


class AbstractVel(AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of vector differentials in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered.
        """
        VELOCITY_CLASSES.add(cls)

    @classproperty
    @classmethod
    @abstractmethod
    def _cartesian_cls(cls) -> type["AbstractVector"]:
        """Return the corresponding Cartesian vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialVel._cartesian_cls
        <class 'coordinax...CartesianVel1D'>

        >>> cx.SphericalVel._cartesian_cls
        <class 'coordinax...CartesianVel3D'>

        """
        # TODO: something nicer than this for getting the corresponding class
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type["AbstractPos"]:
        """Return the corresponding vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialVel.integral_cls.__name__
        'RadialPos'

        >>> cx.SphericalVel.integral_cls.__name__
        'SphericalPos'

        """
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> type["AbstractAcc"]:
        """Return the corresponding differential vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialVel.differential_cls.__name__
        'RadialAcc'

        >>> cx.SphericalVel.differential_cls.__name__
        'SphericalAcc'

        """
        raise NotImplementedError

    # ===============================================================
    # Quax

    def aval(self) -> jax.core.ShapedArray:
        """Return the vector as a JAX array."""
        raise NotImplementedError

    # ===============================================================
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> dr = cx.vecs.RadialVel.from_([1], "m/s")
        >>> -dr
        RadialVel( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

        >>> dp = cx.vecs.PolarVel(u.Quantity(1, "m/s"), u.Quantity(1, "mas/yr"))
        >>> neg_dp = -dp
        >>> print(neg_dp)
        <PolarVel (d_r[m / s], d_phi[mas / yr])
            [-1 -1]>

        """
        return jax.tree.map(jnp.negative, self)

    # ===============================================================
    # Convenience methods

    @partial(eqx.filter_jit, inline=True)
    def norm(self, position: AbstractPos, /) -> u.Quantity["speed"]:
        """Return the norm of the vector."""
        return self.vconvert(self._cartesian_cls, position).norm()
