"""Abstract Accelerations."""

__all__ = ["AbstractAcc"]

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any

import jax

import quaxed.numpy as jnp
import unxt as u

from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel

if TYPE_CHECKING:
    from typing import Self


ACCELERATION_CLASSES: set[type["AbstractAcc"]] = set()


class AbstractAcc(AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of vector differentials in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered.
        """
        ACCELERATION_CLASSES.add(cls)

    @classproperty
    @classmethod
    @abstractmethod
    def _cartesian_cls(cls) -> type["AbstractVector"]:
        """Return the corresponding Cartesian vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.CartesianAcc3D._cartesian_cls
        <class 'coordinax...CartesianAcc3D'>

        >>> cx.vecs.SphericalAcc._cartesian_cls
        <class 'coordinax...CartesianAcc3D'>

        """
        # TODO: something nicer than this for getting the corresponding class
        raise NotImplementedError

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVel]:
        """Return the corresponding vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialAcc.integral_cls.__name__
        'RadialVel'

        >>> cx.vecs.SphericalAcc.integral_cls.__name__
        'SphericalVel'

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

        >>> d2r = cx.vecs.RadialAcc.from_([1], "m/s2")
        >>> -d2r
        RadialAcc( d2_r=Quantity[...](value=i32[], unit=Unit("m / s2")) )

        >>> d2p = cx.vecs.PolarAcc(u.Quantity(1, "m/s2"), u.Quantity(1, "mas/yr2"))
        >>> negd2p = -d2p
        >>> print(negd2p)
        <PolarAcc (d2_r[m / s2], d2_phi[mas / yr2])
            [-1 -1]>

        """
        return jax.tree.map(jnp.negative, self)

    # ===============================================================
    # Convenience methods

    @partial(jax.jit)
    def norm(
        self: "AbstractAcc", velocity: AbstractVel, position: AbstractPos, /
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
        Quantity[...](Array(5..., dtype=float32), unit='m / s2')

        """
        return self.vconvert(self._cartesian_cls, velocity, position).norm()
