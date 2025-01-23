"""Abstract Accelerations."""

__all__ = ["AbstractAcc", "ACCELERATION_CLASSES"]

from abc import abstractmethod
from functools import partial
from typing import Any, cast

import jax

import unxt as u

from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.base_vel import AbstractVel
from coordinax._src.vectors.mixins import AvalMixin


class AbstractAcc(AvalMixin, AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of vector differentials in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered.
        """
        ACCELERATION_CLASSES_MUTABLE[cls] = None

    # ===============================================================
    # Vector API

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
        raise NotImplementedError  # pragma: no cover

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
        raise NotImplementedError  # pragma: no cover

    # ===============================================================
    # Convenience methods

    @partial(jax.jit)
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
        Quantity[...](Array(5..., dtype=float32), unit='m / s2')

        """
        cart_acc = cast(AbstractAcc, self.vconvert(self._cartesian_cls, p, q))
        return cart_acc.norm(p, q)


# -----------------

ACCELERATION_CLASSES_MUTABLE: dict[type[AbstractAcc], None] = {}
ACCELERATION_CLASSES = ACCELERATION_CLASSES_MUTABLE.keys()
