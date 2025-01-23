"""Representation of velocities in different systems."""

__all__ = ["AbstractVel", "VELOCITY_CLASSES"]

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx

import unxt as u

from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractVector
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.mixins import AvalMixin

if TYPE_CHECKING:
    import coordinax.vecs


class AbstractVel(AvalMixin, AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of vector differentials in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered.
        """
        VELOCITY_CLASSES_MUTABLE[cls] = None

    # ===============================================================
    # Vector API

    @classproperty
    @classmethod
    @abstractmethod
    def _cartesian_cls(cls) -> "type[AbstractVector]":
        """Return the corresponding Cartesian vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialVel._cartesian_cls
        <class 'coordinax...CartesianVel1D'>

        >>> cx.SphericalVel._cartesian_cls
        <class 'coordinax...CartesianVel3D'>

        """
        raise NotImplementedError  # pragma: no cover

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> "type[AbstractPos]":
        """Return the corresponding vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialVel.integral_cls.__name__
        'RadialPos'

        >>> cx.SphericalVel.integral_cls.__name__
        'SphericalPos'

        """
        raise NotImplementedError  # pragma: no cover

    @classproperty
    @classmethod
    @abstractmethod
    def differential_cls(cls) -> "type[coordinax.vecs.AbstractAcc]":
        """Return the corresponding differential vector class.

        Examples
        --------
        >>> import coordinax as cx

        >>> cx.vecs.RadialVel.differential_cls.__name__
        'RadialAcc'

        >>> cx.SphericalVel.differential_cls.__name__
        'SphericalAcc'

        """
        raise NotImplementedError  # pragma: no cover

    # ===============================================================
    # Convenience methods

    @partial(eqx.filter_jit)
    def norm(self: "AbstractVel", q: AbstractPos, /) -> u.Quantity["speed"]:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx

        >>> q = cx.vecs.CartesianPos2D.from_([1, 1], "km")
        >>> p = cx.vecs.PolarVel(d_r=u.Quantity(1, "km/s"),
        ...                      d_phi=u.Quantity(1, "deg/s"))

        >>> p.norm(q)
        Quantity['speed'](Array(1.0003046, dtype=float32), unit='km / s')

        """
        cart_vel = cast(AbstractVel, self.vconvert(self._cartesian_cls, q))
        return cart_vel.norm(q)  # type: ignore[call-arg,misc]


#: Registered velocity classes.
VELOCITY_CLASSES_MUTABLE: dict[type[AbstractVel], None] = {}
VELOCITY_CLASSES = VELOCITY_CLASSES_MUTABLE.keys()
