"""Representation of coordinates in different systems."""

__all__ = ["AbstractAcceleration"]

import warnings
from abc import abstractmethod
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import jax
from plum import dispatch

from unxt import Quantity

from ._base import AbstractVector
from ._base_pos import AbstractPosition
from ._base_vel import AbstractVelocity
from ._utils import classproperty, dataclass_items

if TYPE_CHECKING:
    from typing_extensions import Self

DT = TypeVar("DT", bound="AbstractAcceleration")

ACCELERATION_CLASSES: set[type["AbstractAcceleration"]] = set()


class AbstractAcceleration(AbstractVector):  # pylint: disable=abstract-method
    """Abstract representation of vector differentials in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered.
        """
        ACCELERATION_CLASSES.add(cls)

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type[AbstractVelocity]:
        """Return the corresponding vector class.

        Examples
        --------
        >>> from coordinax import RadialAcceleration, SphericalAcceleration

        >>> RadialAcceleration.integral_cls.__name__
        'RadialVelocity'

        >>> SphericalAcceleration.integral_cls.__name__
        'SphericalVelocity'

        """
        raise NotImplementedError

    # ===============================================================
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import RadialVelocity
        >>> dr = RadialVelocity(Quantity(1, "m/s"))
        >>> -dr
        RadialVelocity( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )

        >>> from coordinax import PolarVelocity
        >>> dp = PolarVelocity(Quantity(1, "m/s"), Quantity(1, "mas/yr"))
        >>> neg_dp = -dp
        >>> neg_dp.d_r
        Quantity['speed'](Array(-1., dtype=float32), unit='m / s')
        >>> neg_dp.d_phi
        Quantity['angular frequency'](Array(-1., dtype=float32), unit='mas / yr')

        """
        return replace(self, **{k: -v for k, v in dataclass_items(self)})

    # ===============================================================
    # Binary operations

    @dispatch  # type: ignore[misc]
    def __mul__(self: "AbstractAcceleration", other: Quantity) -> "AbstractVector":
        """Multiply the vector by a :class:`unxt.Quantity`.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import RadialAcceleration

        >>> d2r = RadialAcceleration(Quantity(1, "m/s2"))
        >>> vec = d2r * Quantity(2, "s")
        >>> vec
        RadialVelocity( d_r=Quantity[...]( value=i32[], unit=Unit("m / s") ) )
        >>> vec.d_r
        Quantity['speed'](Array(2, dtype=int32, ...), unit='m / s')

        """
        # TODO: better access to corresponding fields
        return self.integral_cls.constructor(
            {k.replace("2", ""): v * other for k, v in dataclass_items(self)}
        )

    # ===============================================================
    # Convenience methods

    @partial(jax.jit, static_argnums=1)
    def represent_as(
        self,
        target: type[DT],
        velocity: AbstractVelocity,
        position: AbstractPosition,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> DT:
        """Represent the vector as another type."""
        if any(args):
            warnings.warn("Extra arguments are ignored.", UserWarning, stacklevel=2)

        from ._transform import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, velocity, position, **kwargs)

    @partial(jax.jit)
    def norm(
        self, velocity: AbstractVelocity, position: AbstractPosition, /
    ) -> Quantity["speed"]:
        """Return the norm of the vector."""
        return self.represent_as(self._cartesian_cls, velocity, position).norm()
