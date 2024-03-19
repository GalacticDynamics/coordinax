"""Representation of coordinates in different systems."""

__all__ = ["AbstractVectorDifferential"]

import warnings
from abc import abstractmethod
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import jax
from plum import dispatch

from unxt import Quantity

from ._base import AbstractVectorBase
from ._base_vec import AbstractVector
from ._utils import classproperty, dataclass_items

if TYPE_CHECKING:
    from typing_extensions import Self

DT = TypeVar("DT", bound="AbstractVectorDifferential")

DIFFERENTIAL_CLASSES: list[type["AbstractVectorDifferential"]] = []


class AbstractVectorDifferential(AbstractVectorBase):  # pylint: disable=abstract-method
    """Abstract representation of vector differentials in different systems."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass.

        The subclass is registered.
        """
        DIFFERENTIAL_CLASSES.append(cls)

    @classproperty
    @classmethod
    @abstractmethod
    def integral_cls(cls) -> type["AbstractVectorDifferential"]:
        """Return the corresponding vector class.

        Examples
        --------
        >>> from coordinax import RadialDifferential, SphericalDifferential

        >>> RadialDifferential.integral_cls
        <class 'coordinax._d1.builtin.RadialVector'>

        >>> SphericalDifferential.integral_cls
        <class 'coordinax._d3.builtin.SphericalVector'>

        """
        raise NotImplementedError

    # ===============================================================
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import RadialDifferential
        >>> dr = RadialDifferential(Quantity(1, "m/s"))
        >>> -dr
        RadialDifferential( d_r=Quantity[...]( value=f32[], unit=Unit("m / s") ) )

        >>> from coordinax import PolarDifferential
        >>> dp = PolarDifferential(Quantity(1, "m/s"), Quantity(1, "mas/yr"))
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
    def __mul__(
        self: "AbstractVectorDifferential", other: Quantity
    ) -> "AbstractVector":
        """Multiply the vector by a :class:`unxt.Quantity`.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import RadialDifferential

        >>> dr = RadialDifferential(Quantity(1, "m/s"))
        >>> vec = dr * Quantity(2, "s")
        >>> vec
        RadialVector(r=Quantity[PhysicalType('length')](value=f32[], unit=Unit("m")))
        >>> vec.r
        Quantity['length'](Array(2., dtype=float32), unit='m')

        """
        return self.integral_cls.constructor(
            {k[2:]: v * other for k, v in dataclass_items(self)}
        )

    # ===============================================================
    # Convenience methods

    @partial(jax.jit, static_argnums=1)
    def represent_as(
        self, target: type[DT], position: AbstractVector, /, *args: Any, **kwargs: Any
    ) -> DT:
        """Represent the vector as another type."""
        if any(args):
            warnings.warn("Extra arguments are ignored.", UserWarning, stacklevel=2)

        from ._transform import represent_as  # pylint: disable=import-outside-toplevel

        return represent_as(self, target, position, **kwargs)

    @partial(jax.jit)
    def norm(self, position: AbstractVector, /) -> Quantity["speed"]:
        """Return the norm of the vector."""
        return self.represent_as(self._cartesian_cls, position).norm()
