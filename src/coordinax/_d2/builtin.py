"""Built-in vector classes."""

__all__ = [
    # Position
    "Cartesian2DVector",
    "PolarVector",
    # Differential
    "CartesianDifferential2D",
    "PolarDifferential",
]

from dataclasses import replace
from functools import partial
from typing import Any, final

import equinox as eqx
import jax

import quaxed.array_api as xp
from unxt import AbstractDistance, Distance, Quantity

import coordinax._typing as ct
from .base import Abstract2DVector, Abstract2DVectorDifferential
from coordinax._base_pos import AbstractPosition
from coordinax._checks import check_azimuth_range, check_r_non_negative
from coordinax._converters import converter_azimuth_to_range
from coordinax._utils import classproperty

# =============================================================================
# 2D


@final
class Cartesian2DVector(Abstract2DVector):
    """Cartesian vector representation."""

    x: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianDifferential2D"]:
        return CartesianDifferential2D

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian2DVector

        >>> q = Cartesian2DVector.constructor(Quantity([1, 2], "kpc"))
        >>> (-q).x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

        """
        return replace(self, x=-self.x, y=-self.y)

    # -----------------------------------------------------
    # Binary operations

    def __add__(self, other: Any, /) -> "Cartesian2DVector":
        """Add two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian2DVector, PolarVector
        >>> cart = Cartesian2DVector.constructor(Quantity([1, 2], "kpc"))
        >>> polr = PolarVector(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))

        >>> (cart + polr).x
        Quantity['length'](Array(0.9999999, dtype=float32), unit='kpc')

        """
        if not isinstance(other, AbstractPosition):
            msg = f"Cannot add {Cartesian2DVector!r} and {type(other)!r}."
            raise TypeError(msg)

        cart = other.represent_as(Cartesian2DVector)
        return replace(self, x=self.x + cart.x, y=self.y + cart.y)

    def __sub__(self, other: Any, /) -> "Cartesian2DVector":
        """Subtract two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian2DVector, PolarVector
        >>> cart = Cartesian2DVector.constructor(Quantity([1, 2], "kpc"))
        >>> polr = PolarVector(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))

        >>> (cart - polr).x
        Quantity['length'](Array(1.0000001, dtype=float32), unit='kpc')

        """
        if not isinstance(other, AbstractPosition):
            msg = f"Cannot subtract {Cartesian2DVector!r} and {type(other)!r}."
            raise TypeError(msg)

        cart = other.represent_as(Cartesian2DVector)
        return replace(self, x=self.x - cart.x, y=self.y - cart.y)

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian2DVector
        >>> q = Cartesian2DVector.constructor(Quantity([3, 4], "kpc"))
        >>> q.norm()
        Quantity['length'](Array(5., dtype=float32), unit='kpc')

        """
        return xp.sqrt(self.x**2 + self.y**2)


@final
class PolarVector(Abstract2DVector):
    r"""Polar vector representation.

    Parameters
    ----------
    r : BatchableDistance
        Radial distance :math:`r \in [0,+\infty)`.
    phi : BatchableAngle
        Polar angle :math:`\phi \in [0,2\pi)`.  We use the symbol `phi` to
        adhere to the ISO standard 31-11.

    """

    r: ct.BatchableDistance = eqx.field(
        converter=lambda x: x
        if isinstance(x, AbstractDistance)
        else Distance.constructor(x, dtype=float)
    )
    r"""Radial distance :math:`r \in [0,+\infty)`."""

    phi: ct.BatchableAngle = eqx.field(
        converter=lambda x: converter_azimuth_to_range(
            Quantity["angle"].constructor(x, dtype=float)  # pylint: disable=E1120
        )
    )
    r"""Polar angle :math:`\phi \in [0,2\pi)`."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        check_r_non_negative(self.r)
        check_azimuth_range(self.phi)

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["PolarDifferential"]:
        return PolarDifferential

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> q = cx.PolarVector(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))
        >>> q.norm()
        Distance(Array(3., dtype=float32), unit='kpc')

        """
        return self.r


##############################################################################


@final
class CartesianDifferential2D(Abstract2DVectorDifferential):
    """Cartesian differential representation."""

    d_x: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""X coordinate differential :math:`\dot{x} \in (-\infty,+\infty)`."""

    d_y: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Y coordinate differential :math:`\dot{y} \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[Cartesian2DVector]:
        return Cartesian2DVector

    @partial(jax.jit)
    def norm(self, _: Abstract2DVector | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianDifferential2D
        >>> v = CartesianDifferential2D.constructor(Quantity([3, 4], "km/s"))
        >>> v.norm()
        Quantity['speed'](Array(5., dtype=float32), unit='km / s')

        """
        return xp.sqrt(self.d_x**2 + self.d_y**2)


@final
class PolarDifferential(Abstract2DVectorDifferential):
    """Polar differential representation."""

    d_r: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Radial speed :math:`dr/dt \in [-\infty,+\infty]`."""

    d_phi: ct.BatchableAngularSpeed = eqx.field(
        converter=partial(Quantity["angular speed"].constructor, dtype=float)
    )
    r"""Polar angular speed :math:`d\phi/dt \in [-\infty,+\infty]`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[PolarVector]:
        return PolarVector
