"""Built-in vector classes."""

__all__ = [
    # Position
    "CartesianPosition2D",
    "PolarPosition",
    # Differential
    "CartesianVelocity2D",
    "PolarVelocity",
]

from dataclasses import replace
from functools import partial
from typing import final

import equinox as eqx
import jax

import quaxed.array_api as xp
from unxt import AbstractDistance, Distance, Quantity

import coordinax._typing as ct
from .base import AbstractPosition2D, AbstractVelocity2D
from coordinax._base import AbstractVector
from coordinax._base_pos import AbstractPosition
from coordinax._checks import check_azimuth_range, check_r_non_negative
from coordinax._converters import converter_azimuth_to_range
from coordinax._utils import classproperty

# =============================================================================
# 2D


@final
class CartesianPosition2D(AbstractPosition2D):
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
    def differential_cls(cls) -> type["CartesianVelocity2D"]:
        return CartesianVelocity2D

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D

        >>> q = CartesianPosition2D.constructor(Quantity([1, 2], "kpc"))
        >>> (-q).x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

        """
        return replace(self, x=-self.x, y=-self.y)

    # -----------------------------------------------------
    # Binary operations

    @AbstractVector.__add__.dispatch  # type: ignore[misc]
    def __add__(
        self: "CartesianPosition2D", other: AbstractPosition, /
    ) -> "CartesianPosition2D":
        """Add two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D, PolarPosition
        >>> cart = CartesianPosition2D.constructor(Quantity([1, 2], "kpc"))
        >>> polr = PolarPosition(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))

        >>> (cart + polr).x
        Quantity['length'](Array(0.9999999, dtype=float32), unit='kpc')

        """
        cart = other.represent_as(CartesianPosition2D)
        return replace(self, x=self.x + cart.x, y=self.y + cart.y)

    @AbstractVector.__sub__.dispatch  # type: ignore[misc]
    def __sub__(
        self: "CartesianPosition2D", other: AbstractPosition, /
    ) -> "CartesianPosition2D":
        """Subtract two vectors.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D, PolarPosition
        >>> cart = CartesianPosition2D.constructor(Quantity([1, 2], "kpc"))
        >>> polr = PolarPosition(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))

        >>> (cart - polr).x
        Quantity['length'](Array(1.0000001, dtype=float32), unit='kpc')

        """
        cart = other.represent_as(CartesianPosition2D)
        return replace(self, x=self.x - cart.x, y=self.y - cart.y)

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianPosition2D
        >>> q = CartesianPosition2D.constructor(Quantity([3, 4], "kpc"))
        >>> q.norm()
        Quantity['length'](Array(5., dtype=float32), unit='kpc')

        """
        return xp.sqrt(self.x**2 + self.y**2)


@final
class PolarPosition(AbstractPosition2D):
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
    def differential_cls(cls) -> type["PolarVelocity"]:
        return PolarVelocity

    @partial(jax.jit)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> q = cx.PolarPosition(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))
        >>> q.norm()
        Distance(Array(3., dtype=float32), unit='kpc')

        """
        return self.r


##############################################################################


@final
class CartesianVelocity2D(AbstractVelocity2D):
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
    def integral_cls(cls) -> type[CartesianPosition2D]:
        return CartesianPosition2D

    @partial(jax.jit)
    def norm(self, _: AbstractPosition2D | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import CartesianVelocity2D
        >>> v = CartesianVelocity2D.constructor(Quantity([3, 4], "km/s"))
        >>> v.norm()
        Quantity['speed'](Array(5., dtype=float32), unit='km / s')

        """
        return xp.sqrt(self.d_x**2 + self.d_y**2)


@final
class PolarVelocity(AbstractVelocity2D):
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
    def integral_cls(cls) -> type[PolarPosition]:
        return PolarPosition
