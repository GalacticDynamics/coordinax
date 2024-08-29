"""Built-in vector classes."""

__all__ = [
    "CartesianPosition2D",
    "CartesianVelocity2D",
    "CartesianAcceleration2D",
]

from dataclasses import fields, replace
from functools import partial
from typing import final

import equinox as eqx
import jax
from jaxtyping import ArrayLike, Shaped
from quax import register

import quaxed.array_api as xp
from quaxed import lax as qlax
from unxt import AbstractQuantity, Quantity

import coordinax._coordinax.typing as ct
from .base import AbstractAcceleration2D, AbstractPosition2D, AbstractVelocity2D
from coordinax._coordinax.base_pos import AbstractPosition
from coordinax._coordinax.mixins import AvalMixin
from coordinax._coordinax.utils import classproperty


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
        >>> import coordinax as cx

        >>> q = cx.CartesianPosition2D.constructor([1, 2], "kpc")
        >>> (-q).x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

        """
        return replace(self, x=-self.x, y=-self.y)


# -----------------------------------------------------


@CartesianPosition2D.constructor._f.dispatch  # type: ignore[attr-defined, misc] # noqa: SLF001
def constructor(
    cls: type[CartesianPosition2D], obj: Shaped[AbstractQuantity, "*batch 2"], /
) -> CartesianPosition2D:
    """Construct a 2D Cartesian position.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPosition2D.constructor(Quantity([1, 2], "m"))
    >>> vec
    CartesianPosition2D(
        x=Quantity[...](value=f32[], unit=Unit("m")),
        y=Quantity[...](value=f32[], unit=Unit("m"))
    )

    """
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


# -----------------------------------------------------


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_cart2d_pos(
    lhs: CartesianPosition2D, rhs: AbstractPosition, /
) -> CartesianPosition2D:
    """Add two vectors.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> cart = cx.CartesianPosition2D.constructor(Quantity([1, 2], "kpc"))
    >>> polr = cx.PolarPosition(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))
    >>> (cart + polr).x
    Quantity['length'](Array(0.9999999, dtype=float32), unit='kpc')

    >>> xp.add(cart, polr).x
    Quantity['length'](Array(0.9999999, dtype=float32), unit='kpc')

    """
    cart = rhs.represent_as(CartesianPosition2D)
    return jax.tree.map(qlax.add, lhs, cart)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_v_cart2d(lhs: ArrayLike, rhs: CartesianPosition2D, /) -> CartesianPosition2D:
    """Scale a cartesian 2D position by a scalar.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v = cx.CartesianPosition2D.constructor(Quantity([3, 4], "m"))
    >>> xp.multiply(5, v).x
    Quantity['length'](Array(15., dtype=float32), unit='m')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x, y=lhs * rhs.y)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_cart2d_pos2d(
    lhs: CartesianPosition2D, rhs: AbstractPosition, /
) -> CartesianPosition2D:
    """Subtract two vectors.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> cart = cx.CartesianPosition2D.constructor([1, 2], "kpc")
    >>> polr = cx.PolarPosition(r=Quantity(3, "kpc"), phi=Quantity(90, "deg"))

    >>> (cart - polr).x
    Quantity['length'](Array(1.0000001, dtype=float32), unit='kpc')

    """
    cart = rhs.represent_as(CartesianPosition2D)
    return jax.tree.map(qlax.sub, lhs, cart)


#####################################################################


@final
class CartesianVelocity2D(AvalMixin, AbstractVelocity2D):
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

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAcceleration2D"]:
        return CartesianAcceleration2D


# -----------------------------------------------------


@CartesianVelocity2D.constructor._f.dispatch  # type: ignore[attr-defined, misc] # noqa: SLF001
def constructor(
    cls: type[CartesianVelocity2D], obj: Shaped[AbstractQuantity, "*batch 2"], /
) -> CartesianVelocity2D:
    """Construct a 2D Cartesian velocity.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianVelocity2D.constructor(Quantity([1, 2], "m/s"))
    >>> vec
    CartesianVelocity2D(
      d_x=Quantity[...]( value=f32[], unit=Unit("m / s") ),
      d_y=Quantity[...]( value=f32[], unit=Unit("m / s") )
    )

    """
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


# -----------------------------------------------------


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_pp(
    lhs: CartesianVelocity2D, rhs: CartesianVelocity2D, /
) -> CartesianVelocity2D:
    """Add two Cartesian velocities.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v = cx.CartesianVelocity2D.constructor(Quantity([1, 2], "km/s"))
    >>> (v + v).d_x
    Quantity['speed'](Array(2., dtype=float32), unit='km / s')

    >>> xp.add(v, v).d_x
    Quantity['speed'](Array(2., dtype=float32), unit='km / s')

    """
    return jax.tree.map(qlax.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_vp(lhs: ArrayLike, rhts: CartesianVelocity2D, /) -> CartesianVelocity2D:
    """Scale a cartesian 2D velocity by a scalar.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v = cx.CartesianVelocity2D.constructor(Quantity([3, 4], "m/s"))
    >>> (5 * v).d_x
    Quantity['speed'](Array(15., dtype=float32), unit='m / s')

    >>> xp.multiply(5, v).d_x
    Quantity['speed'](Array(15., dtype=float32), unit='m / s')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhts, d_x=lhs * rhts.d_x, d_y=lhs * rhts.d_y)


#####################################################################


@final
class CartesianAcceleration2D(AvalMixin, AbstractAcceleration2D):
    """Cartesian acceleration representation."""

    d2_x: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""X coordinate acceleration :math:`\frac{d^2 x}{dt^2} \in (-\infty,+\infty)`."""

    d2_y: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""Y coordinate acceleration :math:`\frac{d^2 y}{dt^2} \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVelocity2D]:
        return CartesianVelocity2D

    # -----------------------------------------------------

    @partial(jax.jit, inline=True)
    def norm(self, _: AbstractVelocity2D | None = None, /) -> ct.BatchableAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> v = cx.CartesianAcceleration2D.constructor([3, 4], "km/s2")
        >>> v.norm()
        Quantity['acceleration'](Array(5., dtype=float32), unit='km / s2')

        """
        return xp.sqrt(self.d2_x**2 + self.d2_y**2)


# -----------------------------------------------------


@CartesianAcceleration2D.constructor._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: type[CartesianAcceleration2D],
    obj: Shaped[AbstractQuantity, "*batch 2"],
    /,
) -> CartesianAcceleration2D:
    """Construct a 2D Cartesian velocity.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianAcceleration2D.constructor(Quantity([1, 2], "m/s2"))
    >>> vec
    CartesianAcceleration2D(
      d2_x=Quantity[...](value=f32[], unit=Unit("m / s2")),
      d2_y=Quantity[...](value=f32[], unit=Unit("m / s2"))
    )

    """
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


# -----------------------------------------------------


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_aa(
    lhs: CartesianAcceleration2D, rhs: CartesianAcceleration2D, /
) -> CartesianAcceleration2D:
    """Add two Cartesian accelerations.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v = cx.CartesianAcceleration2D.constructor(Quantity([3, 4], "km/s2"))
    >>> (v + v).d2_x
    Quantity['acceleration'](Array(6., dtype=float32), unit='km / s2')

    >>> xp.add(v, v).d2_x
    Quantity['acceleration'](Array(6., dtype=float32), unit='km / s2')

    """
    return jax.tree.map(qlax.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_va(
    lhs: ArrayLike, rhts: CartesianAcceleration2D, /
) -> CartesianAcceleration2D:
    """Scale a cartesian 2D acceleration by a scalar.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v = cx.CartesianAcceleration2D.constructor(Quantity([3, 4], "m/s2"))
    >>> xp.multiply(5, v).d2_x
    Quantity['acceleration'](Array(15., dtype=float32), unit='m / s2')

    >>> (5 * v).d2_x
    Quantity['acceleration'](Array(15., dtype=float32), unit='m / s2')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhts, d2_x=lhs * rhts.d2_x, d2_y=lhs * rhts.d2_y)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_cart2d_pos2d(
    self: CartesianPosition2D, other: AbstractPosition, /
) -> CartesianPosition2D:
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
    return jax.tree.map(qlax.sub, self, cart)
