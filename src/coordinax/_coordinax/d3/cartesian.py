"""Built-in vector classes."""

__all__ = [
    "CartesianPosition3D",
    "CartesianVelocity3D",
    "CartesianAcceleration3D",
]

from dataclasses import fields, replace
from functools import partial
from typing import final

import equinox as eqx
import jax
from jaxtyping import ArrayLike, Shaped
from quax import register

import quaxed.array_api as xp
import quaxed.lax as qlax
from dataclassish import field_items
from unxt import AbstractQuantity, Quantity

import coordinax._coordinax.typing as ct
from .base import AbstractAcceleration3D, AbstractPosition3D, AbstractVelocity3D
from coordinax._coordinax.base_pos import AbstractPosition
from coordinax._coordinax.mixins import AvalMixin
from coordinax._coordinax.utils import classproperty


@final
class CartesianPosition3D(AbstractPosition3D):
    """Cartesian vector representation."""

    x: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    z: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""Z coordinate :math:`z \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVelocity3D"]:
        return CartesianVelocity3D

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> q = cx.CartesianPosition3D.constructor([1, 2, 3], "kpc")
        >>> (-q).x
        Quantity['length'](Array(-1., dtype=float32), unit='kpc')

        """
        return replace(self, x=-self.x, y=-self.y, z=-self.z)


# -----------------------------------------------------


@CartesianPosition3D.constructor._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: type[CartesianPosition3D], obj: Shaped[AbstractQuantity, "*batch 3"], /
) -> CartesianPosition3D:
    """Construct a 3D Cartesian position.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "m"))
    >>> vec
    CartesianPosition3D(
      x=Quantity[...](value=f32[], unit=Unit("m")),
      y=Quantity[...](value=f32[], unit=Unit("m")),
      z=Quantity[...](value=f32[], unit=Unit("m"))
    )

    """
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


# -----------------------------------------------------
# Method dispatches


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_cart3d_pos(
    lhs: CartesianPosition3D, rhs: AbstractPosition, /
) -> CartesianPosition3D:
    """Subtract two vectors.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> q = cx.CartesianPosition3D.constructor([1, 2, 3], "kpc")
    >>> s = cx.SphericalPosition(r=Quantity(1, "kpc"), theta=Quantity(90, "deg"),
    ...                          phi=Quantity(0, "deg"))
    >>> (q + s).x
    Quantity['length'](Array(2., dtype=float32), unit='kpc')

    """
    cart = rhs.represent_as(CartesianPosition3D)
    return replace(
        lhs, **{k: qlax.add(v, getattr(cart, k)) for k, v in field_items(lhs)}
    )


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_cart3d_pos(
    lhs: CartesianPosition3D, rhs: AbstractPosition, /
) -> CartesianPosition3D:
    """Subtract two vectors.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> q = cx.CartesianPosition3D.constructor([1, 2, 3], "kpc")
    >>> s = cx.SphericalPosition(r=Quantity(1, "kpc"), theta=Quantity(90, "deg"),
    ...                          phi=Quantity(0, "deg"))
    >>> (q - s).x
    Quantity['length'](Array(0., dtype=float32), unit='kpc')

    """
    cart = rhs.represent_as(CartesianPosition3D)
    return jax.tree.map(qlax.sub, lhs, cart)


#####################################################################


@final
class CartesianVelocity3D(AvalMixin, AbstractVelocity3D):
    """Cartesian differential representation."""

    d_x: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""X speed :math:`dx/dt \in [-\infty, \infty]."""

    d_y: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Y speed :math:`dy/dt \in [-\infty, \infty]."""

    d_z: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""Z speed :math:`dz/dt \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPosition3D]:
        return CartesianPosition3D

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAcceleration3D"]:
        return CartesianAcceleration3D

    @partial(jax.jit, inline=True)
    def norm(self, _: AbstractPosition3D | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> c = cx.CartesianVelocity3D.constructor([1, 2, 3], "km/s")
        >>> c.norm()
        Quantity['speed'](Array(3.7416575, dtype=float32), unit='km / s')

        """
        return xp.sqrt(self.d_x**2 + self.d_y**2 + self.d_z**2)


# -----------------------------------------------------


@CartesianVelocity3D.constructor._f.dispatch  # type: ignore[attr-defined,misc]  # noqa: SLF001
def constructor(
    cls: type[CartesianVelocity3D], obj: Shaped[AbstractQuantity, "*batch 3"], /
) -> CartesianVelocity3D:
    """Construct a 3D Cartesian velocity.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianVelocity3D.constructor(Quantity([1, 2, 3], "m/s"))
    >>> vec
    CartesianVelocity3D(
      d_x=Quantity[...]( value=f32[], unit=Unit("m / s") ),
      d_y=Quantity[...]( value=f32[], unit=Unit("m / s") ),
      d_z=Quantity[...]( value=f32[], unit=Unit("m / s") )
    )

    """
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


# -----------------------------------------------------
# Method dispatches


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_pp(
    lhs: CartesianVelocity3D, rhs: CartesianVelocity3D, /
) -> CartesianVelocity3D:
    """Add two Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.CartesianVelocity3D.constructor([1, 2, 3], "km/s")
    >>> q2 = q + q
    >>> q2.d_y
    Quantity['speed'](Array(4., dtype=float32), unit='km / s')

    """
    return jax.tree.map(qlax.add, lhs, rhs)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_v3_v3(
    lhs: CartesianVelocity3D, other: CartesianVelocity3D, /
) -> CartesianVelocity3D:
    """Subtract two differentials.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax import CartesianPosition3D, CartesianVelocity3D
    >>> q = CartesianVelocity3D.constructor(Quantity([1, 2, 3], "km/s"))
    >>> q2 = q - q
    >>> q2.d_y
    Quantity['speed'](Array(0., dtype=float32), unit='km / s')

    """
    return jax.tree.map(qlax.sub, lhs, other)


#####################################################################


@final
class CartesianAcceleration3D(AvalMixin, AbstractAcceleration3D):
    """Cartesian differential representation."""

    d2_x: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""X acceleration :math:`d^2x/dt^2 \in [-\infty, \infty]."""

    d2_y: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""Y acceleration :math:`d^2y/dt^2 \in [-\infty, \infty]."""

    d2_z: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""Z acceleration :math:`d^2z/dt^2 \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVelocity3D]:
        return CartesianVelocity3D

    # -----------------------------------------------------
    # Methods

    @partial(jax.jit, inline=True)
    def norm(self, _: AbstractVelocity3D | None = None, /) -> ct.BatchableAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> c = cx.CartesianAcceleration3D.constructor([1, 2, 3], "km/s2")
        >>> c.norm()
        Quantity['acceleration'](Array(3.7416575, dtype=float32), unit='km / s2')

        """
        return xp.sqrt(self.d2_x**2 + self.d2_y**2 + self.d2_z**2)


# -----------------------------------------------------


@CartesianAcceleration3D.constructor._f.dispatch  # type: ignore[attr-defined, misc]  # noqa: SLF001
def constructor(
    cls: type[CartesianAcceleration3D], obj: Shaped[AbstractQuantity, "*batch 3"], /
) -> CartesianAcceleration3D:
    """Construct a 3D Cartesian acceleration.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> vec = cx.CartesianAcceleration3D.constructor(Quantity([1, 2, 3], "m/s2"))
    >>> vec
    CartesianAcceleration3D(
      d2_x=Quantity[...](value=f32[], unit=Unit("m / s2")),
      d2_y=Quantity[...](value=f32[], unit=Unit("m / s2")),
      d2_z=Quantity[...](value=f32[], unit=Unit("m / s2"))
    )

    """
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


# -----------------------------------------------------
# Method dispatches


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_aa(
    lhs: CartesianAcceleration3D, rhs: CartesianAcceleration3D, /
) -> CartesianAcceleration3D:
    """Add two Cartesian accelerations."""
    return jax.tree.map(qlax.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_ac3(lhs: ArrayLike, rhs: CartesianPosition3D, /) -> CartesianPosition3D:
    """Scale a position by a scalar.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v = cx.CartesianPosition3D.constructor([1, 2, 3], "kpc")
    >>> xp.multiply(2, v).x
    Quantity['length'](Array(2., dtype=float32), unit='kpc')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x, y=lhs * rhs.y, z=lhs * rhs.z)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_a3_a3(
    lhs: CartesianAcceleration3D, rhs: CartesianAcceleration3D, /
) -> CartesianAcceleration3D:
    """Subtract two accelerations."""
    return jax.tree.map(qlax.sub, lhs, rhs)
