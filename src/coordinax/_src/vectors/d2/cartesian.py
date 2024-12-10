"""Built-in vector classes."""

__all__ = [
    "CartesianAcc2D",
    "CartesianPos2D",
    "CartesianVel2D",
]

from dataclasses import fields, replace
from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import ArrayLike, Shaped
from quax import register

import quaxed.numpy as jnp
import unxt as u
from quaxed import lax as qlax
from unxt.quantity import AbstractQuantity, Quantity

import coordinax._src.typing as ct
from .base import AbstractAcc2D, AbstractPos2D, AbstractVel2D
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractPos
from coordinax._src.vectors.base.mixins import AvalMixin


@final
class CartesianPos2D(AbstractPos2D):
    """Cartesian vector representation."""

    x: BatchableLength = eqx.field(
        converter=partial(u.Quantity["length"].from_, dtype=float)
    )
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: BatchableLength = eqx.field(
        converter=partial(u.Quantity["length"].from_, dtype=float)
    )
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVel2D"]:
        return CartesianVel2D


# -----------------------------------------------------


@CartesianPos2D.from_.dispatch  # type: ignore[attr-defined, misc]
def from_(
    cls: type[CartesianPos2D], obj: Shaped[AbstractQuantity, "*batch 2"], /
) -> CartesianPos2D:
    """Construct a 2D Cartesian position.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianPos2D.from_(u.Quantity([1, 2], "m"))
    >>> vec
    CartesianPos2D(
        x=Quantity[...](value=f32[], unit=Unit("m")),
        y=Quantity[...](value=f32[], unit=Unit("m"))
    )

    """
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


# -----------------------------------------------------


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_cart2d_pos(lhs: CartesianPos2D, rhs: AbstractPos, /) -> CartesianPos2D:
    """Add two vectors.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> cart = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> polr = cx.vecs.PolarPos(r=u.Quantity(3, "km"), phi=u.Quantity(90, "deg"))
    >>> (cart + polr).x
    Quantity['length'](Array(0.9999999, dtype=float32), unit='km')

    >>> jnp.add(cart, polr).x
    Quantity['length'](Array(0.9999999, dtype=float32), unit='km')

    """
    cart = rhs.vconvert(CartesianPos2D)
    return jax.tree.map(qlax.add, lhs, cart)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_v_cart2d(lhs: ArrayLike, rhs: CartesianPos2D, /) -> CartesianPos2D:
    """Scale a cartesian 2D position by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianPos2D.from_([3, 4], "m")
    >>> jnp.multiply(5, v).x
    Quantity['length'](Array(15., dtype=float32), unit='m')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x, y=lhs * rhs.y)


@register(jax.lax.neg_p)  # type: ignore[misc]
def _neg_p_cart2d_pos(obj: CartesianPos2D, /) -> CartesianPos2D:
    """Negate the `coordinax.vecs.CartesianPos2D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> (-q).x
    Quantity['length'](Array(-1., dtype=float32), unit='km')

    """
    return jax.tree.map(qlax.neg, obj)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_cart2d_pos2d(lhs: CartesianPos2D, rhs: AbstractPos, /) -> CartesianPos2D:
    """Subtract two vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> cart = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> polr = cx.vecs.PolarPos(r=u.Quantity(3, "km"), phi=u.Quantity(90, "deg"))

    >>> (cart - polr).x
    Quantity['length'](Array(1.0000001, dtype=float32), unit='km')

    """
    cart = rhs.vconvert(CartesianPos2D)
    return jax.tree.map(qlax.sub, lhs, cart)


#####################################################################


@final
class CartesianVel2D(AvalMixin, AbstractVel2D):
    """Cartesian differential representation."""

    d_x: ct.BatchableSpeed = eqx.field(
        converter=partial(u.Quantity["speed"].from_, dtype=float)
    )
    r"""X coordinate differential :math:`\dot{x} \in (-\infty,+\infty)`."""

    d_y: ct.BatchableSpeed = eqx.field(
        converter=partial(u.Quantity["speed"].from_, dtype=float)
    )
    r"""Y coordinate differential :math:`\dot{y} \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPos2D]:
        return CartesianPos2D

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAcc2D"]:
        return CartesianAcc2D


# -----------------------------------------------------


@CartesianVel2D.from_.dispatch  # type: ignore[attr-defined, misc]
def from_(
    cls: type[CartesianVel2D], obj: Shaped[AbstractQuantity, "*batch 2"], /
) -> CartesianVel2D:
    """Construct a 2D Cartesian velocity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianVel2D.from_(u.Quantity([1, 2], "m/s"))
    >>> vec
    CartesianVel2D(
      d_x=Quantity[...]( value=f32[], unit=Unit("m / s") ),
      d_y=Quantity[...]( value=f32[], unit=Unit("m / s") )
    )

    """
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


# -----------------------------------------------------


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_pp(lhs: CartesianVel2D, rhs: CartesianVel2D, /) -> CartesianVel2D:
    """Add two Cartesian velocities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianVel2D.from_([1, 2], "km/s")
    >>> (v + v).d_x
    Quantity['speed'](Array(2., dtype=float32), unit='km / s')

    >>> jnp.add(v, v).d_x
    Quantity['speed'](Array(2., dtype=float32), unit='km / s')

    """
    return jax.tree.map(qlax.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_vp(lhs: ArrayLike, rhts: CartesianVel2D, /) -> CartesianVel2D:
    """Scale a cartesian 2D velocity by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianVel2D.from_([3, 4], "m/s")
    >>> (5 * v).d_x
    Quantity['speed'](Array(15., dtype=float32), unit='m / s')

    >>> jnp.multiply(5, v).d_x
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
class CartesianAcc2D(AvalMixin, AbstractAcc2D):
    """Cartesian acceleration representation."""

    d2_x: ct.BatchableAcc = eqx.field(
        converter=partial(Quantity["acceleration"].from_, dtype=float)
    )
    r"""X coordinate acceleration :math:`\frac{d^2 x}{dt^2} \in (-\infty,+\infty)`."""

    d2_y: ct.BatchableAcc = eqx.field(
        converter=partial(Quantity["acceleration"].from_, dtype=float)
    )
    r"""Y coordinate acceleration :math:`\frac{d^2 y}{dt^2} \in (-\infty,+\infty)`."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVel2D]:
        return CartesianVel2D

    # -----------------------------------------------------

    @override
    @partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractVel2D | None = None, /) -> ct.BatchableAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> v = cx.vecs.CartesianAcc2D.from_([3, 4], "km/s2")
        >>> v.norm()
        Quantity['acceleration'](Array(5., dtype=float32), unit='km / s2')

        """
        return jnp.sqrt(self.d2_x**2 + self.d2_y**2)


# -----------------------------------------------------


@CartesianAcc2D.from_.dispatch  # type: ignore[attr-defined, misc]
def from_(cls: type[CartesianAcc2D], obj: AbstractQuantity, /) -> CartesianAcc2D:
    """Construct a 2D Cartesian velocity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianAcc2D.from_(u.Quantity([1, 2], "m/s2"))
    >>> vec
    CartesianAcc2D(
      d2_x=Quantity[...](value=f32[], unit=Unit("m / s2")),
      d2_y=Quantity[...](value=f32[], unit=Unit("m / s2"))
    )

    """
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


# -----------------------------------------------------


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_aa(lhs: CartesianAcc2D, rhs: CartesianAcc2D, /) -> CartesianAcc2D:
    """Add two Cartesian accelerations.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianAcc2D.from_([3, 4], "km/s2")
    >>> (v + v).d2_x
    Quantity['acceleration'](Array(6., dtype=float32), unit='km / s2')

    >>> jnp.add(v, v).d2_x
    Quantity['acceleration'](Array(6., dtype=float32), unit='km / s2')

    """
    return jax.tree.map(qlax.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_va(lhs: ArrayLike, rhts: CartesianAcc2D, /) -> CartesianAcc2D:
    """Scale a cartesian 2D acceleration by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianAcc2D.from_([3, 4], "m/s2")
    >>> jnp.multiply(5, v).d2_x
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
