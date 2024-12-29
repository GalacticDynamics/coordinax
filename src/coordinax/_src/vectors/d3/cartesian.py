"""Built-in vector classes."""

__all__ = [
    "CartesianAcc3D",
    "CartesianPos3D",
    "CartesianVel3D",
]

from dataclasses import replace
from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from plum import dispatch
from quax import register

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u

import coordinax._src.typing as ct
from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from .generic import CartesianGeneric3D
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty, is_any_quantity
from coordinax._src.vectors.base import AbstractPos
from coordinax._src.vectors.base.mixins import AvalMixin

#####################################################################
# Position


@final
class CartesianPos3D(AbstractPos3D):
    """Cartesian 3D Position.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_(u.Quantity([1, 2, 3], "m"))
    >>> print(vec)
    <CartesianPos3D (x[m], y[m], z[m])
        [1 2 3]>

    """

    x: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    z: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""Z coordinate :math:`z \in (-\infty,+\infty)`."""

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVel3D"]:
        """Return the differential class.

        Examples
        --------
        >>> import coordinax as cx
        >>> print(cx.vecs.CartesianPos3D.differential_cls)
        <class 'coordinax...CartesianVel3D'>

        """
        return CartesianVel3D


# =====================================================
# Primitives


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_cart3d_pos(lhs: CartesianPos3D, rhs: AbstractPos, /) -> CartesianPos3D:
    """Subtract two vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> s = cx.SphericalPos(r=u.Quantity(1, "km"), theta=u.Quantity(90, "deg"),
    ...                     phi=u.Quantity(0, "deg"))
    >>> print(q + s)
    <CartesianPos3D (x[km], y[km], z[km])
        [2. 2. 3.]>

    """
    cart = rhs.vconvert(CartesianPos3D)
    return jax.tree.map(jnp.add, lhs, cart, is_leaf=is_any_quantity)


@register(jax.lax.neg_p)  # type: ignore[misc]
def _neg_p_cart3d_pos(obj: CartesianPos3D, /) -> CartesianPos3D:
    """Negate the `coordinax.CartesianPos3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(-q)
    <CartesianPos3D (x[km], y[km], z[km])
        [-1 -2 -3]>

    """
    return jax.tree.map(qlax.neg, obj)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_cart3d_pos(lhs: CartesianPos3D, rhs: AbstractPos, /) -> CartesianPos3D:
    """Subtract two vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> s = cx.SphericalPos(r=u.Quantity(1, "km"), theta=u.Quantity(90, "deg"),
    ...                     phi=u.Quantity(0, "deg"))
    >>> print(q - s)
    <CartesianPos3D (x[km], y[km], z[km])
        [0. 2. 3.]>

    """
    cart = rhs.vconvert(CartesianPos3D)
    return jax.tree.map(jnp.subtract, lhs, cart)


# =====================================================
# Functions


# from coordinax.vectors.funcs
@dispatch  # type: ignore[misc]
@partial(eqx.filter_jit, inline=True)
def normalize_vector(obj: CartesianPos3D, /) -> CartesianGeneric3D:
    """Return the norm of the vector.

    This has length 1.

    .. note::

        The unit vector is dimensionless, even if the input vector has units.
        This is because the unit vector is a ratio of two quantities: each
        component and the norm of the vector.

    Returns
    -------
    CartesianGeneric3D
        The norm of the vector.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(cx.vecs.normalize_vector(q))
    <CartesianGeneric3D (x[], y[], z[])
        [0.267 0.535 0.802]>

    """
    norm = obj.norm()
    return CartesianGeneric3D(x=obj.x / norm, y=obj.y / norm, z=obj.z / norm)


#####################################################################
# Velocity


@final
class CartesianVel3D(AvalMixin, AbstractVel3D):
    """Cartesian 3D Velocity.

    Examples
    --------
    >>> import coordinax as cx

    >>> vec = cx.CartesianVel3D.from_([1, 2, 3], "m/s")
    >>> print(vec)
    <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
        [1 2 3]>

    """

    d_x: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""X speed :math:`dx/dt \in [-\infty, \infty]."""

    d_y: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Y speed :math:`dy/dt \in [-\infty, \infty]."""

    d_z: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""Z speed :math:`dz/dt \in [-\infty, \infty]."""

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPos3D]:
        return CartesianPos3D

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAcc3D"]:
        """Return the differential class.

        Examples
        --------
        >>> import coordinax as cx
        >>> print(cx.vecs.CartesianVel3D.differential_cls)
        <class 'coordinax...CartesianAcc3D'>

        """
        return CartesianAcc3D

    @partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractPos3D | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> c = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
        >>> c.norm()
        Quantity['speed'](Array(3.7416575, dtype=float32), unit='km / s')

        """
        return jnp.sqrt(self.d_x**2 + self.d_y**2 + self.d_z**2)


# -----------------------------------------------------
# Method dispatches


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_pp(lhs: CartesianVel3D, rhs: CartesianVel3D, /) -> CartesianVel3D:
    """Add two Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> print(q + q)
    <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
        [2 4 6]>

    """
    return jax.tree.map(jnp.add, lhs, rhs)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_v3_v3(lhs: CartesianVel3D, other: CartesianVel3D, /) -> CartesianVel3D:
    """Subtract two differentials.

    Examples
    --------
    >>> from coordinax import CartesianPos3D, CartesianVel3D
    >>> q = CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> print(q - q)
    <CartesianVel3D (d_x[km / s], d_y[km / s], d_z[km / s])
        [0 0 0]>

    """
    return jax.tree.map(jnp.subtract, lhs, other)


#####################################################################
# Acceleration


@final
class CartesianAcc3D(AvalMixin, AbstractAcc3D):
    """Cartesian differential representation."""

    d2_x: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""X acceleration :math:`d^2x/dt^2 \in [-\infty, \infty]."""

    d2_y: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Y acceleration :math:`d^2y/dt^2 \in [-\infty, \infty]."""

    d2_z: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""Z acceleration :math:`d^2z/dt^2 \in [-\infty, \infty]."""

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVel3D]:
        return CartesianVel3D

    # -----------------------------------------------------
    # Methods

    @override
    @partial(jax.jit, inline=True)
    def norm(
        self, _: AbstractVel3D | None = None, __: AbstractPos3D | None = None, /
    ) -> ct.BatchableAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> import coordinax as cx
        >>> c = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
        >>> c.norm()
        Quantity['acceleration'](Array(3.7416575, dtype=float32), unit='km / s2')

        """
        return jnp.sqrt(self.d2_x**2 + self.d2_y**2 + self.d2_z**2)


# -----------------------------------------------------
# Method dispatches


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_aa(lhs: CartesianAcc3D, rhs: CartesianAcc3D, /) -> CartesianAcc3D:
    """Add two Cartesian accelerations.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> print(q + q)
    <CartesianAcc3D (d2_x[km / s2], d2_y[km / s2], d2_z[km / s2])
        [2 4 6]>

    """
    return jax.tree.map(jnp.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_ac3(lhs: ArrayLike, rhs: CartesianPos3D, /) -> CartesianPos3D:
    """Scale a position by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.CartesianPos3D.from_([1, 2, 3], "km")

    >>> print(2 * v)
    <CartesianPos3D (x[km], y[km], z[km])
        [2 4 6]>

    >>> print(jnp.multiply(2, v))
    <CartesianPos3D (x[km], y[km], z[km])
        [2 4 6]>

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x, y=lhs * rhs.y, z=lhs * rhs.z)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_a3_a3(lhs: CartesianAcc3D, rhs: CartesianAcc3D, /) -> CartesianAcc3D:
    """Subtract two accelerations.

    Examples
    --------
    >>> import coordinax as cx

    >>> q = cx.vecs.CartesianAcc3D.from_([1, 2, 3], "km/s2")
    >>> print(q - q)
    <CartesianAcc3D (d2_x[km / s2], d2_y[km / s2], d2_z[km / s2])
        [0 0 0]>

    """
    return jax.tree.map(jnp.subtract, lhs, rhs)
