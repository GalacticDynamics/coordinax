"""Built-in vector classes."""

__all__ = [
    "CartesianAcc3D",
    "CartesianPos3D",
    "CartesianVel3D",
]

from dataclasses import fields, replace
from functools import partial
from typing import final
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import ArrayLike, Shaped
from plum import dispatch
from quax import register

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from unxt.quantity import AbstractQuantity

import coordinax._src.typing as ct
from .base import AbstractAcc3D, AbstractPos3D, AbstractVel3D
from .generic import CartesianGeneric3D
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractPos
from coordinax._src.vectors.base.mixins import AvalMixin

#####################################################################
# Position


@final
class CartesianPos3D(AbstractPos3D):
    """Cartesian vector representation."""

    x: BatchableLength = eqx.field(
        converter=partial(u.Quantity["length"].from_, dtype=float)
    )
    r"""X coordinate :math:`x \in (-\infty,+\infty)`."""

    y: BatchableLength = eqx.field(
        converter=partial(u.Quantity["length"].from_, dtype=float)
    )
    r"""Y coordinate :math:`y \in (-\infty,+\infty)`."""

    z: BatchableLength = eqx.field(
        converter=partial(u.Quantity["length"].from_, dtype=float)
    )
    r"""Z coordinate :math:`z \in (-\infty,+\infty)`."""

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVel3D"]:
        """Return the differential of the class."""
        return CartesianVel3D


# =====================================================
# Constructors


@CartesianPos3D.from_.dispatch  # type: ignore[attr-defined, misc]
def from_(
    cls: type[CartesianPos3D],
    obj: AbstractQuantity,  # TODO: Shaped[AbstractQuantity, "*batch 3"]
    /,
) -> CartesianPos3D:
    """Construct a 3D Cartesian position.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CartesianPos3D.from_(u.Quantity([1, 2, 3], "m"))
    >>> vec
    CartesianPos3D(
      x=Quantity[...](value=f32[], unit=Unit("m")),
      y=Quantity[...](value=f32[], unit=Unit("m")),
      z=Quantity[...](value=f32[], unit=Unit("m"))
    )

    """
    comps = {f.name: obj[..., i] for i, f in enumerate(fields(cls))}
    return cls(**comps)


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
    >>> (q + s).x
    Quantity['length'](Array(2., dtype=float32), unit='km')

    """
    cart = rhs.vconvert(CartesianPos3D)
    return replace(
        lhs, **{k: qlax.add(v, getattr(cart, k)) for k, v in field_items(lhs)}
    )


@register(jax.lax.neg_p)  # type: ignore[misc]
def _neg_p_cart3d_pos(obj: CartesianPos3D, /) -> CartesianPos3D:
    """Negate the `coordinax.CartesianPos3D`.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> (-q).x
    Quantity['length'](Array(-1., dtype=float32), unit='km')

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
    >>> (q - s).x
    Quantity['length'](Array(0., dtype=float32), unit='km')

    """
    cart = rhs.vconvert(CartesianPos3D)
    return jax.tree.map(qlax.sub, lhs, cart)


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
    >>> cx.vecs.normalize_vector(q)
    CartesianGeneric3D(
      x=Quantity[...]( value=f32[], unit=Unit(dimensionless) ),
      y=Quantity[...]( value=f32[], unit=Unit(dimensionless) ),
      z=Quantity[...]( value=f32[], unit=Unit(dimensionless) )
    )

    """
    norm = obj.norm()
    return CartesianGeneric3D(x=obj.x / norm, y=obj.y / norm, z=obj.z / norm)


#####################################################################
# Velocity


@final
class CartesianVel3D(AvalMixin, AbstractVel3D):
    """Cartesian differential representation."""

    d_x: ct.BatchableSpeed = eqx.field(
        converter=partial(u.Quantity["speed"].from_, dtype=float)
    )
    r"""X speed :math:`dx/dt \in [-\infty, \infty]."""

    d_y: ct.BatchableSpeed = eqx.field(
        converter=partial(u.Quantity["speed"].from_, dtype=float)
    )
    r"""Y speed :math:`dy/dt \in [-\infty, \infty]."""

    d_z: ct.BatchableSpeed = eqx.field(
        converter=partial(u.Quantity["speed"].from_, dtype=float)
    )
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


@CartesianVel3D.from_.dispatch  # type: ignore[attr-defined,misc]
def from_(
    cls: type[CartesianVel3D],
    obj: AbstractQuantity,  # TODO: Shaped[AbstractQuantity, "*batch 3"]
    /,
) -> CartesianVel3D:
    """Construct a 3D Cartesian velocity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.CartesianVel3D.from_(u.Quantity([1, 2, 3], "m/s"))
    >>> vec
    CartesianVel3D(
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
def _add_pp(lhs: CartesianVel3D, rhs: CartesianVel3D, /) -> CartesianVel3D:
    """Add two Cartesian velocities.

    Examples
    --------
    >>> import coordinax as cx
    >>> q = cx.CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> q2 = q + q
    >>> q2.d_y
    Quantity['speed'](Array(4., dtype=float32), unit='km / s')

    """
    return jax.tree.map(qlax.add, lhs, rhs)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_v3_v3(lhs: CartesianVel3D, other: CartesianVel3D, /) -> CartesianVel3D:
    """Subtract two differentials.

    Examples
    --------
    >>> from coordinax import CartesianPos3D, CartesianVel3D
    >>> q = CartesianVel3D.from_([1, 2, 3], "km/s")
    >>> q2 = q - q
    >>> q2.d_y
    Quantity['speed'](Array(0., dtype=float32), unit='km / s')

    """
    return jax.tree.map(qlax.sub, lhs, other)


#####################################################################
# Acceleration


@final
class CartesianAcc3D(AvalMixin, AbstractAcc3D):
    """Cartesian differential representation."""

    d2_x: ct.BatchableAcc = eqx.field(
        converter=partial(u.Quantity["acceleration"].from_, dtype=float)
    )
    r"""X acceleration :math:`d^2x/dt^2 \in [-\infty, \infty]."""

    d2_y: ct.BatchableAcc = eqx.field(
        converter=partial(u.Quantity["acceleration"].from_, dtype=float)
    )
    r"""Y acceleration :math:`d^2y/dt^2 \in [-\infty, \infty]."""

    d2_z: ct.BatchableAcc = eqx.field(
        converter=partial(u.Quantity["acceleration"].from_, dtype=float)
    )
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


@CartesianAcc3D.from_.dispatch  # type: ignore[attr-defined, misc]
def from_(
    cls: type[CartesianAcc3D], obj: Shaped[AbstractQuantity, "*batch 3"], /
) -> CartesianAcc3D:
    """Construct a 3D Cartesian acceleration.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianAcc3D.from_(u.Quantity([1, 2, 3], "m/s2"))
    >>> vec
    CartesianAcc3D(
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
def _add_aa(lhs: CartesianAcc3D, rhs: CartesianAcc3D, /) -> CartesianAcc3D:
    """Add two Cartesian accelerations."""
    return jax.tree.map(qlax.add, lhs, rhs)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_ac3(lhs: ArrayLike, rhs: CartesianPos3D, /) -> CartesianPos3D:
    """Scale a position by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import coordinax as cx

    >>> v = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> jnp.multiply(2, v).x
    Quantity['length'](Array(2., dtype=float32), unit='km')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, x=lhs * rhs.x, y=lhs * rhs.y, z=lhs * rhs.z)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_a3_a3(lhs: CartesianAcc3D, rhs: CartesianAcc3D, /) -> CartesianAcc3D:
    """Subtract two accelerations."""
    return jax.tree.map(qlax.sub, lhs, rhs)
