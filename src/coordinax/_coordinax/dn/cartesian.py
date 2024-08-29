"""Built-in vector classes."""

__all__ = ["CartesianPositionND", "CartesianVelocityND", "CartesianAccelerationND"]

from dataclasses import replace
from functools import partial
from typing import NoReturn, final
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import ArrayLike, Shaped
from plum import conversion_method
from quax import register

import quaxed.array_api as xp
import quaxed.numpy as jnp
from unxt import Quantity

import coordinax._coordinax.typing as ct
from .base import AbstractAccelerationND, AbstractPositionND, AbstractVelocityND
from coordinax._coordinax.base_pos import AbstractPosition
from coordinax._coordinax.mixins import AvalMixin
from coordinax._coordinax.utils import classproperty

##############################################################################
# Position


@final
class CartesianPositionND(AbstractPositionND):
    """N-dimensional Cartesian vector representation.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt import Quantity

    A 1D vector:

    >>> q = cx.CartesianPositionND(Quantity([[1]], "kpc"))
    >>> q.q
    Quantity['length'](Array([[1.]], dtype=float32), unit='kpc')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = cx.CartesianPositionND(Quantity([1, 2], "kpc"))
    >>> q.q
    Quantity['length'](Array([1., 2.], dtype=float32), unit='kpc')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = cx.CartesianPositionND(Quantity([1, 2, 3], "kpc"))
    >>> q.q
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='kpc')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = cx.CartesianPositionND(Quantity([1, 2, 3, 4], "kpc"))
    >>> q.q
    Quantity['length'](Array([1., 2., 3., 4.], dtype=float32), unit='kpc')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = cx.CartesianPositionND(Quantity([1, 2, 3, 4, 5], "kpc"))
    >>> q.q
    Quantity['length'](Array([1., 2., 3., 4., 5.], dtype=float32), unit='kpc')
    >>> q.shape
    ()

    """

    q: ct.BatchableLength = eqx.field(
        converter=partial(Quantity["length"].constructor, dtype=float)
    )
    r"""N-D coordinate :math:`\vec{x} \in (-\infty,+\infty)`.

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @classproperty
    @classmethod
    @override
    def differential_cls(cls) -> type["CartesianVelocityND"]:  # type: ignore[override]
        return CartesianVelocityND

    # -----------------------------------------------------
    # Unary operations

    def __neg__(self) -> "Self":
        """Negate the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        A 3D vector:

        >>> vec = cx.CartesianPositionND(Quantity([1, 2, 3], "kpc"))
        >>> (-vec).q
        Quantity['length'](Array([-1., -2., -3.], dtype=float32), unit='kpc')

        """
        return replace(self, q=-self.q)

    # -----------------------------------------------------

    @partial(jax.jit, inline=True)
    def norm(self) -> ct.BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        A 3D vector:

        >>> q = cx.CartesianPositionND(Quantity([1, 2, 3], "kpc"))
        >>> q.norm()
        Quantity['length'](Array(3.7416575, dtype=float32), unit='kpc')

        """
        return xp.linalg.vector_norm(self.q, axis=-1)


# -------------------------------------------------------------------


# TODO: move to the class in py3.11+
@CartesianPositionND.constructor._f.dispatch  # type: ignore[attr-defined,misc]  # noqa: SLF001
def constructor(
    cls: type[CartesianPositionND],
    x: Shaped[Quantity["length"], ""] | Shaped[Quantity["length"], "*batch N"],
    /,
) -> CartesianPositionND:
    """Construct an N-dimensional position.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    1D vector:

    >>> cx.CartesianPositionND.constructor(Quantity(1, "kpc"))
    CartesianPositionND(
      q=Quantity[...](value=f32[1], unit=Unit("kpc"))
    )

    >>> cx.CartesianPositionND.constructor(Quantity([1], "kpc"))
    CartesianPositionND(
      q=Quantity[...](value=f32[1], unit=Unit("kpc"))
    )

    2D vector:

    >>> cx.CartesianPositionND.constructor(Quantity([1, 2], "kpc"))
    CartesianPositionND(
      q=Quantity[...](value=f32[2], unit=Unit("kpc"))
    )

    3D vector:

    >>> cx.CartesianPositionND.constructor(Quantity([1, 2, 3], "kpc"))
    CartesianPositionND(
      q=Quantity[...](value=f32[3], unit=Unit("kpc"))
    )

    4D vector:

    >>> cx.CartesianPositionND.constructor(Quantity([1, 2, 3, 4], "kpc"))
    CartesianPositionND(
      q=Quantity[...](value=f32[4], unit=Unit("kpc"))
    )

    """
    return cls(jnp.atleast_1d(x))


@conversion_method(CartesianPositionND, Quantity)  # type: ignore[misc]
def _vec_to_q(obj: CartesianPositionND, /) -> Shaped[Quantity["length"], "*batch N"]:
    """`coordinax.AbstractPosition3D` -> `unxt.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from plum import convert
    >>> from unxt import Quantity

    >>> vec = cx.CartesianPositionND(Quantity([1, 2, 3, 4, 5], unit="kpc"))
    >>> convert(vec, Quantity)
    Quantity['length'](Array([1., 2., 3., 4., 5.], dtype=float32), unit='kpc')

    """
    return obj.q


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_vcnd(
    lhs: CartesianPositionND, rhs: AbstractPosition, /
) -> CartesianPositionND:
    """Add two vectors.

    Examples
    --------
    >>> import coordinax as cx

    A 3D vector:

    >>> q1 = cx.CartesianPositionND.constructor([1, 2, 3], "kpc")
    >>> q2 = cx.CartesianPositionND.constructor([2, 3, 4], "kpc")
    >>> (q1 + q2).q
    Quantity['length'](Array([3., 5., 7.], dtype=float32), unit='kpc')

    """
    cart = rhs.represent_as(CartesianPositionND)
    return replace(lhs, q=lhs.q + cart.q)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_vcnd(lhs: ArrayLike, rhs: CartesianPositionND, /) -> CartesianPositionND:
    """Scale a position by a scalar.

    Examples
    --------
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> v = cx.CartesianPositionND(Quantity([1, 2, 3, 4, 5], "kpc"))
    >>> xp.multiply(2, v).q
    Quantity['length'](Array([ 2.,  4.,  6.,  8., 10.], dtype=float32), unit='kpc')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, q=lhs * rhs.q)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_cnd_pos(
    lhs: CartesianPositionND, rhs: AbstractPosition, /
) -> CartesianPositionND:
    """Subtract two vectors.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax import CartesianPositionND

    A 3D vector:

    >>> q1 = CartesianPositionND(Quantity([1, 2, 3], "kpc"))
    >>> q2 = CartesianPositionND(Quantity([2, 3, 4], "kpc"))
    >>> (q1 - q2).q
    Quantity['length'](Array([-1., -1., -1.], dtype=float32), unit='kpc')

    """
    cart = rhs.represent_as(CartesianPositionND)
    return replace(lhs, q=lhs.q - cart.q)


##############################################################################
# Velocity


@final
class CartesianVelocityND(AvalMixin, AbstractVelocityND):
    """Cartesian differential representation.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    A 1D vector:

    >>> q = cx.CartesianVelocityND(Quantity([[1]], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([[1.]], dtype=float32), unit='km / s')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = cx.CartesianVelocityND(Quantity([1, 2], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = cx.CartesianVelocityND(Quantity([1, 2, 3], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2., 3.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = cx.CartesianVelocityND(Quantity([1, 2, 3, 4], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2., 3., 4.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = cx.CartesianVelocityND(Quantity([1, 2, 3, 4, 5], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2., 3., 4., 5.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    """

    d_q: ct.BatchableSpeed = eqx.field(
        converter=partial(Quantity["speed"].constructor, dtype=float)
    )
    r"""N-D speed :math:`d\vec{x}/dt \in (-\infty, \infty).

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPositionND]:
        return CartesianPositionND

    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAccelerationND"]:
        return CartesianAccelerationND

    @partial(jax.jit, inline=True)
    def norm(self, _: AbstractPositionND | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        A 3D vector:

        >>> c = cx.CartesianVelocityND(Quantity([1, 2, 3], "km/s"))
        >>> c.norm()
        Quantity['speed'](Array(3.7416575, dtype=float32), unit='km / s')

        """
        return xp.linalg.vector_norm(self.d_q, axis=-1)


# -------------------------------------------------------------------


# TODO: move to the class in py3.11+
@CartesianVelocityND.constructor._f.dispatch  # type: ignore[attr-defined,misc]  # noqa: SLF001
def constructor(
    cls: type[CartesianVelocityND],
    x: Shaped[Quantity["speed"], ""] | Shaped[Quantity["speed"], "*batch N"],
    /,
) -> CartesianVelocityND:
    """Construct an N-dimensional velocity.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    1D vector:

    >>> cx.CartesianVelocityND.constructor(Quantity(1, "km/s"))
    CartesianVelocityND(
      d_q=Quantity[...]( value=f32[1], unit=Unit("km / s") )
    )

    >>> cx.CartesianVelocityND.constructor(Quantity([1], "km/s"))
    CartesianVelocityND(
      d_q=Quantity[...]( value=f32[1], unit=Unit("km / s") )
    )

    2D vector:

    >>> cx.CartesianVelocityND.constructor(Quantity([1, 2], "km/s"))
    CartesianVelocityND(
      d_q=Quantity[...]( value=f32[2], unit=Unit("km / s") )
    )

    3D vector:

    >>> cx.CartesianVelocityND.constructor(Quantity([1, 2, 3], "km/s"))
    CartesianVelocityND(
      d_q=Quantity[...]( value=f32[3], unit=Unit("km / s") )
    )

    4D vector:

    >>> cx.CartesianVelocityND.constructor(Quantity([1, 2, 3, 4], "km/s"))
    CartesianVelocityND(
      d_q=Quantity[...]( value=f32[4], unit=Unit("km / s") )
    )

    """
    return cls(jnp.atleast_1d(x))


##############################################################################
# Acceleration


@final
class CartesianAccelerationND(AvalMixin, AbstractAccelerationND):
    """Cartesian N-dimensional acceleration representation.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    A 1D vector:

    >>> q = cx.CartesianAccelerationND(Quantity([[1]], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([[1.]], dtype=float32), unit='km / s2')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = cx.CartesianAccelerationND(Quantity([1, 2], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([1., 2.], dtype=float32), unit='km / s2')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = cx.CartesianAccelerationND(Quantity([1, 2, 3], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([1., 2., 3.], dtype=float32), unit='km / s2')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = cx.CartesianAccelerationND(Quantity([1, 2, 3, 4], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([1., 2., 3., 4.], dtype=float32), unit='km / s2')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = cx.CartesianAccelerationND(Quantity([1, 2, 3, 4, 5], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([1., 2., 3., 4., 5.], dtype=float32), unit='km / s2')
    >>> q.shape
    ()

    """

    d2_q: ct.BatchableAcc = eqx.field(
        converter=partial(Quantity["acceleration"].constructor, dtype=float)
    )
    r"""N-D acceleration :math:`d\vec{x}/dt^2 \in (-\infty, \infty).

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVelocityND]:
        """Return the integral class.

        Examples
        --------
        >>> import coordinax as cx
        >>> cx.CartesianAccelerationND.integral_cls.__name__
        'CartesianVelocityND'

        """
        return CartesianVelocityND

    @classproperty
    @classmethod
    def differential_cls(cls) -> NoReturn:
        """Return the differential class.

        Examples
        --------
        >>> import coordinax as cx
        >>> try: cx.CartesianAccelerationND.differential_cls
        ... except NotImplementedError as e: print(e)
        Not yet supported

        """
        msg = "Not yet supported"
        raise NotImplementedError(msg)  # TODO: Implement this

    @partial(jax.jit, inline=True)
    def norm(
        self,
        velocity: AbstractVelocityND | None = None,  # noqa: ARG002
        position: AbstractPositionND | None = None,  # noqa: ARG002
        /,
    ) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx

        A 3D vector:

        >>> c = cx.CartesianAccelerationND(Quantity([1, 2, 3], "km/s2"))
        >>> c.norm()
        Quantity['acceleration'](Array(3.7416575, dtype=float32), unit='km / s2')

        """
        return xp.linalg.vector_norm(self.d2_q, axis=-1)


# -------------------------------------------------------------------


# TODO: move to the class in py3.11+
@CartesianAccelerationND.constructor._f.dispatch  # type: ignore[attr-defined,misc]  # noqa: SLF001
def constructor(
    cls: type[CartesianAccelerationND],
    x: Shaped[Quantity["acceleration"], ""]
    | Shaped[Quantity["acceleration"], "*batch N"],
    /,
) -> CartesianAccelerationND:
    """Construct an N-dimensional acceleration.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    1D vector:

    >>> cx.CartesianAccelerationND.constructor(Quantity(1, "km/s2"))
    CartesianAccelerationND(
      d2_q=Quantity[...]( value=f32[1], unit=Unit("km / s2") )
    )

    >>> cx.CartesianAccelerationND.constructor(Quantity([1], "km/s2"))
    CartesianAccelerationND(
      d2_q=Quantity[...]( value=f32[1], unit=Unit("km / s2") )
    )

    2D vector:

    >>> cx.CartesianAccelerationND.constructor(Quantity([1, 2], "km/s2"))
    CartesianAccelerationND(
      d2_q=Quantity[...]( value=f32[2], unit=Unit("km / s2") )
    )

    3D vector:

    >>> cx.CartesianAccelerationND.constructor(Quantity([1, 2, 3], "km/s2"))
    CartesianAccelerationND(
      d2_q=Quantity[...]( value=f32[3], unit=Unit("km / s2") )
    )

    4D vector:

    >>> cx.CartesianAccelerationND.constructor(Quantity([1, 2, 3, 4], "km/s2"))
    CartesianAccelerationND(
      d2_q=Quantity[...]( value=f32[4], unit=Unit("km / s2") )
    )

    """
    return cls(jnp.atleast_1d(x))
