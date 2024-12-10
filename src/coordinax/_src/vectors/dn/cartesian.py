"""Built-in vector classes."""

__all__ = ["CartesianAccND", "CartesianPosND", "CartesianVelND"]

from dataclasses import replace
from functools import partial
from typing import NoReturn, final
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import ArrayLike, Shaped
from plum import conversion_method
from quax import register

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u

import coordinax._src.typing as ct
from .base import AbstractAccND, AbstractPosND, AbstractVelND
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractPos
from coordinax._src.vectors.base.mixins import AvalMixin

##############################################################################
# Position


@final
class CartesianPosND(AbstractPosND):
    """N-dimensional Cartesian vector representation.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    A 1D vector:

    >>> q = cx.vecs.CartesianPosND.from_([[1]], "km")
    >>> q.q
    Quantity['length'](Array([[1.]], dtype=float32), unit='km')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2], "km"))
    >>> q.q
    Quantity['length'](Array([1., 2.], dtype=float32), unit='km')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3], "km"))
    >>> q.q
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='km')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4], "km"))
    >>> q.q
    Quantity['length'](Array([1., 2., 3., 4.], dtype=float32), unit='km')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4, 5], "km"))
    >>> q.q
    Quantity['length'](Array([1., 2., 3., 4., 5.], dtype=float32), unit='km')
    >>> q.shape
    ()

    """

    q: BatchableLength = eqx.field(
        converter=partial(u.Quantity["length"].from_, dtype=float)
    )
    r"""N-D coordinate :math:`\vec{x} \in (-\infty,+\infty)`.

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVelND"]:  # type: ignore[override]
        return CartesianVelND

    # -----------------------------------------------------
    # Unary operations

    __neg__ = jnp.negative

    # -----------------------------------------------------

    @partial(eqx.filter_jit, inline=True)
    def norm(self) -> BatchableLength:
        """Return the norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        A 3D vector:

        >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3], "km"))
        >>> q.norm()
        Quantity['length'](Array(3.7416575, dtype=float32), unit='km')

        """
        return jnp.linalg.vector_norm(self.q, axis=-1)


# -------------------------------------------------------------------


# TODO: move to the class in py3.11+
@CartesianPosND.from_.dispatch  # type: ignore[attr-defined,misc]
def from_(
    cls: type[CartesianPosND],
    x: Shaped[u.Quantity["length"], ""] | Shaped[u.Quantity["length"], "*batch N"],
    /,
) -> CartesianPosND:
    """Construct an N-dimensional position.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    1D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity(1, "km"))
    CartesianPosND(
      q=Quantity[...](value=f32[1], unit=Unit("km"))
    )

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1], "km"))
    CartesianPosND(
      q=Quantity[...](value=f32[1], unit=Unit("km"))
    )

    2D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1, 2], "km"))
    CartesianPosND(
      q=Quantity[...](value=f32[2], unit=Unit("km"))
    )

    3D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1, 2, 3], "km"))
    CartesianPosND(
      q=Quantity[...](value=f32[3], unit=Unit("km"))
    )

    4D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1, 2, 3, 4], "km"))
    CartesianPosND(
      q=Quantity[...](value=f32[4], unit=Unit("km"))
    )

    """
    return cls(jnp.atleast_1d(x))


@conversion_method(CartesianPosND, u.Quantity)  # type: ignore[misc]
def _vec_to_q(obj: CartesianPosND, /) -> Shaped[u.Quantity["length"], "*batch N"]:
    """`coordinax.AbstractPos3D` -> `unxt.Quantity`.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4, 5], unit="km"))
    >>> convert(vec, u.Quantity)
    Quantity['length'](Array([1., 2., 3., 4., 5.], dtype=float32), unit='km')

    """
    return obj.q


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_vcnd(lhs: CartesianPosND, rhs: AbstractPos, /) -> CartesianPosND:
    """Add two vectors.

    Examples
    --------
    >>> import coordinax as cx

    A 3D vector:

    >>> q1 = cx.vecs.CartesianPosND.from_([1, 2, 3], "km")
    >>> q2 = cx.vecs.CartesianPosND.from_([2, 3, 4], "km")
    >>> (q1 + q2).q
    Quantity['length'](Array([3., 5., 7.], dtype=float32), unit='km')

    """
    cart = rhs.vconvert(CartesianPosND)
    return replace(lhs, q=lhs.q + cart.q)


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_vcnd(lhs: ArrayLike, rhs: CartesianPosND, /) -> CartesianPosND:
    """Scale a position by a scalar.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> v = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4, 5], "km"))
    >>> jnp.multiply(2, v).q
    Quantity['length'](Array([ 2.,  4.,  6.,  8., 10.], dtype=float32), unit='km')

    """
    # Validation
    lhs = eqx.error_if(
        lhs, any(jax.numpy.shape(lhs)), f"must be a scalar, not {type(lhs)}"
    )

    # Scale the components
    return replace(rhs, q=lhs * rhs.q)


@register(jax.lax.neg_p)  # type: ignore[misc]
def _neg_p_cartnd_pos(obj: CartesianPosND, /) -> CartesianPosND:
    """Negate the `coordinax.CartesianPosND`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    A 3D vector:

    >>> vec = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3], "km"))
    >>> (-vec).q
    Quantity['length'](Array([-1., -2., -3.], dtype=float32), unit='km')

    """
    return jax.tree.map(qlax.neg, obj)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_cnd_pos(lhs: CartesianPosND, rhs: AbstractPos, /) -> CartesianPosND:
    """Subtract two vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    A 3D vector:

    >>> q1 = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3], "km"))
    >>> q2 = cx.vecs.CartesianPosND(u.Quantity([2, 3, 4], "km"))
    >>> (q1 - q2).q
    Quantity['length'](Array([-1., -1., -1.], dtype=float32), unit='km')

    """
    cart = rhs.vconvert(CartesianPosND)
    return replace(lhs, q=lhs.q - cart.q)


##############################################################################
# Velocity


@final
class CartesianVelND(AvalMixin, AbstractVelND):
    """Cartesian differential representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    A 1D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([[1]], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([[1.]], dtype=float32), unit='km / s')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([1, 2], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2., 3.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3, 4], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2., 3., 4.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3, 4, 5], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1., 2., 3., 4., 5.], dtype=float32), unit='km / s')
    >>> q.shape
    ()

    """

    d_q: ct.BatchableSpeed = eqx.field(
        converter=partial(u.Quantity["speed"].from_, dtype=float)
    )
    r"""N-D speed :math:`d\vec{x}/dt \in (-\infty, \infty).

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPosND]:
        return CartesianPosND

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAccND"]:
        return CartesianAccND

    @partial(eqx.filter_jit, inline=True)
    def norm(self, _: AbstractPosND | None = None, /) -> ct.BatchableSpeed:
        """Return the norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        A 3D vector:

        >>> c = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3], "km/s"))
        >>> c.norm()
        Quantity['speed'](Array(3.7416575, dtype=float32), unit='km / s')

        """
        return jnp.linalg.vector_norm(self.d_q, axis=-1)


# -------------------------------------------------------------------


# TODO: move to the class in py3.11+
@CartesianVelND.from_.dispatch  # type: ignore[attr-defined,misc]
def from_(
    cls: type[CartesianVelND],
    x: Shaped[u.Quantity["speed"], ""] | Shaped[u.Quantity["speed"], "*batch N"],
    /,
) -> CartesianVelND:
    """Construct an N-dimensional velocity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    1D vector:

    >>> cx.vecs.CartesianVelND.from_(u.Quantity(1, "km/s"))
    CartesianVelND(
      d_q=Quantity[...]( value=f32[1], unit=Unit("km / s") )
    )

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1], "km/s"))
    CartesianVelND(
      d_q=Quantity[...]( value=f32[1], unit=Unit("km / s") )
    )

    2D vector:

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1, 2], "km/s"))
    CartesianVelND(
      d_q=Quantity[...]( value=f32[2], unit=Unit("km / s") )
    )

    3D vector:

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1, 2, 3], "km/s"))
    CartesianVelND(
      d_q=Quantity[...]( value=f32[3], unit=Unit("km / s") )
    )

    4D vector:

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1, 2, 3, 4], "km/s"))
    CartesianVelND(
      d_q=Quantity[...]( value=f32[4], unit=Unit("km / s") )
    )

    """
    return cls(jnp.atleast_1d(x))


##############################################################################
# Acceleration


@final
class CartesianAccND(AvalMixin, AbstractAccND):
    """Cartesian N-dimensional acceleration representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    A 1D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([[1]], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([[1.]], dtype=float32), unit='km / s2')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([1, 2], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([1., 2.], dtype=float32), unit='km / s2')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([1, 2, 3], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([1., 2., 3.], dtype=float32), unit='km / s2')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([1, 2, 3, 4], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([1., 2., 3., 4.], dtype=float32), unit='km / s2')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([1, 2, 3, 4, 5], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([1., 2., 3., 4., 5.], dtype=float32), unit='km / s2')
    >>> q.shape
    ()

    """

    d2_q: ct.BatchableAcc = eqx.field(
        converter=partial(u.Quantity["acceleration"].from_, dtype=float)
    )
    r"""N-D acceleration :math:`d\vec{x}/dt^2 \in (-\infty, \infty).

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVelND]:
        """Return the integral class.

        Examples
        --------
        >>> import coordinax as cx
        >>> cx.vecs.CartesianAccND.integral_cls.__name__
        'CartesianVelND'

        """
        return CartesianVelND

    @classproperty
    @classmethod
    def differential_cls(cls) -> NoReturn:
        """Return the differential class.

        Examples
        --------
        >>> import coordinax as cx
        >>> try: cx.vecs.CartesianAccND.differential_cls
        ... except NotImplementedError as e: print(e)
        Not yet supported

        """
        msg = "Not yet supported"
        raise NotImplementedError(msg)  # TODO: Implement this

    @override
    @partial(eqx.filter_jit, inline=True)
    def norm(
        self,
        velocity: AbstractVelND | None = None,
        position: AbstractPosND | None = None,
        /,
    ) -> ct.BatchableAcc:
        """Return the norm of the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        A 3D vector:

        >>> c = cx.vecs.CartesianAccND(u.Quantity([1, 2, 3], "km/s2"))
        >>> c.norm()
        Quantity['acceleration'](Array(3.7416575, dtype=float32), unit='km / s2')

        """
        return jnp.linalg.vector_norm(self.d2_q, axis=-1)


# -------------------------------------------------------------------


# TODO: move to the class in py3.11+
@CartesianAccND.from_.dispatch  # type: ignore[attr-defined,misc]
def from_(
    cls: type[CartesianAccND],
    x: Shaped[u.Quantity["acceleration"], ""]
    | Shaped[u.Quantity["acceleration"], "*batch N"],
    /,
) -> CartesianAccND:
    """Construct an N-dimensional acceleration.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    1D vector:

    >>> cx.vecs.CartesianAccND.from_(u.Quantity(1, "km/s2"))
    CartesianAccND(
      d2_q=Quantity[...]( value=f32[1], unit=Unit("km / s2") )
    )

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1], "km/s2"))
    CartesianAccND(
      d2_q=Quantity[...]( value=f32[1], unit=Unit("km / s2") )
    )

    2D vector:

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1, 2], "km/s2"))
    CartesianAccND(
      d2_q=Quantity[...]( value=f32[2], unit=Unit("km / s2") )
    )

    3D vector:

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1, 2, 3], "km/s2"))
    CartesianAccND(
      d2_q=Quantity[...]( value=f32[3], unit=Unit("km / s2") )
    )

    4D vector:

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1, 2, 3, 4], "km/s2"))
    CartesianAccND(
      d2_q=Quantity[...]( value=f32[4], unit=Unit("km / s2") )
    )

    """
    return cls(jnp.atleast_1d(x))
