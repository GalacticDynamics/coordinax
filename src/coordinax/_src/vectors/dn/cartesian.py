"""Built-in vector classes."""

__all__ = ["CartesianAccND", "CartesianPosND", "CartesianVelND"]

from functools import partial
from typing import NoReturn, final
from typing_extensions import override

import equinox as eqx
import jax
from jaxtyping import Shaped
from plum import conversion_method, dispatch

import quaxed.numpy as jnp
import unxt as u
from quaxed.experimental import arrayish
from unxt.quantity import AbstractQuantity

import coordinax._src.typing as ct
from .base import AbstractAccND, AbstractPosND, AbstractVelND
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty

##############################################################################
# Position


@final
class CartesianPosND(AbstractPosND, arrayish.NumpyNegMixin):
    """N-dimensional Cartesian vector representation.

    Examples
    --------
    >>> import coordinax as cx
    >>> import unxt as u

    A 1D vector:

    >>> q = cx.vecs.CartesianPosND.from_([[1]], "km")
    >>> q.q
    Quantity['length'](Array([[1]], dtype=int32), unit='km')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2], "km"))
    >>> q.q
    Quantity['length'](Array([1, 2], dtype=int32), unit='km')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3], "km"))
    >>> q.q
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='km')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4], "km"))
    >>> q.q
    Quantity['length'](Array([1, 2, 3, 4], dtype=int32), unit='km')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4, 5], "km"))
    >>> q.q
    Quantity['length'](Array([1, 2, 3, 4, 5], dtype=int32), unit='km')
    >>> q.shape
    ()

    """

    q: BatchableLength = eqx.field(converter=u.Quantity["length"].from_)
    r"""N-D coordinate :math:`\vec{x} \in (-\infty,+\infty)`.

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @override
    def _dimensionality(self) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        A 3D vector:

        >>> cx.vecs.CartesianPosND(u.Quantity([1, 2, 3], "km"))._dimensionality()
        3

        """
        return self.q.shape[-1]

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianVelND"]:  # type: ignore[override]
        """Return the differential class.

        Examples
        --------
        >>> import coordinax as cx
        >>> cx.vecs.CartesianPosND.differential_cls
        <class 'coordinax...CartesianVelND'>

        """
        return CartesianVelND

    # ===============================================================
    # Quax API

    @override
    def aval(self) -> jax.core.ShapedArray:
        """Simpler aval than superclass."""
        return jax.core.get_aval(self.q.value)  # type: ignore[no-untyped-call]

    # -----------------------------------------------------

    @override
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


@conversion_method(CartesianPosND, u.Quantity)  # type: ignore[arg-type]
def vec_to_q(obj: CartesianPosND, /) -> Shaped[u.Quantity["length"], "*batch N"]:
    """`coordinax.AbstractPos3D` -> `unxt.Quantity`.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4, 5], unit="km"))
    >>> convert(vec, u.Quantity)
    Quantity['length'](Array([1, 2, 3, 4, 5], dtype=int32), unit='km')

    """
    return obj.q


##############################################################################
# Velocity


@final
class CartesianVelND(AbstractVelND):
    """Cartesian differential representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    A 1D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([[1]], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([[1]], dtype=int32), unit='km / s')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([1, 2], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1, 2], dtype=int32), unit='km / s')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1, 2, 3], dtype=int32), unit='km / s')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3, 4], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1, 2, 3, 4], dtype=int32), unit='km / s')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3, 4, 5], "km/s"))
    >>> q.d_q
    Quantity['speed'](Array([1, 2, 3, 4, 5], dtype=int32), unit='km / s')
    >>> q.shape
    ()

    """

    d_q: ct.BatchableSpeed = eqx.field(converter=u.Quantity["speed"].from_)
    r"""N-D speed :math:`d\vec{x}/dt \in (-\infty, \infty).

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @override
    def _dimensionality(self) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        A 3D vector:

        >>> cx.vecs.CartesianVelND(u.Quantity([1, 2, 3], "km/s"))._dimensionality()
        3

        """
        return self.d_q.shape[-1]

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianPosND]:  # type: ignore[override]
        """Return the integral class.

        Examples
        --------
        >>> import coordinax as cx
        >>> cx.vecs.CartesianVelND.integral_cls
        <class 'coordinax...CartesianPosND'>

        """
        return CartesianPosND

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> type["CartesianAccND"]:  # type: ignore[override]
        """Return the differential class.

        Examples
        --------
        >>> import coordinax as cx
        >>> cx.vecs.CartesianVelND.differential_cls
        <class 'coordinax...CartesianAccND'>

        """
        return CartesianAccND

    @override
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


##############################################################################
# Acceleration


@final
class CartesianAccND(AbstractAccND):
    """Cartesian N-dimensional acceleration representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    A 1D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([[1]], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([[1]], dtype=int32), unit='km / s2')
    >>> q.shape
    (1,)

    A 2D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([1, 2], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([1, 2], dtype=int32), unit='km / s2')
    >>> q.shape
    ()

    A 3D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([1, 2, 3], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([1, 2, 3], dtype=int32), unit='km / s2')
    >>> q.shape
    ()

    A 4D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([1, 2, 3, 4], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([1, 2, 3, 4], dtype=int32), unit='km / s2')
    >>> q.shape
    ()

    A 5D vector:

    >>> q = cx.vecs.CartesianAccND(u.Quantity([1, 2, 3, 4, 5], "km/s2"))
    >>> q.d2_q
    Quantity['acceleration'](Array([1, 2, 3, 4, 5], dtype=int32), unit='km / s2')
    >>> q.shape
    ()

    """

    d2_q: ct.BatchableAcc = eqx.field(converter=u.Quantity["acceleration"].from_)
    r"""N-D acceleration :math:`d\vec{x}/dt^2 \in (-\infty, \infty).

    Should have shape (*batch, F) where F is the number of features /
    dimensions. Arbitrary batch shapes are supported.
    """

    @override
    def _dimensionality(self) -> int:
        """Dimensionality of the vector.

        Examples
        --------
        >>> import coordinax as cx

        A 3D vector:

        >>> cx.vecs.CartesianAccND(u.Quantity([1, 2, 3], "km/s2"))._dimensionality()
        3

        """
        return self.d2_q.shape[-1]

    @override
    @classproperty
    @classmethod
    def integral_cls(cls) -> type[CartesianVelND]:  # type: ignore[override]
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
    def norm(  # type: ignore[override]
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


# ==============================================================================
# Constructors


@dispatch
def vector(
    cls: type[CartesianPosND] | type[CartesianVelND] | type[CartesianAccND],
    x: AbstractQuantity,
    /,
) -> CartesianPosND | CartesianVelND | CartesianAccND:
    """Construct an N-dimensional acceleration.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    1D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity(1, "km"))
    CartesianPosND(
      q=Quantity[...](value=...i32[1], unit=Unit("km"))
    )

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1], "km"))
    CartesianPosND(
      q=Quantity[...](value=...i32[1], unit=Unit("km"))
    )

    >>> cx.vecs.CartesianVelND.from_(u.Quantity(1, "km/s"))
    CartesianVelND(
      d_q=Quantity[...]( value=...i32[1], unit=Unit("km / s") )
    )

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1], "km/s"))
    CartesianVelND(
      d_q=Quantity[...]( value=...i32[1], unit=Unit("km / s") )
    )

    >>> cx.vecs.CartesianAccND.from_(u.Quantity(1, "km/s2"))
    CartesianAccND(
      d2_q=Quantity[...]( value=...i32[1], unit=Unit("km / s2") )
    )

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1], "km/s2"))
    CartesianAccND(
      d2_q=Quantity[...]( value=...i32[1], unit=Unit("km / s2") )
    )

    2D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1, 2], "km"))
    CartesianPosND(
      q=Quantity[...](value=...i32[2], unit=Unit("km"))
    )

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1, 2], "km/s"))
    CartesianVelND(
      d_q=Quantity[...]( value=...i32[2], unit=Unit("km / s") )
    )

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1, 2], "km/s2"))
    CartesianAccND(
      d2_q=Quantity[...]( value=...i32[2], unit=Unit("km / s2") )
    )

    3D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1, 2, 3], "km"))
    CartesianPosND(
      q=Quantity[...](value=...i32[3], unit=Unit("km"))
    )

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1, 2, 3], "km/s"))
    CartesianVelND(
      d_q=Quantity[...]( value=...i32[3], unit=Unit("km / s") )
    )

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1, 2, 3], "km/s2"))
    CartesianAccND(
      d2_q=Quantity[...]( value=...i32[3], unit=Unit("km / s2") )
    )

    4D vector:

    >>> cx.vecs.CartesianPosND.from_(u.Quantity([1, 2, 3, 4], "km"))
    CartesianPosND(
      q=Quantity[...](value=...i32[4], unit=Unit("km"))
    )

    >>> cx.vecs.CartesianVelND.from_(u.Quantity([1, 2, 3, 4], "km/s"))
    CartesianVelND(
      d_q=Quantity[...]( value=...i32[4], unit=Unit("km / s") )
    )

    >>> cx.vecs.CartesianAccND.from_(u.Quantity([1, 2, 3, 4], "km/s2"))
    CartesianAccND(
      d2_q=Quantity[...]( value=...i32[4], unit=Unit("km / s2") )
    )

    """
    return cls(jnp.atleast_1d(x))
