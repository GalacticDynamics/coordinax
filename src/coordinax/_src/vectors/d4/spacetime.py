"""Built-in 4-vector classes."""

__all__ = ["FourVector"]

from dataclasses import KW_ONLY, replace
from functools import partial
from typing import TYPE_CHECKING, Any, final
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Shaped
from plum import dispatch
from quax import register

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_values
from dataclassish.converters import Unless
from unxt.quantity import AbstractQuantity

import coordinax._src.typing as ct
from .base import AbstractPos4D
from coordinax._src.distances import BatchableLength
from coordinax._src.utils import classproperty
from coordinax._src.vectors.base import AbstractVector, AttrFilter, VectorAttribute
from coordinax._src.vectors.d3.base import AbstractPos3D
from coordinax._src.vectors.d3.cartesian import CartesianPos3D

if TYPE_CHECKING:
    from typing import Never

##############################################################################
# Position


@final
class FourVector(AbstractPos4D):
    """3+1 vector representation.

    The 3+1 vector representation is a 4-vector with 3 spatial coordinates and 1
    time coordinate.

    Parameters
    ----------
    t : Quantity[float, (*batch,), "time"]
        Time coordinate.
    q : AbstractPos3D[float, (*batch, 3)]
        Spatial coordinates.
    c : Quantity[float, (), "speed"], optional
        Speed of light, by default ``Quantity(299_792.458, "km/s")``.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Create a 3+1 vector with a time and 3 spatial coordinates:

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> print(w)
    <FourVector (t[s], q=(x[m], y[m], z[m]))
        [1 1 2 3]>

    Note that we used a shortcut to create the 3D vector by passing a ``(*batch,
    3)`` array to the `q` argument. This assumes that `q` is a
    :class:`coordinax.CartesianPos3D` and uses the
    :meth:`coordinax.CartesianPos3D.from_` method to create the 3D vector.

    We can also create a 3D vector explicitly:

    >>> q = cx.SphericalPos(theta=u.Quantity(1, "deg"), phi=u.Quantity(2, "deg"),
    ...                     r=u.Quantity(3, "m"))
    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=q)
    >>> print(w)
    <FourVector (t[s], q=(r[m], theta[deg], phi[deg]))
        [1 3 1 2]>

    """

    t: ct.BatchableTime | ct.ScalarTime = eqx.field(converter=u.Quantity["time"].from_)
    """Time coordinate."""

    q: AbstractPos3D = eqx.field(converter=Unless(AbstractPos3D, CartesianPos3D.from_))
    """Spatial coordinates."""

    _: KW_ONLY
    c: Shaped[u.Quantity["speed"], ""] = eqx.field(
        default=VectorAttribute(default=u.Quantity(299_792.458, "km/s")), repr=False
    )
    """Speed of light, by default ``Quantity(299_792.458, "km/s")``."""

    def __check_init__(self) -> None:
        """Check that the initialization is valid."""
        # Check the shapes are the same, allowing for broadcasting of the time.
        shape = jnp.broadcast_shapes(self.t.shape, self.q.shape)
        if shape != self.q.shape:
            msg = "t and q must be broadcastable to the same shape."
            raise ValueError(msg)

    # ===============================================================

    def __getattr__(self, name: str) -> Any:
        """Get the attribute from the 3-vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
        >>> w.x
        Quantity['length'](Array(1, dtype=int32), unit='m')

        """
        return getattr(self.q, name)

    # -------------------------------------------

    @override
    @classproperty
    @classmethod
    def _cartesian_cls(cls) -> type[AbstractVector]:
        msg = "Not yet implemented"
        raise NotImplementedError(msg)

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> "Never":  # type: ignore[override]
        msg = "Not yet implemented"
        raise NotImplementedError(msg)

    # -------------------------------------------
    # Unary operations

    @override
    def __neg__(self) -> "FourVector":
        """Negate the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
        >>> print(-w)
        <FourVector (t[s], q=(x[m], y[m], z[m]))
            [-1 -1 -2 -3]>

        """
        return replace(self, t=-self.t, q=-self.q)

    # -------------------------------------------

    @partial(eqx.filter_jit, inline=True)
    def _norm2(self) -> Shaped[u.Quantity["area"], "*#batch"]:
        r"""Return the squared vector norm :math:`(ct)^2 - (x^2 + y^2 + z^2)`.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
        >>> w._norm2()
        Quantity['area'](Array(8.987552e+16, dtype=float32), unit='m2')

        """
        return -(self.q.norm() ** 2) + (self.c * self.t) ** 2  # for units

    @override
    @partial(eqx.filter_jit, inline=True)
    def norm(self) -> BatchableLength:
        r"""Return the vector norm :math:`\sqrt{(ct)^2 - (x^2 + y^2 + z^2)}`.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
        >>> w.norm()
        Quantity['length'](Array(2.9979248e+08+0.j, dtype=complex64), unit='m')

        """
        return jnp.sqrt(jnp.asarray(self._norm2(), dtype=complex))

    # -------------------------------------------
    # misc

    def __str__(self) -> str:
        r"""Return a string representation of the spacetime vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> w = cx.FourVector(t=u.Quantity(0.5, "s"), q=u.Quantity([1, 2, 3], "m"))
        >>> print(w)
        <FourVector (t[s], q=(x[m], y[m], z[m]))
            [0.5 1.  2.  3. ]>

        """
        cls_name = type(self).__name__
        qcomps = ", ".join(f"{c}[{self.q.units[c]}]" for c in self.q.components)
        comps = f"t[{self.units['t']}], q=({qcomps})"
        vs = np.array2string(
            jnp.stack(
                tuple(
                    v.value
                    for v in jnp.broadcast_arrays(
                        self.t, *field_values(AttrFilter, self.q)
                    )
                ),
                axis=-1,
            ),
            precision=3,
            prefix="    ",
        )
        return f"<{cls_name} ({comps})\n    {vs}>"


# ===============================================================
# Constructors


@dispatch  # type: ignore[misc]
def vector(cls: type[FourVector], obj: AbstractQuantity, /) -> FourVector:
    """Construct a vector from a Quantity array.

    The ``Quantity[Any, (*#batch, 4), "..."]`` is expected to have the
    components as the last dimension. The 4 components are the (c x) time, x, y,
    z.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> xs = u.Quantity([0, 1, 2, 3], "meter")  # [ct, x, y, z]
    >>> vec = cx.FourVector.from_(xs)
    >>> vec
    FourVector(
        t=Quantity[...](value=...f32[], unit=Unit("m s / km")),
        q=CartesianPos3D(
            x=Quantity[...](value=i32[], unit=Unit("m")),
            y=Quantity[...](value=i32[], unit=Unit("m")),
            z=Quantity[...](value=i32[], unit=Unit("m"))
        )
    )

    >>> xs = u.Quantity(jnp.array([[0, 1, 2, 3], [10, 4, 5, 6]]), "meter")
    >>> vec = cx.FourVector.from_(xs)
    >>> vec
    FourVector(
        t=Quantity[...](value=...f32[2], unit=Unit("m s / km")),
        q=CartesianPos3D(
            x=Quantity[...](value=i32[2], unit=Unit("m")),
            y=Quantity[...](value=i32[2], unit=Unit("m")),
            z=Quantity[...](value=i32[2], unit=Unit("m"))
        )
    )

    """
    _ = eqx.error_if(
        obj,
        obj.shape[-1] != 4,
        f"Cannot construct {cls} from array with shape {obj.shape}.",
    )
    c = cls.__dataclass_fields__["c"].default.default
    return cls(t=obj[..., 0] / c, q=obj[..., 1:], c=c)


# ===============================================================


@register(jax.lax.add_p)  # type: ignore[misc]
def _add_4v4v(self: FourVector, other: FourVector) -> FourVector:
    """Add two 4-vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w1 = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> w2 = cx.FourVector(t=u.Quantity(2, "s"), q=u.Quantity([4, 5, 6], "m"))
    >>> w3 = w1 + w2
    >>> print(w3)
    <FourVector (t[s], q=(x[m], y[m], z[m]))
        [3 5 7 9]>

    """
    return replace(self, t=self.t + other.t, q=self.q + other.q)


@register(jax.lax.sub_p)  # type: ignore[misc]
def _sub_4v_4v(lhs: FourVector, rhs: FourVector) -> FourVector:
    """Add two 4-vectors.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w1 = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> w2 = cx.FourVector(t=u.Quantity(2, "s"), q=u.Quantity([4, 5, 6], "m"))
    >>> w3 = w1 - w2
    >>> print(w3)
    <FourVector (t[s], q=(x[m], y[m], z[m]))
        [-1 -3 -3 -3]>

    """
    return replace(lhs, t=lhs.t - rhs.t, q=lhs.q - rhs.q)
