"""Built-in 4-vector classes."""

__all__ = ["FourVector"]

from dataclasses import KW_ONLY
from functools import partial
from typing import TYPE_CHECKING, Any, cast, final
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Shaped

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
from coordinax._src.vectors.d3 import AbstractPos3D, CartesianPos3D

if TYPE_CHECKING:
    import typing

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
    def _cartesian_cls(cls) -> type[AbstractVector]:  # type: ignore[override]
        return CartesianPos3D

    @override
    @classproperty
    @classmethod
    def differential_cls(cls) -> "typing.Never":  # type: ignore[override]
        msg = "Not yet implemented"
        raise NotImplementedError(msg)

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
        Quantity['area'](Array(8.987552e+10, dtype=float32), unit='km2')

        """
        return (self.c * self.t) ** 2 - (self.q.norm() ** 2)  # type: ignore[misc,operator]

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
        Quantity['length'](Array(299792.47+0.j, dtype=complex64), unit='km')

        """
        norm2 = jnp.asarray(self._norm2(), dtype=complex)  # type: ignore[misc]
        return jnp.sqrt(norm2)

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
            jnp.stack(  # type: ignore[arg-type]
                tuple(
                    u.ustrip(v)
                    for v in cast(
                        list[AbstractQuantity],
                        jnp.broadcast_arrays(self.t, *field_values(AttrFilter, self.q)),
                    )
                ),
                axis=-1,
            ),
            precision=3,
            prefix="    ",
        )
        return f"<{cls_name} ({comps})\n    {vs}>"
