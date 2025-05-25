"""Built-in 4-vector classes."""

__all__ = ["FourVector"]

import functools as ft
from dataclasses import KW_ONLY
from typing import Any, cast, final
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import wadler_lindig as wl
from jaxtyping import Shaped

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_values
from dataclassish.converters import Unless

import coordinax._src.custom_types as ct
from .base import AbstractPos4D
from coordinax._src.distances import BBtLength
from coordinax._src.vectors import dims
from coordinax._src.vectors.base import AttrFilter, VectorAttribute
from coordinax._src.vectors.d3 import AbstractPos3D, CartesianPos3D

##############################################################################
# Position


@final
class FourVector(AbstractPos4D):
    """3+1 vector representation.

    The 3+1 vector representation is a 4-vector with 3 spatial coordinates and 1
    time coordinate.

    Parameters
    ----------
    t, q
        Time and spatial coordinates.
    c
        Speed of light, by default ``Quantity(299_792.458, "km/s")``.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    Create a 3+1 vector with a time and 3 spatial coordinates:

    >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> print(w)
    <FourVector: (t[s], q=(x, y, z) [m])
        [1 1 2 3]>

    Note that we used a shortcut to create the 3D vector by passing a ``(*batch,
    3)`` array to the `q` argument. This assumes that `q` is a
    `coordinax.CartesianPos3D` and uses the
    :meth:`coordinax.CartesianPos3D.from_` method to create the 3D vector.

    We can also create a 3D vector explicitly:

    >>> q = cx.SphericalPos(theta=u.Quantity(1, "deg"), phi=u.Quantity(2, "deg"),
    ...                     r=u.Quantity(3, "m"))
    >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=q)
    >>> print(w)
    <FourVector: (t[s], q=(r[m], theta[deg], phi[deg]))
        [1 3 1 2]>

    """

    t: ct.BBtTime | ct.ScalarTime = eqx.field(converter=u.Quantity["time"].from_)
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

    # TODO: merge with `AvalMixin` and generalize!
    def aval(self) -> jax.core.ShapedArray:
        avals = (self.t.aval(), self.q.aval())
        shape = (*jnp.broadcast_shapes(avals[0].shape, avals[1].shape[:-1]), 4)
        dtype = jnp.result_type(*map(jnp.dtype, avals))
        return jax.core.ShapedArray(shape, dtype)

    # ===============================================================

    def __getattr__(self, name: str) -> Any:
        """Get the attribute from the 3-vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
        >>> w.x
        Quantity(Array(1, dtype=int32), unit='m')

        """
        return getattr(self.q, name)

    # -------------------------------------------

    @ft.partial(eqx.filter_jit, inline=True)
    def _norm2(self) -> Shaped[u.Quantity["area"], "*#batch"]:
        r"""Return the squared vector norm :math:`(ct)^2 - (x^2 + y^2 + z^2)`.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
        >>> w._norm2()
        Quantity(Array(8.987552e+10, dtype=float32), unit='km2')

        """
        return (self.c * self.t) ** 2 - (self.q.norm() ** 2)  # type: ignore[misc,operator]

    @override
    @ft.partial(eqx.filter_jit, inline=True)
    def norm(self) -> BBtLength:
        r"""Return the vector norm :math:`\sqrt{(ct)^2 - (x^2 + y^2 + z^2)}`.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
        >>> w.norm()
        Quantity(Array(299792.47+0.j, dtype=complex64), unit='km')

        """
        norm2 = jnp.asarray(self._norm2(), dtype=complex)  # type: ignore[misc]
        return jnp.sqrt(norm2)

    @override
    @property
    def dimensions(self) -> dict[str, u.dims.AbstractDimension]:  # type: ignore[override]
        """Vector physical dimensions.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> cx.FourVector.dimensions
        <property object at ...>

        >>> w = cx.FourVector (t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
        >>> w.dimensions
        {'t': PhysicalType('time'),
         'q': {'x': PhysicalType('length'), 'y': PhysicalType('length'),
               'z': PhysicalType('length')}}

        """
        return {"t": dims.T, "q": self.q.dimensions}

    # ===============================================================
    # Wadler-Lindig

    @override
    def __pdoc__(self, *, vector_form: bool = False, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig docstring for the vector.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> w = cx.FourVector (t=u.Quantity(0.5, "s"), q=u.Quantity([1, 2, 3], "m"))
        >>> print(w)
        <FourVector: (t[s], q=(x, y, z) [m])
            [0.5 1.  2.  3. ]>

        """
        if not vector_form:
            return super().__pdoc__(vector_form=vector_form, **kwargs)

        cls_name = wl.TextDoc(self.__class__.__name__)

        # make the components string
        comps_doc = (
            wl.TextDoc("(")
            + wl.TextDoc(f"t[{self.units['t']}]")
            + wl.comma
            + wl.TextDoc("q=")
            + self.q._pdoc_comps()
            + wl.TextDoc(")")
        ).group()

        vs = np.array2string(  # type: ignore[call-overload]  # TODO: use other method
            jnp.stack(
                tuple(
                    u.ustrip(v)
                    for v in cast(
                        list[u.AbstractQuantity],
                        jnp.broadcast_arrays(self.t, *field_values(AttrFilter, self.q)),
                    )
                ),
                axis=-1,
            ),
            precision=3,
            suffix=">",
        )
        return (
            (wl.TextDoc("<") + cls_name + wl.TextDoc(":")).group()
            + wl.BreakDoc(" ")
            + comps_doc
            + wl.TextDoc("\n")  # force a line break
            + (wl.TextDoc("    ") + wl.TextDoc(vs).nest(4) + wl.TextDoc(">")).group()
        )
