"""Abstract additive operator base class."""

__all__ = ("AbstractAdd",)

from dataclasses import KW_ONLY, replace

from collections.abc import Callable
from typing import Any, Union

import equinox as eqx
import jax.tree as jtu
import numpy as np
import wadler_lindig as wl  # type: ignore[import-untyped]

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items

import coordinax.charts as cxc
from .base import AbstractOperator, Neg
from .pipe import Pipe
from coordinax.api import CsDict


class AbstractAdd(AbstractOperator):
    """Abstract base class for additive operators (Translate, Boost, etc.).

    Additive operators represent field-like quantities (displacements, velocity
    offsets, etc.) that can be combined via addition and negated.

    Common features:
    - Addition of two operators combines their offsets
    - Negation inverts the offset
    - Time-dependent offsets via callables
    - Chart-aware representation
    """

    delta: CsDict | Callable[[Any], Any]
    """The additive offset (displacement for Translate, velocity for Boost)."""

    chart: cxc.AbstractChart = eqx.field(static=True)  # type: ignore[type-arg]
    """Chart in which the offset is expressed."""

    _: KW_ONLY

    right_add: bool = eqx.field(default=True, static=True)
    """Whether to add on the right (x + offset) or left (offset + x)."""

    def _get_offset(self) -> CsDict | Callable[[Any], Any]:
        """Get the offset field value."""
        return self.delta

    def _make_inverse_offset(
        self, offset: CsDict | Callable[[Any], Any]
    ) -> CsDict | Callable[[Any], Any]:
        """Create the inverse (negated) offset.

        For CsDict: negates each component.
        For callable: wraps in Neg.
        """
        if not callable(offset) or isinstance(offset, Neg):
            return jtu.map(jnp.negative, offset, is_leaf=u.quantity.is_any_quantity)
        return Neg(offset)

    def _combine_offsets(
        self, other_offset: CsDict | Callable[[Any], Any]
    ) -> CsDict | Callable[[Any], Any]:
        """Combine this offset with another via addition.

        Only works for non-callable offsets.
        """
        self_offset = self._get_offset()
        if callable(self_offset) or callable(other_offset):
            raise TypeError("Cannot combine callable offsets")
        return jtu.map(
            jnp.add,
            self_offset,
            other_offset,
            is_leaf=u.quantity.is_any_quantity,
        )

    def __neg__(self) -> "AbstractAdd":
        """Return negative of the operator."""
        return self.inverse

    @property
    def inverse(self) -> "AbstractAdd":
        """The inverse operator (negated offset).

        Examples
        --------
        >>> import coordinax.ops as cxop

        >>> shift = cxop.Translate.from_([1, 2, 3], "km")
        >>> shift.inverse
        Translate(
            {'x': Q(-1, 'km'), 'y': Q(-2, 'km'), 'z': Q(-3, 'km')},
            chart=Cart3D()
        )

        >>> boost = cxop.Boost.from_([100, 0, 0], "km/s")
        >>> boost.inverse
        Boost(...)

        """
        inv = self._make_inverse_offset(self.delta)
        return replace(self, delta=inv)

    def __add__(self, other: object, /) -> Union["AbstractAdd", Pipe]:
        """Combine two operators of the same type."""
        if not isinstance(other, type(self)):
            return NotImplemented

        self_offset = self._get_offset()
        other_offset = other._get_offset()

        if not callable(self_offset) and not callable(other_offset):
            combined = self._combine_offsets(other_offset)
            # Create new instance with combined offset
            # Subclass must support construction with offset as first arg
            return type(self)(combined, chart=self.chart)
        return Pipe((self, other))

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, **kw: Any) -> wl.AbstractDoc:
        """Wadler-Lindig documentation for Translate operator."""
        kw.setdefault("include_params", False)
        kw.setdefault("short_arrays", "compact")
        kw.setdefault("use_short_names", True)
        kw.setdefault("named_unit", False)
        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=[
                wl.pdoc(self.delta, **kw),
                *wl.named_objs(
                    [
                        (k, v)
                        for k, v in field_items(self)
                        if k != "delta"
                        and not np.array_equal(v, self.__dataclass_fields__[k].default)
                    ],
                    **kw,
                ),
            ],
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=4,
        )

    def __repr__(self) -> str:
        """Return string representation of Add operator."""
        return wl.pformat(
            self.__pdoc__(
                short_arrays="compact",
                use_short_name=True,
                include_params=False,
                named_unit=False,
            ),
            width=80,
        )

    def __str__(self) -> str:
        """Return string representation of Add operator."""
        return wl.pformat(
            self.__pdoc__(
                short_arrays="compact",
                use_short_name=True,
                include_params=False,
                named_unit=False,
            ),
            width=80,
        )
