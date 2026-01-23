"""Abstract additive operator base class."""

__all__ = ("AbstractAdd",)

from dataclasses import KW_ONLY, replace

from collections.abc import Callable
from typing import Any, Union

import equinox as eqx
import jax.tree as jtu

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractOperator, Neg
from .pipe import Pipe
from coordinax._src import charts as cxc
from coordinax._src.custom_types import CsDict


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

    chart: cxc.AbstractChart = eqx.field(static=True)
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
            raise ValueError("Cannot combine callable offsets")
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
        >>> import coordinax.ops as cxo

        >>> shift = cxo.Translate.from_([1, 2, 3], "km")
        >>> shift.inverse
        Translate(
            {'x': Q(-1, 'km'), 'y': Q(-2, 'km'), 'z': Q(-3, 'km')},
            chart=Cart3D()
        )

        >>> boost = cxo.Boost.from_([100, 0, 0], "km/s")
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
