"""Abstract additive operator base class."""

__all__ = ("AbstractAdd",)

from dataclasses import KW_ONLY, replace

from collections.abc import Callable
from jaxtyping import ArrayLike
from typing import Any, Union, cast

import equinox as eqx
import jax.tree as jtu
import plum
import wadler_lindig as wl

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from unxt.quantity import AllowValue, is_any_quantity

import coordinax.charts as cxc
from .base import AbstractTransform
from .composed import Composed
from .custom_types import CDict
from .identity import Identity, identity
from .utils import Neg
from coordinax.internal import jax_scalar_handler, pos_named_objs


class AbstractAdd(AbstractTransform):
    """Abstract base class for additive operators (Translate, Boost, etc.).

    Additive operators represent field-like quantities (displacements, velocity
    offsets, etc.) that can be combined via addition and negated.

    Common features:
    - Addition of two operators combines their offsets
    - Negation inverts the offset
    - Time-dependent offsets via callables
    - Chart-aware representation
    """

    delta: CDict | Callable[[Any], Any]
    """The additive offset (displacement for Translate, velocity for Boost)."""

    chart: cxc.AbstractChart = eqx.field(static=True)
    """Chart in which the offset is expressed."""

    _: KW_ONLY

    right_add: bool = eqx.field(default=True, static=True)
    """Whether to add on the right (x + offset) or left (offset + x)."""

    def _combine_offsets(
        self, other_offset: CDict | Callable[[Any], Any]
    ) -> CDict | Callable[[Any], Any]:
        """Combine this offset with another via addition.

        Only works for non-callable offsets.
        """
        self_offset = self.delta
        if callable(self_offset) or callable(other_offset):
            raise TypeError("Cannot combine callable offsets")
        return jtu.map(jnp.add, self_offset, other_offset, is_leaf=is_any_quantity)

    def __neg__(self) -> "AbstractAdd":
        """Return negative of the operator."""
        return self.inverse

    @property
    def inverse(self) -> "AbstractAdd":
        """The inverse operator (negated offset).

        Examples
        --------
        >>> import coordinax.transforms as cxfm

        >>> shift = cxfm.Translate.from_([1, 2, 3], "km")
        >>> shift.inverse
        Translate(
            {'x': Q(-1, 'km'), 'y': Q(-2, 'km'), 'z': Q(-3, 'km')},
            chart=Cart3D()
        )

        """
        delta = self.delta
        if not callable(delta) or isinstance(delta, Neg):
            inv = jtu.map(jnp.negative, delta, is_leaf=is_any_quantity)
        else:
            inv = Neg(delta)
        return replace(self, delta=inv)

    def __add__(self, other: object, /) -> Union["AbstractAdd", Composed]:
        """Combine two operators of the same type."""
        if not isinstance(other, type(self)):
            return NotImplemented

        other_offset = other.delta

        if not callable(self.delta) and not callable(other_offset):
            combined = self._combine_offsets(other_offset)
            return replace(self, delta=combined)
        return Composed((self, other))

    # ===============================================================
    # Wadler-Lindig API

    def __pdoc__(self, **kw: Any) -> wl.AbstractDoc:
        """Wadler-Lindig documentation for Translate operator."""
        # Set pdoc option defaults
        kw.setdefault("include_params", False)
        kw.setdefault("short_arrays", "compact")
        kw.setdefault("use_short_names", True)
        kw.setdefault("named_unit", False)

        # Build the fields
        fitems = cast("list[tuple[str, Any]]", field_items(self))
        kw = {**kw, "custom": jax_scalar_handler}
        docs = pos_named_objs(fitems, ["delta"], self.__dataclass_fields__, **kw)

        # Return the full doc
        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=docs,
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


# ============================================================================
# Constructors


@AbstractAdd.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[AbstractAdd], obj: AbstractAdd, /) -> AbstractAdd:
    """Construct a AbstractAdd from another AbstractAdd.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> shift1 = cxfm.Translate.from_([1, 2, 3], "km")
    >>> cxfm.Translate.from_(shift1) is shift1
    True

    """
    if type(obj) is not cls:
        raise TypeError(f"Cannot construct {cls.__name__} from {type(obj).__name__}")
    return obj


@AbstractAdd.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[AbstractAdd], q: u.AbstractQuantity, /) -> AbstractAdd:
    """Construct an AbstractAdd subclass from a Quantity.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> cxfm.Translate.from_(u.Q([1, 2, 3], "km"))
    Translate({'x': Q(1, 'km'), 'y': Q(2, 'km'), 'z': Q(3, 'km')}, chart=Cart3D())

    """
    chart = cxc.guess_chart(q)
    x = cxc.cdict(q, chart)
    return cls(x, chart=chart)


@AbstractAdd.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[AbstractAdd], x: ArrayLike, unit: str) -> AbstractAdd:
    """Construct an Add operator from an array-like offset and unit.

    Examples
    --------
    >>> import coordinax.transforms as cxfm

    >>> cxfm.Translate.from_([1, 2, 3], "km")
    Translate({'x': Q(1, 'km'), 'y': Q(2, 'km'), 'z': Q(3, 'km')}, chart=Cart3D())

    """
    return cls.from_(u.Q(x, unit))  # ty: ignore[invalid-return-type]


# ============================================================================
# Simplification


@plum.dispatch
def simplify(op: AbstractAdd, /, **kw: Any) -> AbstractAdd | Identity:
    """Simplify a AbstractAdd operator.

    A translation with zero delta simplifies to Identity.

    Examples
    --------
    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Translate.from_([1, 2, 3], "km")
    >>> cxfm.simplify(op)
    Translate(...)

    >>> op = cxfm.Translate.from_([0, 0, 0], "km")
    >>> cxfm.simplify(op)
    Identity()

    """
    is_zero = jtu.all(
        jtu.map(lambda v: jnp.allclose(u.ustrip(AllowValue, v), 0, **kw), op.delta)
    )
    if is_zero:
        return identity
    return op
