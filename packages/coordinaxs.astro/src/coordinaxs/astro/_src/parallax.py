"""Parallax distance quantity."""

__all__ = ("Parallax",)

from dataclasses import KW_ONLY

from jaxtyping import Array, ArrayLike, Shaped
from typing import Any, final

import equinox as eqx
import plum

import quaxed.numpy as jnp
import unxt as u

import coordinax.distances as cxd
from .constants import ANGLE, LENGTH, MAGNITUDE

parallax_base_length = u.Q(jnp.array(1), "AU")


@final
class Parallax(cxd.AbstractDistance):
    """Parallax distance quantity.

    Examples
    --------
    >>> from coordinaxs.astro import Parallax
    >>> Parallax(1, "mas")
    Parallax(1, 'mas')

    The units are checked to have angle dimensions.

    >>> try: Parallax(1, "pc")
    ... except ValueError as e: print(e)
    Parallax must have angular dimensions.

    The parallax is checked to be non-negative by default.

    >>> try: Parallax(-1, "mas")
    ... except Exception: print("negative")
    negative

    To disable this check, set `check_negative=False`.

    >>> Parallax(-1, "mas", check_negative=False)
    Parallax(-1, 'mas', check_negative=False)

    """

    value: Shaped[Array, "*shape"] = eqx.field(
        converter=u.quantity.convert_to_quantity_value
    )
    """The value of the `unxt.AbstractQuantity`."""

    unit: u.AbstractUnit = eqx.field(static=True, converter=u.unit)  # ty: ignore[invalid-assignment]
    """The unit associated with this value."""

    _: KW_ONLY
    check_negative: bool = eqx.field(default=True, static=True, compare=False)
    """Whether to check that the parallax is strictly non-negative.

    Theoretically the parallax must be strictly non-negative ($\tan(p) = 1
    AU / d$), however noisy direct measurements of the parallax can be negative.
    """

    def __check_init__(self) -> None:
        """Check the initialization."""
        if u.dimension_of(self) != ANGLE:
            msg = "Parallax must have angular dimensions."
            raise ValueError(msg)

        if self.check_negative:
            # Store the checked value back so the guard survives jit (an
            # unused `error_if` result is dead-code-eliminated under trace).
            checked = eqx.error_if(
                self.value,
                jnp.any(jnp.less(self.value, 0)),
                "Parallax must be non-negative.",
            )
            object.__setattr__(self, "value", checked)


@Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Parallax], value: ArrayLike, unit: Any, /, **kw: Any) -> Parallax:
    """Construct a distance.

    >>> import unxt as u
    >>> from coordinaxs.astro import Parallax

    >>> Parallax.from_(1, "mas")
    Parallax(1, 'mas')

    """
    return cls(jnp.asarray(value, **kw), unit)


@Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Parallax], p: Parallax, /, **kw: Any) -> Parallax:
    """Compute parallax from parallax.

    >>> import unxt as u
    >>> from coordinaxs.astro import Parallax

    >>> p = Parallax(1, "mas")
    >>> Parallax.from_(p) is p
    True

    >>> Parallax.from_(p, dtype=float)
    Parallax(1., 'mas')

    """
    if len(kw) == 0:
        return p
    return jnp.asarray(p, **kw)  # ty: ignore[invalid-return-type]


def _from_angle(cls: type[Parallax], q: u.AbstractQuantity, /, **kw: Any) -> Parallax:
    """Parallax from an angle (already a parallax angle)."""
    unit = u.unit_of(q)
    return cls(jnp.asarray(q.ustrip(unit), **kw), unit)


def _from_length(cls: type[Parallax], q: u.AbstractQuantity, /, **kw: Any) -> Parallax:
    """Parallax from a length (distance)."""
    p = jnp.atan2(parallax_base_length, q)
    # atan2(1 AU, d) lies in [0, pi] for any d -- never negative, so the
    # guard cannot fire. Closed at 0: d = +inf gives exactly 0.
    return cls._make(jnp.asarray(p.value, **kw), p.unit)  # ty: ignore[unresolved-attribute]


def _from_mag(cls: type[Parallax], q: u.AbstractQuantity, /, **kw: Any) -> Parallax:
    """Parallax from a magnitude (distance modulus)."""
    d = u.Q(10 ** (1 + q.ustrip("mag") / 5), "pc")
    p = jnp.atan2(parallax_base_length, d)
    unit = u.unit_of(p)
    # d = 10**x >= 0, so atan2(1 AU, d) is in [0, pi/2] -- never negative, so
    # the guard cannot fire. Both endpoints are reachable: d underflows to 0
    # (-> pi/2) for very negative dm, overflows to +inf (-> 0) for large dm.
    return cls._make(jnp.asarray(p.ustrip(unit), **kw), unit)  # ty: ignore[unresolved-attribute]


@Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Parallax], q: u.AbstractQuantity, /, **kw: Any) -> Parallax:
    """Construct a parallax from a quantity, dispatching on its dimensions.

    From a parallax angle:

    >>> import unxt as u
    >>> import coordinax.distances as cxd
    >>> from coordinaxs.astro import Parallax

    >>> q = u.Q(1, "mas")
    >>> Parallax.from_(q, dtype=float)
    Parallax(1., 'mas')

    From a distance:

    >>> d = cxd.Distance(10, "pc")
    >>> Parallax.from_(d).uconvert("mas").round(2)
    Parallax(100., 'mas')

    >>> q = u.Q(10, "pc")
    >>> Parallax.from_(q).uconvert("mas").round(2)
    Parallax(100., 'mas')

    From a distance modulus:

    >>> dm = u.Q(10, "mag")
    >>> Parallax.from_(dm).uconvert("mas").round(2)
    Parallax(1., 'mas')

    """
    dim = u.dimension_of(q)
    if dim == ANGLE:  # already a parallax angle
        return _from_angle(cls, q, **kw)
    if dim == LENGTH:  # distance
        return _from_length(cls, q, **kw)
    if dim == MAGNITUDE:  # distance modulus
        return _from_mag(cls, q, **kw)
    msg = f"cannot build a Parallax from a quantity with dimension {dim}"
    raise ValueError(msg)


# When the optional ``unxts.parametric`` package is installed, also register
# static type-dispatched overloads on its parametric ``Quantity`` classes: a
# ``ParametricQuantity["length"|"angle"|"mag"]`` is routed by type (plum prefers
# these over the ``AbstractQuantity`` catch-all above). Plain ``unxt.Quantity``
# and other ``AbstractQuantity`` subclasses still fall through to the
# dimension-branching dispatch. If the package is not installed this is a no-op.
try:
    from unxts.parametric import PQ
except ImportError:
    pass
else:

    @Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
    def from_(cls: type[Parallax], q: PQ["angle"], /, **kw: Any) -> Parallax:
        """Construct a parallax from a parametric angle quantity."""
        return _from_angle(cls, q, **kw)

    @Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
    def from_(cls: type[Parallax], q: PQ["length"], /, **kw: Any) -> Parallax:
        """Construct a parallax from a parametric length (distance) quantity."""
        return _from_length(cls, q, **kw)

    @Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
    def from_(cls: type[Parallax], q: PQ["mag"], /, **kw: Any) -> Parallax:
        """Construct a parallax from a parametric magnitude quantity."""
        return _from_mag(cls, q, **kw)


@cxd.Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[cxd.Distance], p: Parallax, /, **kw: Any) -> cxd.Distance:
    """Compute distance from parallax.

    >>> import coordinax.distances as cxd
    >>> from coordinaxs.astro import Parallax

    >>> p = Parallax(1, "mas")
    >>> cxd.Distance.from_(p).uconvert("pc").round(2)
    Distance(1000., 'pc')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    unit = u.unit_of(d)
    return cls(jnp.asarray(d.ustrip(unit), **kw), unit)


@plum.dispatch
def dimension_of(obj: type[Parallax], /) -> u.AbstractDimension:
    """Return the angle dimension: a parallax is an angle, not a length.

    `coordinax.distances` registers `dimension_of` for
    `type[AbstractDistance]` returning length, which is right for
    `AbstractDistance` itself and for `Distance` but wrong for this subclass --
    every `Parallax` *instance* reports angle, so the class-level answer has to
    agree with it.

    >>> import unxt as u
    >>> from coordinaxs.astro import Parallax
    >>> u.dimension_of(Parallax)
    PhysicalType('angle')

    Which is what the instances say:

    >>> u.dimension_of(Parallax(1, "mas"))
    PhysicalType('angle')

    """
    return ANGLE
