"""Distance quantities."""

__all__ = ("Distance",)

from dataclasses import KW_ONLY

from jaxtyping import Array, ArrayLike, Shaped
from typing import Any, cast, final

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractDistance
from .constants import ANGLE, LENGTH, MAGNITUDE

parallax_base_length = u.Q(jnp.array(1), "AU")


@final
class Distance(AbstractDistance):
    """Distance quantities.

    The distance is a quantity with dimensions of length.

    Examples
    --------
    >>> import coordinax.distances as cxd
    >>> cxd.Distance(10, "km")
    Distance(10, 'km')

    The units are checked to have length dimensions.

    >>> try: cxd.Distance(10, "s")
    ... except ValueError as e: print(e)
    Distance must have dimensions length.

    """

    value: Shaped[Array, "*shape"] = eqx.field(
        converter=u.quantity.convert_to_quantity_value
    )
    """The distance value."""

    unit: u.AbstractUnit = eqx.field(static=True, converter=u.unit)  # ty: ignore[invalid-assignment]
    """The unit associated with this value."""

    _: KW_ONLY
    check_negative: bool = eqx.field(default=True, static=True, compare=False)
    """Whether to check that the distance is strictly non-negative."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if u.dimension_of(self) != LENGTH:
            msg = "Distance must have dimensions length."
            raise ValueError(msg)

        if self.check_negative:
            # Store the checked value back so the guard survives jit (an
            # unused `error_if` result is dead-code-eliminated under trace).
            checked = eqx.error_if(
                self.value,
                jnp.any(jnp.less(self.value, 0)),
                "Distance must be non-negative.",
            )
            object.__setattr__(self, "value", checked)


@Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Distance], value: ArrayLike, unit: Any, /, **kw: Any) -> Distance:
    """Construct a distance.

    >>> import unxt as u
    >>> import coordinax.distances as cxd
    >>> cxd.Distance.from_(1, "kpc")
    Distance(1, 'kpc')

    """
    return cls(jnp.asarray(value, **kw), unit)


@Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Distance], d: Distance, /, **kw: Any) -> Distance:
    """Compute distance from distance.

    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> d = cxd.Distance(1, "kpc")
    >>> cxd.Distance.from_(d) is d
    True

    >>> cxd.Distance.from_(d, dtype=float)
    Distance(1., 'kpc')

    """
    if len(kw) == 0:
        return d
    return cast("Distance", jnp.asarray(d, **kw))


def _from_length(cls: type[Distance], q: u.AbstractQuantity, /, **kw: Any) -> Distance:
    """Distance from a length quantity."""
    unit = u.unit_of(q)
    return cls(jnp.asarray(q.ustrip(unit), **kw), unit)


def _from_angle(cls: type[Distance], q: u.AbstractQuantity, /, **kw: Any) -> Distance:
    """Distance from an angle (parallax)."""
    d = parallax_base_length / jnp.tan(q)  # [AU]
    unit = u.unit_of(d)
    return cls(jnp.asarray(d.ustrip(unit), **kw), unit)


def _from_mag(cls: type[Distance], q: u.AbstractQuantity, /, **kw: Any) -> Distance:
    """Distance from a magnitude (distance modulus)."""
    d = 10 ** (1 + q.ustrip("mag") / 5)
    # The guard is free here: 10**x lowers to exp(x*ln10), and XLA folds
    # `exp(...) < 0` to false, eliminating the check entirely.
    return cls(jnp.asarray(d, **kw), "pc")


@Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Distance], q: u.AbstractQuantity, /, **kw: Any) -> Distance:
    """Construct a distance from a quantity, dispatching on its dimensions.

    From a length quantity:

    >>> import unxt as u
    >>> import coordinax.distances as cxd
    >>> q = u.Q(1, "kpc")
    >>> cxd.Distance.from_(q, dtype=float)
    Distance(1., 'kpc')

    From a parallax angle:

    >>> q = u.Q(1, "mas")
    >>> cxd.Distance.from_(q).uconvert("pc").round(2)
    Distance(1000., 'pc')

    From a distance modulus:

    >>> q = u.Q(10, "mag")
    >>> cxd.Distance.from_(q).uconvert("pc").round(2)
    Distance(1000., 'pc')

    """
    dim = u.dimension_of(q)
    if dim == LENGTH:
        return _from_length(cls, q, **kw)
    if dim == ANGLE:  # parallax
        return _from_angle(cls, q, **kw)
    if dim == MAGNITUDE:  # distance modulus
        return _from_mag(cls, q, **kw)
    msg = f"cannot build a Distance from a quantity with dimension {dim}"
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

    @Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
    def from_(cls: type[Distance], q: PQ["length"], /, **kw: Any) -> Distance:
        """Construct a distance from a parametric length quantity."""
        return _from_length(cls, q, **kw)

    @Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
    def from_(cls: type[Distance], q: PQ["angle"], /, **kw: Any) -> Distance:
        """Construct a distance from a parametric angle (parallax) quantity."""
        return _from_angle(cls, q, **kw)

    @Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
    def from_(cls: type[Distance], q: PQ["mag"], /, **kw: Any) -> Distance:
        """Construct a distance from a parametric magnitude quantity."""
        return _from_mag(cls, q, **kw)
