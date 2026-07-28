"""Distance quantities."""

__all__ = ("DistanceModulus",)


from jaxtyping import Array, ArrayLike, Shaped
from typing import Any, final

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u

import coordinax.distances as cxd
from .constants import ANGLE, LENGTH, MAGNITUDE

parallax_base_length = u.Q(jnp.array(1), "AU")


@final
class DistanceModulus(cxd.AbstractDistance):
    """Distance modulus quantity.

    Examples
    --------
    >>> from coordinaxs.astro import DistanceModulus
    >>> DistanceModulus(10, "mag")
    DistanceModulus(10, 'mag')

    The units are checked to have magnitude dimensions.

    >>> try: DistanceModulus(10, "pc")
    ... except ValueError as e: print(e)
    Distance modulus must have units of magnitude.

    """

    value: Shaped[Array, "*shape"] = eqx.field(
        converter=u.quantity.convert_to_quantity_value
    )
    """The value of the `unxt.AbstractQuantity`."""

    unit: u.AbstractUnit = eqx.field(static=True, converter=u.unit)  # ty: ignore[invalid-assignment]
    """The unit associated with this value."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if self.unit != u.unit("mag"):
            msg = "Distance modulus must have units of magnitude."
            raise ValueError(msg)


@DistanceModulus.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[DistanceModulus], value: ArrayLike, unit: Any, /, **kw: Any
) -> DistanceModulus:
    """Construct a distance.

    >>> import unxt as u
    >>> from coordinaxs.astro import DistanceModulus

    >>> DistanceModulus.from_(1, "mag")
    DistanceModulus(1, 'mag')

    """
    return cls(jnp.asarray(value, **kw), unit)


@DistanceModulus.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[DistanceModulus], dm: DistanceModulus, /, **kw: Any
) -> DistanceModulus:
    """Compute distance modulus from distance modulus.

    >>> import unxt as u
    >>> from coordinaxs.astro import DistanceModulus

    >>> dm = DistanceModulus(1, "mag")
    >>> DistanceModulus.from_(dm) is dm
    True

    >>> DistanceModulus.from_(dm, dtype=float)
    DistanceModulus(1., 'mag')

    """
    if len(kw) == 0:
        return dm
    return jnp.asarray(dm, **kw)  # ty: ignore[invalid-return-type]


@DistanceModulus.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[DistanceModulus], q: u.AbstractQuantity, /, **kw: Any
) -> DistanceModulus:
    """Construct a distance modulus from a quantity, dispatching on dimensions.

    From a distance:

    >>> import unxt as u
    >>> import coordinax.distances as cxd
    >>> from coordinaxs.astro import DistanceModulus

    >>> d = cxd.Distance(1, "pc")
    >>> DistanceModulus.from_(d)
    DistanceModulus(-5., 'mag')

    >>> q = u.Q(1, "pc")
    >>> DistanceModulus.from_(q)
    DistanceModulus(-5., 'mag')

    From a parallax angle:

    >>> q = u.Q(1, "mas")
    >>> DistanceModulus.from_(q)
    DistanceModulus(10., 'mag')

    From a distance modulus (magnitude):

    >>> q = u.Q(1, "mag")
    >>> DistanceModulus.from_(q)
    DistanceModulus(1, 'mag')

    """
    dim = u.dimension_of(q)

    if dim == LENGTH:  # distance
        dm = 5 * jnp.log10(q.ustrip("pc")) - 5
        return cls(jnp.asarray(dm, **kw), "mag")

    if dim == ANGLE:  # parallax
        d = parallax_base_length / jnp.tan(q)  # [AU]
        dm = 5 * jnp.log10(d.ustrip("pc")) - 5
        return cls(jnp.asarray(dm, **kw), "mag")

    if dim == MAGNITUDE:  # already a distance modulus (magnitude)
        unit = u.unit_of(q)
        return cls(jnp.asarray(u.ustrip(unit, q), **kw), unit)

    msg = f"cannot build a DistanceModulus from a quantity with dimension {dim}"
    raise ValueError(msg)


@cxd.Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[cxd.Distance],
    dm: DistanceModulus,
    /,
    **kw: Any,
) -> cxd.Distance:
    """Compute distance from distance modulus.

    >>> import coordinax.distances as cxd
    >>> from coordinaxs.astro import DistanceModulus

    >>> dm = DistanceModulus(10, "mag")
    >>> cxd.Distance.from_(dm).uconvert("pc").round(2)
    Distance(1000., 'pc')

    """
    d = 10 ** (1 + dm.ustrip("mag") / 5)
    return cls(jnp.asarray(d, **kw), "pc")
