"""Distance quantities."""

__all__ = ("Distance", "DistanceModulus", "Parallax")

from dataclasses import KW_ONLY

from jaxtyping import Array, ArrayLike, Shaped
from typing import Any, Union, final

import equinox as eqx
import jax.numpy as jnp
import wadler_lindig as wl  # type: ignore[import-untyped]

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity

from .base import AbstractDistance
from coordinax._src.constants import ANGLE, LENGTH

parallax_base_length = u.Q(jnp.array(1), "AU")
distance_modulus_base_distance = u.Q(jnp.array(10), "pc")


@final
class Distance(AbstractDistance):
    """Distance quantities.

    The distance is a quantity with dimensions of length.

    Examples
    --------
    >>> from coordinax.distances import Distance
    >>> Distance(10, "km")
    Distance(Array(10, dtype=int64, ...), unit='km')

    The units are checked to have length dimensions.

    >>> try: Distance(10, "s")
    ... except ValueError as e: print(e)
    Distance must have dimensions length.

    """

    value: Shaped[Array, "*shape"] = eqx.field(
        converter=u.quantity.convert_to_quantity_value
    )
    """The distance value."""

    unit: u.AbstractUnit = eqx.field(static=True, converter=u.unit)
    """The unit associated with this value."""

    _: KW_ONLY
    check_negative: bool = eqx.field(default=True, static=True, compare=False)
    """Whether to check that the distance is strictly non-negative."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if u.dimension_of(self) != LENGTH:
            msg = "Distance must have dimensions length."
            raise ValueError(msg)

        if self.check_negative:  # pylint: disable=unreachable
            eqx.error_if(
                self.value,
                jnp.any(jnp.less(self.value, 0)),
                "Distance must be non-negative.",
            )

    def __pdoc__(self, **kwargs: Any) -> wl.AbstractDoc:
        """Return a Wadler-Lindig document for the parallax."""
        # Use the default __pdoc__ method to get the base document.
        pdoc = super().__pdoc__(**kwargs)

        # TODO: enable filtering in AbstractQuantity.__pdoc__ to avoid this.
        # Don't show check_negative if it's the default.
        fs = pdoc.children[2].child.child.children
        if fs[-1].children[-1].text == str(self.__class__.check_negative):
            object.__setattr__(pdoc.children[2].child.child, "children", fs[:-2])

        return pdoc


@Distance.from_.dispatch
def from_(cls: type[Distance], value: ArrayLike, unit: Any, /, **kw: Any) -> Distance:
    """Construct a distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> cxd.Distance.from_(1, "kpc")
    Distance(Array(1, dtype=int64, ...), unit='kpc')

    """
    return Distance(jnp.asarray(value, **kw), unit)


@Distance.from_.dispatch
def from_(cls: type[Distance], d: Distance, /, **kw: Any) -> Distance:
    """Compute distance from distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> d = cxd.Distance(1, "kpc")
    >>> cxd.Distance.from_(d) is d
    True

    >>> cxd.Distance.from_(d, dtype=float)
    Distance(Array(1., dtype=float64), unit='kpc')

    """
    if len(kw) == 0:
        return d
    return jnp.asarray(d, **kw)


@Distance.from_.dispatch
def from_(cls: type[Distance], d: u.Q["length"], /, **kw: Any) -> Distance:  # type: ignore[type-arg]
    """Compute distance from distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> q = u.Q(1, "kpc")
    >>> cxd.Distance.from_(q, dtype=float)
    Distance(Array(1., dtype=float64), unit='kpc')

    """
    unit = u.unit_of(d)
    return Distance(jnp.asarray(d.ustrip(unit), **kw), unit)


@Distance.from_.dispatch
def from_(
    cls: type[Distance], p: Union["Parallax", u.Q["angle"]], /, **kw: Any
) -> Distance:  # type: ignore[type-arg]
    """Compute distance from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> p = cxd.Parallax(1, "mas")
    >>> cxd.Distance.from_(p).uconvert("pc").round(2)
    Distance(Array(1000., dtype=float64, ...), unit='pc')

    >>> q = u.Q(1, "mas")
    >>> cxd.Distance.from_(q).uconvert("pc").round(2)
    Distance(Array(1000., dtype=float64, ...), unit='pc')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    unit = u.unit_of(d)
    return Distance(jnp.asarray(d.ustrip(unit), **kw), unit)


@Distance.from_.dispatch
def from_(
    cls: type[Distance], dm: Union["DistanceModulus", u.Q["mag"]], /, **kw: Any
) -> Distance:  # type: ignore[type-arg]
    """Compute distance from distance modulus.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> dm = cxd.DistanceModulus(10, "mag")
    >>> cxd.Distance.from_(dm).uconvert("pc").round(2)
    Distance(Array(1000., dtype=float64, ...), unit='pc')

    >>> q = u.Q(10, "mag")
    >>> cxd.Distance.from_(q).uconvert("pc").round(2)
    Distance(Array(1000., dtype=float64, ...), unit='pc')

    """
    d = 10 ** (1 + dm.ustrip("mag") / 5)
    return Distance(jnp.asarray(d, **kw), "pc")


##############################################################################


@final
class DistanceModulus(AbstractDistance):
    """Distance modulus quantity.

    Examples
    --------
    >>> from coordinax.distances import DistanceModulus
    >>> DistanceModulus(10, "mag")
    DistanceModulus(Array(10, dtype=int64, ...), unit='mag')

    The units are checked to have magnitude dimensions.

    >>> try: DistanceModulus(10, "pc")
    ... except ValueError as e: print(e)
    Distance modulus must have units of magnitude.

    """

    value: Shaped[Array, "*shape"] = eqx.field(
        converter=u.quantity.convert_to_quantity_value
    )
    """The value of the `unxt.AbstractQuantity`."""

    unit: u.AbstractUnit = eqx.field(static=True, converter=u.unit)
    """The unit associated with this value."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if self.unit != u.unit("mag"):
            msg = "Distance modulus must have units of magnitude."
            raise ValueError(msg)


@DistanceModulus.from_.dispatch
def from_(
    cls: type[DistanceModulus], value: ArrayLike, unit: Any, /, **kw: Any
) -> DistanceModulus:
    """Construct a distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> cxd.DistanceModulus.from_(1, "mag")
    DistanceModulus(Array(1, dtype=int64, ...), unit='mag')

    """
    return DistanceModulus(jnp.asarray(value, **kw), unit)


@DistanceModulus.from_.dispatch
def from_(
    cls: type[DistanceModulus], dm: DistanceModulus, /, **kw: Any
) -> DistanceModulus:
    """Compute distance modulus from distance modulus.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> dm = cxd.DistanceModulus(1, "mag")
    >>> cxd.DistanceModulus.from_(dm) is dm
    True

    >>> cxd.DistanceModulus.from_(dm, dtype=float)
    DistanceModulus(Array(1., dtype=float64), unit='mag')

    """
    if len(kw) == 0:
        return dm
    return jnp.asarray(dm, **kw)


@DistanceModulus.from_.dispatch
def from_(cls: type[DistanceModulus], dm: u.Q["mag"], /, **kw: Any) -> DistanceModulus:  # type: ignore[type-arg]
    """Compute parallax from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> q = u.Q(1, "mag")
    >>> cxd.DistanceModulus.from_(q)
    DistanceModulus(Array(1, dtype=int64, ...), unit='mag')

    """
    unit = u.unit_of(dm)
    return DistanceModulus(jnp.asarray(u.ustrip(unit, dm), **kw), unit)


@DistanceModulus.from_.dispatch
def from_(
    cls: type[DistanceModulus], d: Distance | u.Q["length"], /, **kw: Any
) -> DistanceModulus:  # type: ignore[type-arg]
    """Compute distance modulus from distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> d = cxd.Distance(1, "pc")
    >>> cxd.DistanceModulus.from_(d)
    DistanceModulus(Array(-5., dtype=float64), unit='mag')

    >>> q = u.Q(1, "pc")
    >>> cxd.DistanceModulus.from_(q)
    DistanceModulus(Array(-5., dtype=float64), unit='mag')

    """
    dm = 5 * jnp.log10(d.ustrip("pc")) - 5
    return DistanceModulus(jnp.asarray(dm, **kw), "mag")


@DistanceModulus.from_.dispatch
def from_(
    cls: type[DistanceModulus], p: Union["Parallax", u.Q["angle"]], /, **kw: Any
) -> DistanceModulus:  # type: ignore[type-arg]
    """Compute distance modulus from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> p = cxd.Parallax(1, "mas")
    >>> cxd.DistanceModulus.from_(p)
    DistanceModulus(Array(10., dtype=float64), unit='mag')

    >>> q = u.Q(1, "mas")
    >>> cxd.DistanceModulus.from_(q)
    DistanceModulus(Array(10., dtype=float64), unit='mag')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    dm = 5 * jnp.log10(d.ustrip("pc")) - 5
    return DistanceModulus(jnp.asarray(dm, **kw), "mag")


##############################################################################


@final
class Parallax(AbstractDistance):
    """Parallax distance quantity.

    Examples
    --------
    >>> from coordinax.distances import Parallax
    >>> Parallax(1, "mas")
    Parallax(Array(1, dtype=int64, ...), unit='mas')

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
    Parallax(Array(-1, dtype=int64, ...), unit='mas', check_negative=False)

    """

    value: Shaped[Array, "*shape"] = eqx.field(
        converter=u.quantity.convert_to_quantity_value
    )
    """The value of the `unxt.AbstractQuantity`."""

    unit: u.AbstractUnit = eqx.field(static=True, converter=u.unit)
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

        if self.check_negative:  # pylint: disable=unreachable
            eqx.error_if(
                self.value,
                jnp.any(jnp.less(self.value, 0)),
                "Parallax must be non-negative.",
            )

    def __pdoc__(self, **kwargs: Any) -> wl.AbstractDoc:
        """Return a Wadler-Lindig document for the parallax."""
        # Use the default __pdoc__ method to get the base document.
        pdoc = super().__pdoc__(**kwargs)

        # TODO: enable filtering in AbstractQuantity.__pdoc__ to avoid this.
        # Don't show check_negative if it's the default.
        fs = pdoc.children[2].child.child.children
        if fs[-1].children[-1].text == str(self.__class__.check_negative):
            object.__setattr__(pdoc.children[2].child.child, "children", fs[:-2])

        return pdoc


@Parallax.from_.dispatch
def from_(cls: type[Parallax], value: ArrayLike, unit: Any, /, **kw: Any) -> Parallax:
    """Construct a distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> cxd.Parallax.from_(1, "mas")
    Parallax(Array(1, dtype=int64, ...), unit='mas')

    """
    return Parallax(jnp.asarray(value, **kw), unit)


@Parallax.from_.dispatch
def from_(cls: type[Parallax], p: Parallax, /, **kw: Any) -> Parallax:
    """Compute parallax from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> p = cxd.Parallax(1, "mas")
    >>> cxd.Parallax.from_(p) is p
    True

    >>> cxd.Parallax.from_(p, dtype=float)
    Parallax(Array(1., dtype=float64), unit='mas')

    """
    if len(kw) == 0:
        return p
    return jnp.asarray(p, **kw)


@Parallax.from_.dispatch
def from_(cls: type[Parallax], p: u.Q["angle"], /, **kw: Any) -> Parallax:  # type: ignore[type-arg]
    """Compute parallax from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> q = u.Q(1, "mas")
    >>> cxd.Parallax.from_(q, dtype=float)
    Parallax(Array(1., dtype=float64), unit='mas')

    """
    unit = u.unit_of(p)
    return Parallax(jnp.asarray(p.ustrip(unit), **kw), unit)


@Parallax.from_.dispatch
def from_(cls: type[Parallax], d: Distance | u.Q["length"], /, **kw: Any) -> Parallax:  # type: ignore[type-arg]
    """Compute parallax from distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> d = cxd.Distance(10, "pc")
    >>> cxd.Parallax.from_(d).uconvert("mas").round(2)
    Parallax(Array(100., dtype=float64, ...), unit='mas')

    >>> q = u.Q(10, "pc")
    >>> cxd.Parallax.from_(q).uconvert("mas").round(2)
    Parallax(Array(100., dtype=float64, ...), unit='mas')

    """
    p = jnp.atan2(parallax_base_length, d)
    return Parallax(jnp.asarray(p.value, **kw), p.unit)


@Parallax.from_.dispatch
def from_(
    cls: type[Parallax], dm: DistanceModulus | u.Q["mag"], /, **kw: Any
) -> Parallax:  # type: ignore[type-arg]
    """Convert distance modulus to parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> dm = cxd.DistanceModulus(10, "mag")
    >>> cxd.Parallax.from_(dm).uconvert("mas").round(2)
    Parallax(Array(1., dtype=float64, ...), unit='mas')

    """
    d = BareQuantity(10 ** (1 + dm.ustrip("mag") / 5), "pc")
    p = jnp.atan2(parallax_base_length, d)
    unit = u.unit_of(p)
    return Parallax(jnp.asarray(p.ustrip(unit), **kw), unit)
