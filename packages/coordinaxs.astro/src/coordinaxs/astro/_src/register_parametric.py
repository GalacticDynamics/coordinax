"""`unxts.parametric` hooks; imported only when it is installed.

Two kinds: promotion rules, and the `from_` overloads that let plum pick a
branch from the *type* of a `ParametricQuantity` instead of reading
`u.dimension_of(q)` at runtime. Plain `unxt.Quantity` still takes the runtime
path in `parallax` and `distance_modulus`.
"""

__all__: tuple[str, ...] = ()

from typing import Any

from plum import add_promotion_rule

# Optional dependency: absent from the lint environment by design.
from unxts.parametric import PQ, ParametricQuantity

from .distance_modulus import (
    DistanceModulus,
    _from_angle as _dm_from_angle,
    _from_length as _dm_from_length,
    _from_mag as _dm_from_mag,
)
from .parallax import (
    Parallax,
    _from_angle as _plx_from_angle,
    _from_length as _plx_from_length,
    _from_mag as _plx_from_mag,
)

# Degrade to the parametric quantity, as the `AbstractDistance`/`Q` rules in
# `coordinax.distances` do. `ParametricQuantity` is not a `unxt.Q` subclass, so
# those rules never reach it.
add_promotion_rule(Parallax, ParametricQuantity, ParametricQuantity)
add_promotion_rule(DistanceModulus, ParametricQuantity, ParametricQuantity)


@Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Parallax], q: PQ["angle"], /, **kw: Any) -> Parallax:
    """Construct a parallax from a parametric angle quantity."""
    return _plx_from_angle(cls, q, **kw)


@Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Parallax], q: PQ["length"], /, **kw: Any) -> Parallax:
    """Construct a parallax from a parametric length (distance) quantity."""
    return _plx_from_length(cls, q, **kw)


@Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Parallax], q: PQ["mag"], /, **kw: Any) -> Parallax:
    """Construct a parallax from a parametric magnitude quantity."""
    return _plx_from_mag(cls, q, **kw)


@DistanceModulus.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[DistanceModulus], q: PQ["length"], /, **kw: Any) -> DistanceModulus:
    """Construct a distance modulus from a parametric length quantity."""
    return _dm_from_length(cls, q, **kw)


@DistanceModulus.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[DistanceModulus], q: PQ["angle"], /, **kw: Any) -> DistanceModulus:
    """Construct a distance modulus from a parametric angle (parallax)."""
    return _dm_from_angle(cls, q, **kw)


@DistanceModulus.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[DistanceModulus], q: PQ["mag"], /, **kw: Any) -> DistanceModulus:
    """Construct a distance modulus from a parametric magnitude quantity."""
    return _dm_from_mag(cls, q, **kw)
