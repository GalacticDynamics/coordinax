"""`from_` overloads for `unxts.parametric`; imported only when it is installed.

A `ParametricQuantity["length"|"angle"|"mag"]` carries its physical type in the
*type*, so plum can pick the branch statically and prefers these over the
`AbstractQuantity` catch-all in `measures`, which has to read
`u.dimension_of(q)` at runtime. Plain `unxt.Quantity` still takes that path.
"""

__all__: tuple[str, ...] = ()

from typing import Any

# Optional dependency: absent from the lint environment by design.
from unxts.parametric import PQ  # ty: ignore[unresolved-import]

from .measures import Distance, _from_angle, _from_length, _from_mag


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
