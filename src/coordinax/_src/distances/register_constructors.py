"""Distance constructors."""

__all__: list[str] = []

from typing import Any

import unxt as u

from .base import AbstractDistance
from .funcs import distance, distance_modulus, parallax
from .measures import Distance, DistanceModulus, Parallax

parallax_base_length = u.Quantity(1, "AU")
distance_modulus_base_distance = u.Quantity(10, "pc")


@u.AbstractQuantity.from_.dispatch
def from_(
    cls: type[Distance],
    obj: AbstractDistance
    | u.Quantity["length"]
    | u.Quantity["angle"]
    | u.Quantity["mag"],
    /,
    **kw: Any,
) -> Distance:
    """Construct a `Distance` from the inputs."""
    return distance(obj, **kw)


@u.AbstractQuantity.from_.dispatch
def from_(
    cls: type[DistanceModulus],
    dm: AbstractDistance
    | u.Quantity["mag"]
    | u.Quantity["length"]
    | u.Quantity["angle"],
    /,
    **kwargs: Any,
) -> DistanceModulus:
    """Construct a `DistanceModulus` from the input."""
    return distance_modulus(dm, **kwargs)


@u.AbstractQuantity.from_.dispatch
def from_(
    cls: type[Parallax],
    obj: AbstractDistance
    | u.Quantity["angle"]
    | u.Quantity["length"]
    | u.Quantity["mag"],
    /,
    **kwargs: Any,
) -> Parallax:
    """Construct a `Parallax` the input."""
    return parallax(obj, **kwargs)
