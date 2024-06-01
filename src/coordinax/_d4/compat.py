"""Compatibility via :func:`plum.convert`."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.array_api as xp
from unxt import Quantity

from .spacetime import FourVector


@conversion_method(type_from=FourVector, type_to=Quantity)  # type: ignore[misc]
def vec_to_q(obj: FourVector, /) -> Shaped[Quantity["length"], "*batch 4"]:
    """`coordinax.AbstractPosition3D` -> `unxt.Quantity`."""
    cart = convert(obj.q, Quantity)
    return xp.concat([obj.c * obj.t[..., None], cart], axis=-1)
