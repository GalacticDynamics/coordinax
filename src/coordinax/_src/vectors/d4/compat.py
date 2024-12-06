"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method, convert

import quaxed.numpy as jnp
import unxt as u

from .spacetime import FourVector


@conversion_method(type_from=FourVector, type_to=u.Quantity)  # type: ignore[misc]
def vec_to_q(obj: FourVector, /) -> Shaped[u.Quantity["length"], "*batch 4"]:
    """`coordinax.AbstractPos3D` -> `unxt.Quantity`."""
    cart = convert(obj.q, u.Quantity)
    return jnp.concat([obj.c * obj.t[..., None], cart], axis=-1)
