"""Promotion rules for `unxts.parametric`; imported only when it is installed."""

__all__: tuple[str, ...] = ()

from plum import add_promotion_rule

from unxts.parametric import ParametricQuantity

from .distance_modulus import DistanceModulus
from .parallax import Parallax

# Degrade to the parametric quantity, as the `AbstractDistance`/`Q` rules in
# `coordinax.distances` do. `ParametricQuantity` is not a `unxt.Q` subclass, so
# those rules never reach it.
add_promotion_rule(Parallax, ParametricQuantity, ParametricQuantity)
add_promotion_rule(DistanceModulus, ParametricQuantity, ParametricQuantity)
