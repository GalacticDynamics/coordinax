"""Registrations that apply only when `unxts.parametric` is installed.

Imported from `coordinaxs.astro._src` behind
``OptDeps.UNXTS_PARAMETRIC.installed``, so nothing here may be imported
unconditionally: `unxts.parametric` is an optional extra
(``pip install "coordinaxs.astro[parametric]"``).
"""

__all__: tuple[str, ...] = ()

from plum import add_promotion_rule

from unxts.parametric import ParametricQuantity

from .distance_modulus import DistanceModulus
from .parallax import Parallax

# When a distance interacts with a `ParametricQuantity`, degrade to the
# parametric quantity -- the same reasoning as the `AbstractDistance`/`Q` rules
# in `coordinax.distances`: the result of e.g. dividing a parallax by a
# non-dimensionless quantity has units that are not a parallax's.
#
# `ParametricQuantity` is not a `unxt.Q` subclass, so the core rules do not
# reach it. Without these, `Parallax(1, "mas") * PQ(1.0, "rad")` dispatched to
# the `Parallax`-returning multiply and raised ``Parallax must have angular
# dimensions``, while the mirrored `PQ * Parallax` returned a `PQ` -- the
# operand order decided whether the expression worked.
add_promotion_rule(Parallax, ParametricQuantity, ParametricQuantity)
add_promotion_rule(DistanceModulus, ParametricQuantity, ParametricQuantity)
