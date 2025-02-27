"""Coordinax package."""

from plum import add_promotion_rule

import unxt as u

from .angles import AbstractAngle
from .distances import AbstractDistance

# Add a rule that when a AbstractDistance interacts with a Quantity, the
# distance degrades to a Quantity. This is necessary for many operations, e.g.
# division of a distance by non-dimensionless quantity where the resulting units
# are not those of a distance.
add_promotion_rule(AbstractDistance, AbstractAngle, u.Quantity)
